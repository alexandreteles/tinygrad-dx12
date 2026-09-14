"""Dependency, replay-lifetime, and hardware JIT regression checks."""

import ctypes as C
import os, struct, threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from unittest.mock import patch
import pytest
from tinygrad import Tensor, TinyJit
from tinygrad.device import DepsTracker
from tinygrad.engine.jit import GraphRunner
from tinygrad_dx12.runtime.graph.dx12 import DX12Graph, GraphSlot, barrier_access

UAV, SRV = 0x10, 0x80
gpu = pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="set DX12_TEST_GPU=1 to run native graph checks")


@dataclass
class Span:
    """A base-buffer range matching tinygrad's dependency tracker contract."""

    base: object
    offset: int
    nbytes: int


def test_overlapping_views_and_read_write_dependencies() -> None:
    """Only overlapping accesses produce hazards, including write-after-read."""
    runner = GraphRunner.__new__(GraphRunner)
    runner.deps = DepsTracker()
    base = object()
    assert runner._access_resources([Span(base, 0, 16)], [0], 0) == []
    assert runner._access_resources([Span(base, 16, 16)], [0], 1) == []
    dependencies = runner._access_resources([Span(base, 8, 16)], [], 2)
    assert set(dependencies) == {0, 1}
    covered = {}
    assert barrier_access(dependencies, [UAV, UAV], covered, SRV) == (UAV, SRV)
    assert barrier_access([0, 1], [UAV, UAV, SRV], covered, SRV) == (0, 0)
    dependencies = runner._access_resources([Span(base, 12, 4)], [0], 3)
    assert set(dependencies) == {0, 2}
    assert barrier_access(dependencies, [UAV, UAV, SRV], covered, UAV) == (UAV | SRV, UAV)


def test_uav_reads_use_uav_access_masks() -> None:
    """An RWByteAddressBuffer read cannot reuse SRV-only visibility coverage."""
    covered = {}
    assert barrier_access([0], [UAV], covered, SRV) == (UAV, SRV)
    assert barrier_access([0], [UAV], covered, UAV) == (UAV, UAV)
    assert barrier_access([], [UAV, SRV], covered, UAV) == (0, 0)


@dataclass
class Fences:
    """Controllable completion points for deterministic ring acquisition checks."""

    done: int
    waited: list[int] = field(default_factory=list)
    lock: object = field(default_factory=threading.RLock)

    def completed(self) -> int:
        """Read the current completion point without advancing it."""
        return self.done

    def wait(self, fence: int) -> None:
        """Advance only when the graph explicitly requests a fence wait."""
        self.waited.append(fence)
        self.done = fence

    def retire(self) -> int:
        """Return the retirement snapshot without advancing native resources."""
        return self.done


def ring(fences: tuple[int, ...], done: int, start: int = 0) -> DX12Graph:
    """Build only the ring state used by the actual acquisition implementation."""
    graph = DX12Graph.__new__(DX12Graph)
    graph.dev, graph.next_slot = Fences(done), start
    graph.slots = [GraphSlot.__new__(GraphSlot) for _ in fences]
    for slot, fence in zip(graph.slots, fences):
        slot.fence = fence
    return graph


def test_slot_ring_skips_active_memory() -> None:
    """A completed later slot is preferred over waiting for the current slot."""
    graph = ring((9, 3, 7), done=4)
    assert graph._acquire() is graph.slots[1]
    assert graph.dev.waited == []
    assert tuple(slot.fence for slot in graph.slots) == (9, 3, 7)


def test_busy_slot_ring_waits_oldest_fence() -> None:
    """When all slots are busy, the oldest completion point is sufficient."""
    graph = ring((12, 10, 15), done=8, start=2)
    assert graph._acquire() is graph.slots[1]
    assert graph.dev.waited == [10]


def staged(x: Tensor) -> Tensor:
    """Materialize two kernels so graph dependencies are observable in results."""
    return ((x + 3).realize() * 2).realize()


@gpu
def test_jit_changes_inputs_and_preserves_queued_outputs() -> None:
    """More replays than slots must preserve each queued result without caller waits."""
    from tinygrad_dx12 import register

    register()
    run = TinyJit(staged)
    results = [run(Tensor([i, i + 1, i + 2], device="DX12")).clone().realize() for i in range(11)]
    assert [result.tolist() for result in results] == [[2 * (i + 3), 2 * (i + 4), 2 * (i + 5)] for i in range(11)]
    assert any(call.src[0].op.name == "CUSTOM_FUNCTION" and call.src[0].arg == "graph" for call in run.captured.linear.src)


@gpu
def test_graph_switches_optional_timing_modes() -> None:
    """Timing mode changes preserve outputs and profiles without rebuilding stable replays."""
    from tinygrad_dx12 import register
    from tinygrad.device import Device, ProfileGraphEvent
    from tinygrad.engine.realize import get_graph_runtime
    from tinygrad.helpers import Context

    register()
    run = TinyJit(staged)
    first = Tensor([0] * 16, device="DX12").realize()
    run(first)
    output = run(first)
    graph = next(get_graph_runtime(call.src[0]) for call in run.captured.linear.src if call.src[0].arg == "graph")
    dev = Device["DX12"]
    dev.synchronize()
    assert all(slot.timing is None for slot in graph.slots)
    profile_start = len(dev.profile_events)
    with patch.object(graph, "_record", wraps=graph._record) as record:
        for mode, (wait, profile) in enumerate(((False, 0), (True, 0), (False, 1), (False, 0), (True, 0))):
            for replay in range(6):
                value = mode * 6 + replay
                x = Tensor([value] * 16, device="DX12").realize()
                with Context(PROFILE=profile):
                    elapsed = graph((x.uop,), {}, wait=wait)
                assert (elapsed is not None and elapsed > 0) if wait else elapsed is None
                assert output.tolist() == [2 * (value + 3)] * 16
                if replay == 2:
                    recorded = record.call_count
            assert record.call_count == recorded, "stable timing mode re-recorded an idle command list"
    dev.synchronize()
    profiles = [event for event in dev.profile_events[profile_start:] if isinstance(event, ProfileGraphEvent)]
    assert len(profiles) == 6
    assert all(len(event.sigs) == 3 and event.sigs[0] < event.sigs[1] < event.sigs[2] for event in profiles)


@gpu
def test_native_ring_does_not_patch_active_arguments() -> None:
    """A queue readiness fence makes three replays active without timing assumptions."""
    from tinygrad_dx12 import register
    from tinygrad.device import Device
    from tinygrad.engine.realize import get_graph_runtime
    from tinygrad_dx12.runtime.support.dx12 import check

    register()
    run = TinyJit(staged)
    inputs = [Tensor([i] * 16, device="DX12").realize() for i in range(4)]
    run(inputs[0])
    output = run(inputs[0])
    graph = next(get_graph_runtime(call.src[0]) for call in run.captured.linear.src if call.src[0].arg == "graph")
    dev = Device["DX12"]
    copies = [dev.buffer(64) for _ in range(3)]
    dev.synchronize()
    gate = dev.new_fence()
    check(dev.queue.call(15, C.c_int32, (C.c_void_p, C.c_uint64), gate.ptr, 1), "test queue readiness wait")
    reached_wait, original_wait = threading.Event(), dev.wait

    def wait(fence: int, timeout: int | None = None) -> None:
        """Expose when the fourth replay reaches a real pending fence."""
        reached_wait.set()
        original_wait(fence, timeout)

    try:
        for x, copy in zip(inputs, copies):
            graph((x.uop,), {})
            dev.allocator._copy(copy, output.uop.base.buffer._buf, 64)
        assert all(slot.fence > dev.completed() for slot in graph.slots)
        arguments = [bytes(slot.arguments.mapped) for slot in graph.slots]
        with patch.object(dev, "wait", wait), ThreadPoolExecutor(max_workers=1) as pool:
            replay = pool.submit(graph, (inputs[3].uop,), {})
            try:
                assert reached_wait.wait(15), "fourth replay did not wait for its active slot"
                assert [bytes(slot.arguments.mapped) for slot in graph.slots] == arguments
            finally:
                check(gate.call(10, C.c_int32, (C.c_uint64,), 1), "release test queue readiness")
            replay.result(timeout=15)
    finally:
        check(gate.call(10, C.c_int32, (C.c_uint64,), 1), "release test queue readiness")
    for i, copy in enumerate(copies):
        data = bytearray(64)
        dev.allocator._copyout(memoryview(data), copy)
        assert struct.unpack("<16i", data) == (2 * (i + 3),) * 16
    assert output.tolist() == [12] * 16


@gpu
def test_jit_nonzero_device_and_input_views() -> None:
    """Graph lookup and replacement views preserve the selected adapter ordinal."""
    from tinygrad_dx12 import register
    from tinygrad.device import Device
    from tinygrad.engine.realize import get_graph_runtime

    register()
    if Device["DX12"].count() < 2:
        pytest.skip("two adapters required")
    run = TinyJit(staged)
    for i in range(6):
        x = Tensor([0, i, i + 1, i + 2, 0], device="DX12:1").shrink(((1, 4),))
        assert run(x).tolist() == [2 * (i + 3), 2 * (i + 4), 2 * (i + 5)]
    graphs = [get_graph_runtime(call.src[0]) for call in run.captured.linear.src if call.src[0].arg == "graph"]
    assert graphs and all(graph.device == "DX12:1" for graph in graphs)


@gpu
def test_two_device_graph_copy_graph() -> None:
    """Immediate graph consumers observe transfers in both directions."""
    from tinygrad_dx12 import register
    from tinygrad.device import Device

    register()
    if Device["DX12"].count() < 2:
        pytest.skip("two adapters required")
    source, destination = TinyJit(staged), TinyJit(staged)
    results = []
    for i in range(7):
        first = source(Tensor([i, i + 1], device="DX12:1"))
        second = destination(first.to("DX12"))
        results.append(second.to("DX12:1").clone().realize())
    assert [result.tolist() for result in results] == [[4 * i + 18, 4 * i + 22] for i in range(7)]


@gpu
def test_descriptor_table_and_cbv_replay() -> None:
    """Root-signature overflow uses slot-local descriptors and CBV view offsets."""
    from tinygrad_dx12 import register
    from tinygrad.engine.realize import get_graph_runtime

    register()

    def many_inputs(*xs: Tensor) -> Tensor:
        """Force one kernel to bind enough independent buffers to require a table."""
        return (sum(xs).realize() * 2).realize()

    run = TinyJit(many_inputs)
    results = []
    for i in range(8):
        inputs = [Tensor([0, j + i, j + i + 1, 0], device="DX12").shrink(((1, 3),)) for j in range(33)]
        results.append(run(*inputs).clone().realize())
    assert [result.tolist() for result in results] == [[2 * (528 + 33 * i), 2 * (561 + 33 * i)] for i in range(8)]
    graphs = [get_graph_runtime(call.src[0]) for call in run.captured.linear.src if call.src[0].arg == "graph"]
    assert any(layout.table and layout.scalar_cbv for graph in graphs for layout in graph.layouts)


@gpu
def test_dynamic_local_sizes_scalars_groups_and_timestamps() -> None:
    """A changed local size must change numthreads, while scalar/groups patch records."""
    from tinygrad_dx12 import register
    from tinygrad.device import Device, ProfileGraphEvent
    from tinygrad.dtype import dtypes
    from tinygrad.helpers import Context
    from tinygrad.renderer import Estimates
    from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp
    from tinygrad_dx12.renderer.cstyle import BindingLayout, BufferBinding, ScalarBinding

    register()
    if Device["DX12"].count() < 2:
        pytest.skip("two adapters required")
    dev = Device["DX12:1"]
    variables = tuple(
        UOp.variable(name, lo, hi, dtype=dtypes.int32, param=True) for name, lo, hi in (("width", 1, 8), ("groups", 1, 4), ("value", 0, 100))
    )
    layout = BindingLayout(
        (BufferBinding(0, "data0", True, 0, 0, 0),),
        tuple(ScalarBinding(i + 1, v.expr, "int", i + 1, 1) for i, v in enumerate(variables)),
        4,
        1,
        False,
        False,
    )
    source = (
        "// DX12:"
        + layout.to_json()
        + "\n"
        + "\n".join([
            "#define DX12_LOCAL_X 1",
            "#define DX12_LOCAL_Y 1",
            "#define DX12_LOCAL_Z 1",
            "RWByteAddressBuffer data0 : register(u0);",
            "cbuffer Params : register(b0) { uint offset:packoffset(c0.x); uint width:packoffset(c0.y);",
            "uint groups:packoffset(c0.z); uint value:packoffset(c0.w); }",
            "[numthreads(DX12_LOCAL_X,DX12_LOCAL_Y,DX12_LOCAL_Z)]",
            "void main(uint3 g:SV_GroupID, uint3 l:SV_GroupThreadID) {",
            "uint i=g.x*DX12_LOCAL_X+l.x; data0.Store(offset+i*4,value+i); }",
        ])
    )
    param = UOp.param(0, dtypes.int32, 64, "DX12:1")
    info = ProgramInfo("dynamic_launch", (variables[1], 1, 1), (variables[0], 1, 1), variables, (0,), (0,), (), dev.renderer.target)
    prg = UOp(
        Ops.PROGRAM,
        src=(
            UOp(Ops.SINK, arg=KernelInfo(estimates=Estimates())),
            UOp(Ops.LINEAR, src=(param,)),
            UOp(Ops.SOURCE, arg=source),
            UOp(Ops.BINARY, arg=dev.compiler.compile_cached(source)),
        ),
        arg=info,
    )
    initial = Tensor([-999] * 64, device="DX12:1").realize()
    graph = DX12Graph(UOp(Ops.CUSTOM_FUNCTION, src=(UOp(Ops.LINEAR, src=(prg.call(param),)),), arg="graph"), (initial.uop,))
    profile_start = len(dev.profile_events)
    for i, width in enumerate((2, 4, 8, 1, 2, 8, 4)):
        output = Tensor([-999] * 64, device="DX12:1").realize()
        groups, value = i % 4 + 1, 10 + i
        with Context(PROFILE=1):
            elapsed = graph((output.uop,), {"width": width, "groups": groups, "value": value}, wait=True)
        assert elapsed is not None and elapsed > 0
        assert output.tolist() == list(range(value, value + width * groups)) + [-999] * (64 - width * groups)
    dev.synchronize()
    profiles = [event for event in dev.profile_events[profile_start:] if isinstance(event, ProfileGraphEvent)]
    assert len(profiles) == 7 and all(event.sigs[1] > event.sigs[0] for event in profiles)
