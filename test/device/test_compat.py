"""External registration and per-adapter compilation regression checks."""

import gc, json, os, struct, weakref
from unittest.mock import patch
import pytest
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from tinygrad_dx12.compat import split_dx12_calls


def kernel(devices: tuple[str, ...], device_var: bool = False) -> UOp:
    """Build a kernel whose result reveals its original tuple lane index."""
    out = UOp.param(0, dtypes.int32, 1, devices)
    lane = (
        UOp.variable("_device_num", 0, len(devices) - 1, dtype=dtypes.int32, param=True)
        if device_var
        else UOp.range(len(devices), 0, AxisType.DEVICE, dtype=dtypes.int32)
    )
    return out.index(0).store(lane).sink(arg=KernelInfo()).call(UOp.new_buffer(devices, 1, dtypes.int32))


def test_tuple_lanes_keep_adapter_ordinals() -> None:
    """Reordered nonzero adapter ordinals must not become lane indices."""
    calls = split_dx12_calls(UOp(Ops.LINEAR, src=(kernel(("DX12:4", "DX12:2")),))).src
    assert tuple(call.device for call in calls) == ("DX12:4", "DX12:2")
    assert tuple(call.src[0].src[0].src[1].val for call in calls) == (0, 1)
    assert tuple(call.src[1].arg for call in calls) == (0, 1)


def test_device_num_param_is_bound_before_compilation() -> None:
    """Already lowered device variables retain their original lane values."""
    calls = split_dx12_calls(UOp(Ops.LINEAR, src=(kernel(("DX12:2", "DX12"), device_var=True),))).src
    assert tuple(call.src[0].src[0].src[1].val for call in calls) == (0, 1)
    assert not any(u.op is Ops.PARAM and u.arg.name == "_device_num" for call in calls for u in call.src[0].toposort())


def test_copies_preserve_order_and_non_dx12_calls() -> None:
    """Splitting cannot move kernels across copies or affect other runtimes."""
    first, second = kernel(("DX12", "DX12:1")), kernel(("CL", "CL:1"))
    dest, source = UOp.new_buffer("DX12", 1, dtypes.int32), UOp.new_buffer("DX12:1", 1, dtypes.int32)
    copy = source.copy_to_device("DX12").call(dest, source)
    calls = split_dx12_calls(UOp(Ops.LINEAR, src=(first, copy, second, first))).src
    assert calls[2] is copy and calls[3] is second
    assert tuple(call.device for call in calls) == ("DX12", "DX12:1", "DX12", ("CL", "CL:1"), "DX12", "DX12:1")


def test_registration_is_idempotent() -> None:
    """Registering twice must preserve the wrapper and expose the package runtime."""
    from tinygrad_dx12 import register
    from tinygrad import schedule
    from tinygrad.engine import jit, realize
    from tinygrad.schedule import memory
    from tinygrad.device import Device
    from tinygrad_dx12.runtime.ops_dx12 import DX12Device

    register()
    lowering = realize.lower_and_compile
    planning = memory.memory_plan_rewrite
    jit_lowering = jit.jit_lower
    register()
    assert realize.lower_and_compile is lowering
    assert jit.jit_lower is jit_lowering
    assert schedule.memory_plan_rewrite is jit.memory_plan_rewrite is memory.memory_plan_rewrite is planning
    assert Device.get_class("DX12:2") is DX12Device


def test_view_geometry_and_cache_ownership() -> None:
    """Cached mixed-dtype geometry cannot retain either its view or base allocation UOp."""
    base = UOp.new_buffer("DX12", 32, dtypes.float16)
    view = base[3:9].bitcast(dtypes.uint8)
    for _ in range(2):
        assert view.contiguous_view() == (base, 3)
    assert base.flip(0).contiguous_view() is None
    original, derived = weakref.ref(base), weakref.ref(view)
    del base, view
    gc.collect()
    assert original() is None and derived() is None


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="set DX12_TEST_GPU=1 to check native storage replacement")
def test_view_geometry_resolves_replaced_storage() -> None:
    """Reallocating the base buffer must change native view storage despite cached geometry."""
    from tinygrad.device import Device

    base = UOp.new_buffer("DX12", 32, dtypes.uint8)
    view = base[3:9]
    buffer = base.buffer.ensure_allocated()
    buffer.allocator._copyin(buffer._buf, memoryview(bytes(range(32))))
    assert bytes(view.buffer.ensure_allocated().as_memoryview()) == bytes(range(3, 9))
    original = buffer._buf
    buffer.deallocate()
    Device["DX12"].allocator.free_cache()
    buffer.ensure_allocated()
    assert buffer._buf is not original
    buffer.allocator._copyin(buffer._buf, memoryview(bytes(range(32, 64))))
    assert bytes(view.buffer.ensure_allocated().as_memoryview()) == bytes(range(35, 41))


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="set DX12_TEST_GPU=1 to run per-adapter compilation")
def test_each_tuple_lane_uses_its_own_capability_target() -> None:
    """Different adapter capabilities must produce distinct programs and correct lane values."""
    from tinygrad_dx12 import register
    from tinygrad.device import Device
    from tinygrad.engine.realize import compile_linear, run_linear
    from tinygrad.helpers import Target
    from tinygrad_dx12.renderer.cstyle import DX12Renderer

    register()
    if Device["DX12"].count() < 2:
        pytest.skip("two adapters required")
    second = Device["DX12:1"]
    capabilities = {**second.renderer.capabilities, "native16": False, "doubles": False}
    reduced = DX12Renderer(Target(device="DX12", arch=json.dumps(capabilities, sort_keys=True)))
    call = kernel(("DX12:1", "DX12"))
    with patch.object(second, "_renderer", reduced):
        linear = compile_linear(UOp(Ops.LINEAR, src=(call,)))
        assert tuple(item.device for item in linear.src) == ("DX12:1", "DX12")
        assert linear.src[0].src[0].arg.target != linear.src[1].src[0].arg.target
        run_linear(linear)
        outputs = call.src[1].buffer.bufs
        assert [struct.unpack("<i", output.as_memoryview())[0] for output in outputs] == [0, 1]
