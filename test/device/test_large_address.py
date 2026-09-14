"""Exercise real GPU addresses above 4/96 GiB without requiring a 128 GiB GPU."""

import ctypes as C
import os, struct
from collections.abc import Iterator

import pytest
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Buffer, TinyELF
from tinygrad.dtype import DType
from tinygrad.uop.ops import AxisType, KernelInfo, UOp
from tinygrad_dx12.runtime.autogen import d3d12 as d
from tinygrad_dx12.runtime.ops_dx12 import DX12Buffer, create

pytestmark = pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires a DX12 adapter")
SPAN = (128 << 30) + 4096
POSITIONS = (0, (2 << 30) - 8, (4 << 30) - 8, 4 << 30, (96 << 30) - 8, 96 << 30, 128 << 30)


@pytest.fixture(scope="module")
def sparse_allocations() -> Iterator[tuple[DX12Buffer, ...]]:
    """Map distinct physical tiles at distant addresses in two real reserved buffers.

    These are native resources, not mocked addresses. Only the touched 64 KiB
    tiles consume memory. Heap ownership lasts through every submitted command.
    Production uses one heap with placed resources; these sparse aliases exercise
    the same shader/copy windows without allocating 128 GiB of physical memory.
    """
    device = Device["DX12"]
    device.synchronize()
    tiles = sorted({(position + extra) // 65536 for position in POSITIONS for extra in (0, 64)})
    heaps, buffers = [], []
    physical_span = (SPAN + 65535) & ~65535
    for _ in range(2):
        heap_desc = d.D3D12_HEAP_DESC(len(tiles) * 65536, d.D3D12_HEAP_PROPERTIES(1, 0, 0, 1, 1), 65536, 0xC0)
        heap = create(device.native, 28, d.IID_ID3D12Heap, (C.c_void_p,), C.byref(heap_desc))

        def reserved(start: int, size: int) -> DX12Buffer:
            """Map the matching physical tiles into a resource-relative window."""
            desc = d.D3D12_RESOURCE_DESC(1, 0, size, 1, 1, 1, 0, d.DXGI_SAMPLE_DESC(1, 0), 1, 4)
            resource = create(device.native, 30, d.IID_ID3D12Resource, (C.c_void_p, C.c_uint, C.c_void_p), C.byref(desc), 0, None)
            mappings = [(tile - start // 65536, i) for i, tile in enumerate(tiles) if start <= tile * 65536 < start + size]
            if mappings:
                count = len(mappings)
                coordinates = ((C.c_uint * 4) * count)(*[(C.c_uint * 4)(tile, 0, 0, 0) for tile, _ in mappings])
                offsets, counts = (C.c_uint * count)(*[i for _, i in mappings]), (C.c_uint * count)(*([1] * count))
                device.queue.call(
                    8,
                    None,
                    (C.c_void_p, C.c_uint, C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint, C.c_void_p, C.c_void_p, C.c_void_p, C.c_uint),
                    resource.ptr,
                    count,
                    coordinates,
                    None,
                    heap.ptr,
                    count,
                    None,
                    offsets,
                    counts,
                    0,
                )
            return DX12Buffer(resource, resource.call(11, C.c_uint64, ()), size, device)

        heaps.append(heap)
        buffer = reserved(0, physical_span)
        buffer.size = SPAN
        starts = sorted(set(range(0, SPAN, 1 << 30)) | {position & ~65535 for position in POSITIONS})
        aliases = {start: reserved(start, min(physical_span - start, (2 << 30) + 65536)) for start in starts}
        buffer.windows = {start: (alias.resource, alias.gpu_address) for start, alias in aliases.items()}
        buffers.append(buffer)
    yield tuple(buffers)
    device.synchronize()
    assert len(heaps) == 2


@pytest.mark.parametrize("dtype", list(dict.fromkeys(dtypes.all)))
def test_large_scatter_gather_replay(sparse_allocations: tuple[DX12Buffer, ...], dtype: DType) -> None:
    """Scatter/gather divergent 64-bit indices across windows and changed JIT buffers/views."""
    positions = Tensor([position // dtype.itemsize for position in POSITIONS], device="DX12", dtype=dtypes.int64).realize()

    def scatter(output: UOp, indices: UOp, values: UOp) -> UOp:
        """Write separate positions, retaining the full logical output parameter span."""
        lane = UOp.range(len(POSITIONS), 0)
        return output.index(indices.index(lane).load()).store(values.index(lane)).end(lane).sink(arg=KernelInfo())

    def gather(output: UOp, source: UOp, indices: UOp) -> UOp:
        """Read the same positions in one shader invocation with a large source binding."""
        lane = UOp.range(len(POSITIONS), 0)
        return output.index(lane).store(source.index(indices.index(lane).load())).end(lane).sink(arg=KernelInfo())

    @TinyJit
    def run(storage: Tensor, values: Tensor) -> Tensor:
        """Keep scatter/gather dependencies and buffer ownership in tinygrad's graph."""
        written = storage.custom_kernel(positions, values, fxn=scatter)[0]
        return Tensor.empty(len(POSITIONS), device="DX12", dtype=dtype).custom_kernel(written, positions, fxn=gather)[0].realize()

    pending = []
    for step, offset in enumerate((0, 1, 3, 15, 0, 1, 15, 3, 0, 1, 3, 15)):
        allocation = sparse_allocations[step % 2]
        view = allocation.dev.allocator._offset(allocation, SPAN - 128, offset)
        storage = Tensor(UOp.from_buffer(Buffer("DX12", view.size // dtype.itemsize, dtype, opaque=view)))
        values = [(step + i) % (2 if dtype is dtypes.bool else 8) for i in range(len(POSITIONS))]
        pending.append((run(storage, Tensor(values, device="DX12", dtype=dtype)).clone().realize(), values))
    for result, expected in pending:
        assert result.tolist() == expected


@pytest.mark.parametrize("crossing", [False, True])
def test_large_native_matrix(sparse_allocations: tuple[DX12Buffer, ...], crossing: bool) -> None:
    """Execute a queried LinAlg tile at or across 96 GiB in full 128 GiB bindings."""
    from tinygrad_dx12.codegen.linalg import _support
    from tinygrad_dx12.renderer.linalg import linalg_source

    left, output = sparse_allocations
    device, renderer = left.dev, left.dev.renderer
    info = _support(device, renderer, dtypes.float16, dtypes.float16, dtypes.float32, "Wave", (16, 16, 16))
    if not info or not info["flags"] & 1 or (16, 16, 16) not in info.get("shapes", ()):
        pytest.skip("requires the queried FP16 wave matrix operation")
    source = linalg_source(dtypes.float16, dtypes.float16, 1 << 31, 16, 32, "Wave", renderer.capabilities["wave_min"])
    position = (96 << 30) - (512 if crossing else 0)
    source = source.replace("void main(uint3 gid : SV_GroupID) {", f"void main(uint3 gid : SV_GroupID) {{ gid.y+={position // 1024}u;")
    program = device.runtime(TinyELF(device.compiler.compile_cached(source), "large_matrix", renderer.target, ()))
    right = device.buffer(1024)
    device.allocator._copyin(right, memoryview(struct.pack("<512e", *([1] * 512))))
    device.allocator._copyin(device.allocator._offset(left, 1024, position), memoryview(struct.pack("<512e", *([1] * 512))))
    arguments = [device.allocator._offset(buf, buf.size - position % 1024, position % 1024) for buf in (output, left)]
    program(*arguments, right, local_size=(renderer.capabilities["wave_min"], 1, 1), wait=True)
    data = bytearray(1024)
    device.allocator._copyout(memoryview(data), device.allocator._offset(output, 1024, position))
    assert struct.unpack("<256f", data) == (32.0,) * 256


def test_large_physical_allocation() -> None:
    """Allocate one physical buffer above 4 GiB and read/write its far end through a shader."""
    device = Device["DX12"]
    storage = Tensor(UOp.from_buffer(Buffer("DX12", (4 << 30) + 65536, dtypes.uint8))).realize()
    positions = Tensor([0, (4 << 30) - 1, (4 << 30) + 1], device="DX12", dtype=dtypes.int64)
    values = Tensor([11, 22, 33], device="DX12", dtype=dtypes.uint8)

    def scatter(output: UOp, indices: UOp, data: UOp) -> UOp:
        """Write three independently addressed locations in the single heap allocation."""
        lane = UOp.range(3, 0)
        return output.index(indices.index(lane).load()).store(data.index(lane)).end(lane).sink(arg=KernelInfo())

    storage = storage.custom_kernel(positions, values, fxn=scatter)[0].realize()
    assert storage[positions].tolist() == [11, 22, 33]
    assert storage.uop.buffer.nbytes > 4 << 30
    device.synchronize()


def test_large_copy_and_root_views(sparse_allocations: tuple[DX12Buffer, ...]) -> None:
    """Copies and ordinary small shader parameters preserve high offsets and boundary crossings."""
    allocation = sparse_allocations[0]
    for step, position in enumerate(POSITIONS):
        values = [step * 16 + i for i in range(16)]
        view = allocation.dev.allocator._offset(allocation, 64, position)
        payload = struct.pack("<16i", *values)
        allocation.dev.allocator._copyin(view, memoryview(payload))
        tensor = Tensor(UOp.from_buffer(Buffer("DX12", 16, dtypes.int32, opaque=view)))
        assert (tensor + 1).tolist() == [value + 1 for value in values]
        copied = bytearray(64)
        allocation.dev.allocator._copyout(memoryview(copied), view)
        assert copied == payload


def test_large_declared_span_with_short_replays() -> None:
    """Large parameter metadata permits short actual spans selected by changing scalar arguments."""
    from dataclasses import replace
    from tinygrad.codegen import to_program

    device = Device["DX12"]
    length = UOp.variable("large_length", 1, 1 << 35, dtype=dtypes.int64, param=True)
    length = length.replace(arg=replace(length.arg, slot=2))
    output, source = (UOp.placeholder((int(length.vmax),), dtypes.float32, slot) for slot in range(2))
    lane = UOp.range(length, 0, AxisType.LOOP)
    sink = output.index(lane).store(source.index(lane) + 1).end(lane).sink(arg=KernelInfo())
    program = to_program(sink, device.renderer)
    runtime = device.runtime(TinyELF(program.src[-1].arg, program.arg.name, device.renderer.target, ()))
    assert runtime.layout.table and any(binding.views > 1 for binding in runtime.layout.buffers)
    for size in (17, 33, 65, 17, 129):
        out, value = device.buffer(size * 4), device.buffer(size * 4)
        device.allocator._copyin(value, memoryview(struct.pack(f"<{size}f", *range(size))))
        variables = {"large_length": size}
        groups, local = program.arg.launch_dims(variables)
        runtime(out, value, global_size=groups, local_size=local, vals=program.arg.vals(variables), wait=True)
        copied = bytearray(size * 4)
        device.allocator._copyout(memoryview(copied), out)
        assert struct.unpack(f"<{size}f", copied) == tuple(i + 1 for i in range(size))


@pytest.mark.parametrize("staging", [False, True])
def test_large_cross_adapter_copy(sparse_allocations: tuple[DX12Buffer, ...], monkeypatch: pytest.MonkeyPatch, staging: bool) -> None:
    """Both shared and host-staged transfers preserve large source and destination offsets."""
    allocation = sparse_allocations[0]
    if allocation.dev.count() < 2:
        pytest.skip("requires two DX12 adapters")
    monkeypatch.setenv("DX12_FORCE_STAGING", "1" if staging else "0")
    other = Device["DX12:1"]
    remote = other.buffer(64)
    view = allocation.dev.allocator._offset(allocation, 64, 96 << 30)
    payload, copied = bytes(range(64)), bytearray(64)
    allocation.dev.allocator._copyin(view, memoryview(payload))
    other.allocator._transfer(remote, view, 64, allocation.dev, other)
    other.allocator._copyout(memoryview(copied), remote)
    assert copied == payload
    other.allocator._copyin(remote, memoryview(payload[::-1]))
    allocation.dev.allocator._transfer(view, remote, 64, other, allocation.dev)
    allocation.dev.allocator._copyout(memoryview(copied), view)
    assert copied == payload[::-1]
