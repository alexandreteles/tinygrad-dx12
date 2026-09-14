"""Hardware contracts for storage, ordered transfers, and independent adapters."""

import ctypes as C
import gc, os, struct, weakref
from concurrent.futures import ThreadPoolExecutor
import pytest
from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer

pytestmark = pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="set DX12_TEST_GPU=1 to execute GPU contracts")


def test_byte_views_and_owned_upload() -> None:
    """Unaligned byte views retain neighboring bytes and copyin owns the caller's data."""
    base = Buffer("DX12", 513, dtypes.uint8).allocate()
    payload = bytearray((i * 19 + 7) % 256 for i in range(513))
    expected = bytes(payload)
    base.allocator._copyin(base._buf, memoryview(payload))
    payload[:] = bytes(len(payload))
    view = base.view(257, dtypes.uint8, 3).ensure_allocated()
    assert bytes(view.as_memoryview()) == expected[3:260]
    view.allocator._copyin(view._buf, memoryview(bytes([77]) * 257))
    assert bytes(base.as_memoryview()) == expected[:3] + bytes([77]) * 257 + expected[260:]


@pytest.mark.parametrize("source,destination", [("DX12", "DX12:1"), ("DX12:1", "DX12")])
@pytest.mark.parametrize("staging", [False, True])
def test_transfer_views_immediate_consumer(source: str, destination: str, staging: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both transfer paths order immediate consumers and preserve view offsets and ownership."""
    if Device["DX12"].count() < 2:
        pytest.skip("two adapters required")
    monkeypatch.setenv("DX12_FORCE_STAGING", str(int(staging)))
    outputs = []
    for iteration in range(12):
        src = Buffer(source, 513, dtypes.uint8).allocate()
        dest = Buffer(destination, 520, dtypes.uint8).allocate()
        data = bytes((i + iteration * 13) % 256 for i in range(513))
        src.allocator._copyin(src._buf, memoryview(data))
        target = dest.view(257, dtypes.uint8, 7).ensure_allocated()
        origin = src.view(257, dtypes.uint8, 3).ensure_allocated()
        target.copy_from(origin)
        outputs.append((target, data[3:260]))
        del src, origin, dest
    gc.collect()
    for output, expected in outputs:
        assert bytes(output.as_memoryview()) == expected


def test_independent_devices_concurrent_work() -> None:
    """Separate adapters execute concurrently without sharing programs or submission state."""
    if Device["DX12"].count() < 2:
        pytest.skip("two adapters required")
    assert Device["DX12"] is Device["DX12:0"]
    assert Device["DX12"].adapter.luid != Device["DX12:1"].adapter.luid

    from tinygrad.engine.realize import runtime_cache

    streams = []
    for ordinal in range(2):
        device = "DX12" if ordinal == 0 else "DX12:1"
        source = Tensor([ordinal + 2], device=device).realize()
        output = (source * 3).realize()
        program = next(p for p in reversed(list(runtime_cache.values())) if p.dev is Device[device])
        bufs = tuple((output if b.writable else source).uop.buffer._buf for b in program.layout.buffers)
        streams.append((program, bufs, output.uop.buffer))

    def execute(ordinal: int) -> bytes:
        """Submit precompiled work; tinygrad's compiler Context is intentionally process-global."""
        program, bufs, output = streams[ordinal]
        for _ in range(12):
            program(*bufs, local_size=program.layout.local_size)
        return bytes(output.as_memoryview())

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(execute, (0, 1)))
    assert results == [struct.pack("<i", (ordinal + 2) * 3) for ordinal in range(2)]


def test_subword_stores_preserve_adjacent_views() -> None:
    """GPU writes to odd byte offsets do not corrupt other elements in the shared DWORD."""
    tensor = Tensor(bytes(range(64)), device="DX12", dtype=dtypes.uint8).contiguous().realize()
    tensor[1:62].assign(Tensor.full((61,), 201, device="DX12", dtype=dtypes.uint8)).realize()
    assert tensor.tolist() == [0] + [201] * 61 + [62, 63]


def test_copy_roundtrip_float_bit_patterns() -> None:
    """Storage copies preserve NaN payloads, infinities, signed zeros, and integer bits."""
    bits = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345, 0xFFFFFFFF]
    data = struct.pack("<6I", *bits)
    buffer = Buffer("DX12", 6, dtypes.float32).allocate()
    buffer.allocator._copyin(buffer._buf, memoryview(data))
    assert bytes(buffer.as_memoryview()) == data


def test_overlapping_same_resource_copy() -> None:
    """Overlapping views copy through ordered temporary GPU storage."""
    base = Buffer("DX12", 64, dtypes.uint8).allocate()
    base.allocator._copyin(base._buf, memoryview(bytes(range(64))))
    base.view(37, dtypes.uint8, 3).ensure_allocated().copy_from(base.view(37, dtypes.uint8, 0).ensure_allocated())
    assert bytes(base.as_memoryview()) == bytes(range(3)) + bytes(range(37)) + bytes(range(40, 64))


def test_free_cache_retains_active_and_releases_completed_storage() -> None:
    """Cache eviction retains queued resources and releases them once their fence has completed."""
    from tinygrad_dx12.runtime.ops_dx12 import DX12Allocator, P, Q, H
    from tinygrad_dx12.runtime.support.dx12 import check

    dev = Device["DX12"]
    allocator = DX12Allocator(dev)
    storage = allocator.alloc(64)
    owner = weakref.ref(storage.buf)
    gate = dev.new_fence()
    check(dev.queue.call(15, H, (P, Q), gate.ptr, 1), "test cache readiness wait")
    try:
        allocator._copyin(storage.buf, memoryview(bytes(64)))
        allocator.free(storage, 64)
        del storage
        allocator.free_cache()
        assert owner() is not None
    finally:
        check(gate.call(10, H, (Q,), 1), "release test cache readiness")
    dev.synchronize()
    allocator.free_cache()
    gc.collect()
    assert owner() is None


def test_failed_staging_submission_releases_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure after destination Wait must release that wait and report failure on both devices."""
    from tinygrad_dx12.runtime.ops_dx12 import DX12Device, DX12Transfer, P, Q, H
    from tinygrad_dx12.runtime.support.dx12 import check

    if Device["DX12"].count() < 2:
        pytest.skip("two adapters required")
    source, destination = DX12Device("DX12"), DX12Device("DX12:1")
    probe = source.command()
    probe.close()
    ready = destination.new_fence()
    monkeypatch.setattr(destination, "new_fence", lambda: ready)

    def fail_submit(*args: object, **kwargs: object) -> int:
        """Inject failure after native queue Wait, before the readiness worker can start."""
        raise RuntimeError("injected destination submission failure")

    monkeypatch.setattr(destination, "submit", fail_submit)
    with pytest.raises(RuntimeError, match="injected"):
        DX12Transfer(source.buffer(16), destination.buffer(16), 16)
    assert source.failure is destination.failure
    with pytest.raises(RuntimeError, match="submission/transfer failed"):
        source.submit(probe)
    with pytest.raises(RuntimeError, match="submission/transfer failed"):
        source.synchronize()
    with pytest.raises(RuntimeError, match="submission/transfer failed"):
        destination.synchronize()
    # This signal follows the blocked native wait. Its event proves that the error path released the queue.
    marker = DX12Device.new_fence(destination)
    check(destination.queue.call(14, H, (P, Q), marker.ptr, 1), "Signal post-failure marker")
    event = destination.loader.kernel32.CreateEventW(None, False, False, None)
    assert event
    try:
        check(marker.call(9, H, (Q, P), 1, event), "Set post-failure event")
        assert destination.loader.kernel32.WaitForSingleObject(event, 5000) == 0
        assert ready.call(8, C.c_uint64, ()) == 1
    finally:
        destination.loader.kernel32.CloseHandle(event)
