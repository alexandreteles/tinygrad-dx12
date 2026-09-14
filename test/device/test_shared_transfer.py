"""Cross-adapter byte-copy validation independent of tensor arithmetic."""

import os
import pytest
from tinygrad_dx12.runtime.support.dx12 import NativeRuntime
from tinygrad_dx12.runtime.ops_dx12 import DX12Device
from tinygrad_dx12.runtime.support.transfer import shared_transfer

pytestmark = pytest.mark.skipif(os.name != "nt" or not os.environ.get("TINYGRAD_DX12_CACHE"), reason="requires installed DX12 toolchain")


@pytest.fixture(scope="module")
def devices() -> tuple[DX12Device, DX12Device]:
    """Construct independent devices only when two hardware adapters are available."""
    if len(NativeRuntime.get().adapters) < 2:
        pytest.skip("requires two DXGI adapters")
    return DX12Device("DX12"), DX12Device("DX12:1")


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_transfer_views_and_inflight_ownership(devices: tuple[DX12Device, DX12Device], reverse: bool) -> None:
    """Copy changing, unaligned views without intermediate caller waits, then verify every destination."""
    source, destination = devices[::-1] if reverse else devices
    pending = []
    for index in range(8):
        payload = bytes((byte * 17 + index * 23) % 256 for byte in range(1021))
        source_base, dest_base = source.buffer(2048), destination.buffer(2048)
        source_view = source.allocator._offset(source_base, len(payload), 5)
        dest_view = destination.allocator._offset(dest_base, len(payload), 13)
        source.allocator._copyin(source_view, memoryview(payload))
        transfer = shared_transfer(source_view, dest_view, len(payload))
        if transfer is None:
            pytest.skip(f"sharing unavailable: {source.transfer_diagnostics[destination]}")
        pending.append((dest_view, payload, transfer))
    for dest_view, payload, transfer in pending:
        actual = bytearray(len(payload))
        destination.allocator._copyout(memoryview(actual), dest_view)
        assert actual == payload
        transfer.wait()
        assert transfer.complete()
