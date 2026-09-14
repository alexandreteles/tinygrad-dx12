"""Cross-adapter copies through shared DEFAULT storage and a shared GPU fence."""

import ctypes as C
import os
from tinygrad_dx12.runtime.autogen import d3d12 as d
from tinygrad_dx12.runtime.autogen.win32 import GUID
from tinygrad_dx12.runtime.support.dx12 import Com, check
from tinygrad_dx12.runtime.ops_dx12 import DX12Buffer, DX12Transfer, create
from tinygrad_dx12.renderer.cstyle import WINDOW

P, U, Q, H = C.c_void_p, C.c_uint32, C.c_uint64, C.c_int32


def _open_shared(source: DX12Buffer, destination: DX12Buffer, obj: Com, iid: GUID) -> Com:
    """Open one shared heap/fence on the destination and close the temporary NT handle."""
    handle = P()
    check(
        source.dev.native.call(31, H, (P, P, U, C.c_wchar_p, P), obj.ptr, None, 0x10000000, None, C.byref(handle)),
        "CreateSharedHandle(cross-adapter)",
    )
    try:
        return create(destination.dev.native, 32, iid, (P,), handle)
    finally:
        source.dev.loader.kernel32.CloseHandle(handle)


class SharedTransfer(DX12Transfer):
    """Retain both local views of shared storage until both ordered GPU copies complete."""

    def __init__(self, source: DX12Buffer, destination: DX12Buffer, size: int) -> None:
        """Create shared heap/fence resources without submitting commands, allowing safe fallback."""
        self.src, self.dest = source, destination
        self.source_fence = self.dest_fence = 0
        self.size = size
        allocation_size = (size + 65535) & ~65535
        heap_desc = d.D3D12_HEAP_DESC(allocation_size, d.D3D12_HEAP_PROPERTIES(1, 0, 0, 1, 1), 65536, 0x21)
        self.source_heap = create(source.dev.native, 28, d.IID_ID3D12Heap, (P,), C.byref(heap_desc))
        self.dest_heap = _open_shared(source, destination, self.source_heap, d.IID_ID3D12Heap)
        resource_desc = d.D3D12_RESOURCE_DESC(1, 65536, allocation_size, 1, 1, 1, 0, d.DXGI_SAMPLE_DESC(1, 0), 1, 0x10)
        source_resource = create(
            source.dev.native, 29, d.IID_ID3D12Resource, (P, Q, P, U, P), self.source_heap.ptr, 0, C.byref(resource_desc), 0, None
        )
        dest_resource = create(
            destination.dev.native, 29, d.IID_ID3D12Resource, (P, Q, P, U, P), self.dest_heap.ptr, 0, C.byref(resource_desc), 0, None
        )
        self.source_buffer = DX12Buffer(
            source_resource,
            source_resource.call(11, Q, ()),
            allocation_size,
            source.dev,
            heap=self.source_heap if size > 2 * WINDOW - 4 else None,
        )
        self.dest_buffer = DX12Buffer(
            dest_resource,
            dest_resource.call(11, Q, ()),
            allocation_size,
            destination.dev,
            heap=self.dest_heap if size > 2 * WINDOW - 4 else None,
        )
        self.signal = source.dev.new_fence(3)
        self.ready = _open_shared(source, destination, self.signal, d.IID_ID3D12Fence)

    def enqueue(self) -> None:
        """Enqueue both sides with a GPU fence dependency while holding device locks in a fixed order."""
        source, destination = self.src.dev, self.dest.dev
        first, second = sorted((source, destination), key=id)
        with first.lock, second.lock:
            source_command = source.command()
            source_command.copy(self.source_buffer, self.src, self.size)
            source_command.close()
            dest_command = destination.command()
            dest_command.copy(self.dest, self.dest_buffer, self.size)
            dest_command.close()
            try:
                self.source_fence = source.submit(source_command, (self.src, self))
                check(source.queue.call(14, H, (P, Q), self.signal.ptr, 1), "Signal cross-adapter copy")
                check(destination.queue.call(15, H, (P, Q), self.ready.ptr, 1), "Wait cross-adapter copy")
                self.dest_fence = destination.submit(dest_command, (self.dest, self))
            except BaseException as exc:
                source.failure = destination.failure = exc
                raise
            self.src.owner.uses[destination] = self.dest_fence
            self.dest.owner.uses[source] = self.source_fence
            source.transfers.append(self)
            destination.transfers.append(self)

    def complete(self) -> bool:
        """Report completion of both queue uses, propagating device loss or transfer failure."""
        return self.src.dev.completed() >= self.source_fence and self.dest.dev.completed() >= self.dest_fence

    def wait(self, timeout: int | None = None) -> None:
        """Wait only for this transfer's two completion points."""
        self.src.dev.wait(self.source_fence, timeout)
        self.dest.dev.wait(self.dest_fence, timeout)


def shared_transfer(source: DX12Buffer, destination: DX12Buffer, size: int) -> SharedTransfer | None:
    """Try an ordered-pair shared copy, returning None before submission when sharing is unavailable.

    ``DX12_FORCE_STAGING=1`` bypasses sharing for transfer validation. Capability results
    remain device-local; one rejected adapter pair does not affect another pair.
    """
    if os.environ.get("DX12_FORCE_STAGING", "0") == "1":
        return None
    first, second = sorted((source.dev, destination.dev), key=id)
    with first.lock, second.lock:
        support = source.dev.transfer_support
        if support.get(destination.dev) is False:
            return None
        try:
            transfer = SharedTransfer(source, destination, size)
        except (RuntimeError, OSError) as exc:
            support[destination.dev] = False
            source.dev.transfer_diagnostics[destination.dev] = str(exc)
            return None
        support[destination.dev] = True
        transfer.enqueue()
        return transfer
