"""Direct3D 12 compute device, byte storage, ordered submission, and dispatch."""

import ctypes as C
import decimal, os, struct, threading, time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from tinygrad.device import Allocator, BufferSpec, BufferStorage, Compiled, Program, TinyELF, ProfileDeviceEvent
from tinygrad.helpers import PROFILE, ProfileRangeEvent, Target, round_up, unwrap
from tinygrad_dx12.runtime.autogen import d3d12 as d
from tinygrad_dx12.runtime.autogen.win32 import GUID
from tinygrad_dx12.runtime.support.dx12 import Com, NativeRuntime, check
from tinygrad_dx12.runtime.support.compiler_dx12 import unpack_shader
from tinygrad_dx12.renderer.cstyle import DX12Renderer, WINDOW, parse_source, specialize_source

P, U, Q, H = C.c_void_p, C.c_uint32, C.c_uint64, C.c_int32


def create(obj: Com, slot: int, iid: GUID, argtypes: tuple, *args: object) -> Com:
    """Invoke an HRESULT creation method, retaining its returned COM reference."""
    result = P()
    check(obj.call(slot, H, (*argtypes, P, P), *args, C.byref(iid), C.byref(result)), f"COM create slot {slot}")
    return Com(result)


@dataclass(eq=False)
class DX12Buffer:
    """A byte view retaining its allocation; GPU addresses exclude the view offset."""

    resource: Com
    gpu_address: int
    size: int
    dev: DX12Device
    offset: int = 0
    _owner: DX12Buffer | None = None
    mapped: memoryview | None = None
    uses: dict[DX12Device, int] = field(default_factory=dict)
    heap: Com | None = None
    windows: dict[int, tuple[Com, int]] | None = None
    initial_state: int = 0

    @property
    def owner(self) -> DX12Buffer:
        """Return the base allocation without introducing a self-reference cycle."""
        return self if self._owner is None else self._owner

    @property
    def binding_address(self) -> int:
        """Return the full GPU virtual address; shader windows are described by the program."""
        if self.owner.heap is None and self.owner.windows is None:
            return self.gpu_address + self.offset
        _, address, offset = self.binding()
        return address + offset

    def binding(self, offset: int | None = None) -> tuple[Com, int, int]:
        """Resolve a shader address through a cached placed resource over the same allocation.

        The heap is allocated once. Windows alias its bytes without staging and
        keep shader and copy resource-relative addressing within a window,
        including view alignment. Existing global compute barriers also order
        accesses through overlapping buffer resources.
        """
        owner, offset = self.owner, self.offset if offset is None else offset
        if owner.heap is None and owner.windows is None:
            return self.resource, self.gpu_address, offset
        start = offset & ~65535
        with self.dev.lock:
            if owner.windows is None:
                owner.windows = {}
            if start not in owner.windows:
                if owner.heap is None:
                    return self.resource, self.gpu_address, offset
                desc = d.D3D12_RESOURCE_DESC()
                owner.resource.call(10, P, (P,), C.byref(desc))
                desc.Width = min(desc.Width - start, 2 * WINDOW + 65536)
                resource = create(
                    self.dev.native, 29, d.IID_ID3D12Resource, (P, Q, P, U, P), owner.heap.ptr, start, C.byref(desc), owner.initial_state, None
                )
                owner.windows[start] = resource, resource.call(11, Q, ())
            resource, address = owner.windows[start]
        return resource, address, offset - start


class DX12Command:
    """One allocator/list pair, reset only after its last fence completes."""

    def __init__(self, dev: DX12Device, pooled: bool) -> None:
        """Create an open compute command list with Enhanced Barriers support."""
        self.dev, self.pooled, self.fence, self.busy = dev, pooled, 0, True
        self.key: tuple | None = None
        self.owners: tuple = ()
        self.timing: DX12Timing | None = None
        self.allocator = create(dev.native, 9, d.IID_ID3D12CommandAllocator, (U,), 2)
        self.list = create(dev.native, 12, d.IID_ID3D12GraphicsCommandList7, (U, U, P, P), 0, 2, self.allocator.ptr, None)

    def reset(self) -> None:
        """Open a completed list for recording; active command storage cannot be reset."""
        if self.fence > self.dev.completed():
            raise RuntimeError("cannot reset an active DX12 command list")
        self.key, self.owners, self.timing = None, (), None
        check(self.allocator.call(8, H, ()), "Reset command allocator")
        check(self.list.call(10, H, (P, P), self.allocator.ptr, None), "Reset command list")
        self.busy = True

    def close(self) -> None:
        """Finish recording, reporting native validation errors."""
        check(self.list.call(9, H, ()), "Close command list")

    def copy(self, dest: DX12Buffer, src: DX12Buffer, size: int) -> None:
        """Record copies through the same resource windows used for shader addressing.

        Keep resource-relative offsets and sizes within a window while retaining
        one contiguous allocation. Callers retain the buffers through completion.
        """
        for start in range(0, size, WINDOW):
            dst, _, dst_offset = dest.binding(dest.offset + start)
            source, _, src_offset = src.binding(src.offset + start)
            self.list.call(15, None, (P, Q, P, Q, Q), dst.ptr, dst_offset, source.ptr, src_offset, min(WINDOW, size - start))


class DX12Timing:
    """Slot-local GPU timestamps resolved to persistently mapped readback storage."""

    def __init__(self, dev: DX12Device, command: DX12Command, count: int = 2) -> None:
        """Allocate query storage and record the first timestamp."""
        self.dev, self.count = dev, count
        desc = d.D3D12_QUERY_HEAP_DESC(1, count, 0)
        self.heap = create(dev.native, 39, d.IID_ID3D12QueryHeap, (P,), C.byref(desc))
        self.readback = dev.buffer(8 * count, 3)
        self.mark(command, 0)

    def mark(self, command: DX12Command, index: int) -> None:
        """Record a timestamp at an explicit query index."""
        command.list.call(53, None, (P, U, U), self.heap.ptr, 2, index)

    def finish(self, command: DX12Command) -> None:
        """Record the end timestamp and resolve every query into readback storage."""
        self.mark(command, self.count - 1)
        command.list.call(54, None, (P, U, U, U, P, Q), self.heap.ptr, 2, 0, self.count, self.readback.resource.ptr, 0)

    def timestamps(self) -> tuple[int, ...]:
        """Read raw GPU ticks after the caller has waited for submission completion."""
        return struct.unpack_from(f"<{self.count}Q", unwrap(self.readback.mapped))

    def seconds(self) -> float:
        """Return elapsed GPU seconds after submission completion."""
        ticks = self.timestamps()
        return (ticks[-1] - ticks[0]) / self.dev.frequency


class DX12Device(Compiled):
    """Independent adapter, compute queue, fence, compiler target, and submission lock."""

    def __init__(self, device: str) -> None:
        """Initialize the selected DXGI adapter; fail on missing required DX12 features."""
        self.lock = threading.RLock()
        self.loader = NativeRuntime.get()
        ordinal = int(device.split(":")[1]) if ":" in device else 0
        self.native, self.adapter, self.caps = self.loader.create_device(ordinal, debug=os.getenv("DX12_DEBUG", "0") == "1")
        self.queue = create(self.native, 8, d.IID_ID3D12CommandQueue, (P,), C.byref(d.D3D12_COMMAND_QUEUE_DESC(2, 0, 0, 0)))
        self.fence = self.new_fence()
        self.fence_value = 0
        frequency = Q()
        check(self.queue.call(16, H, (P,), C.byref(frequency)), "GetTimestampFrequency")
        self.frequency = frequency.value
        self.commands: list[DX12Command] = []
        self.pending_commands: list[DX12Command] = []
        self.inflight: list[tuple[int, tuple, Callable[[], None] | None]] = []
        self.transfers: list[DX12Transfer] = []
        self.transfer_support: dict[DX12Device, bool] = {}
        self.transfer_diagnostics: dict[DX12Device, str] = {}
        self.failure: BaseException | None = None
        self._renderer = DX12Renderer(Target(device="DX12", arch=self.caps.arch))
        from tinygrad_dx12.runtime.graph.dx12 import DX12Graph

        super().__init__(device, DX12Allocator(self), [DX12Renderer], None, DX12Graph, arch=self.caps.arch)
        Compiled.profile_events.append(ProfileDeviceEvent(device))

    @property
    def renderer(self) -> DX12Renderer:
        """Return this adapter's renderer with capability-specific cache identity."""
        return self._renderer

    def count(self) -> int:
        """Return the process-stable DXGI hardware adapter count."""
        return len(self.loader.adapters)

    def runtime(self, obj: TinyELF) -> DX12Program:
        """Create a pipeline on this concrete device, retaining the tinygrad program contract."""
        return DX12Program(self, obj)

    def new_fence(self, flags: int = 0) -> Com:
        """Create an independent fence, optionally shareable across adapters."""
        return create(self.native, 36, d.IID_ID3D12Fence, (Q, U), 0, flags)

    def completed(self) -> int:
        """Read the completion point, recognizing device removal and worker failures."""
        if self.failure is not None:
            raise RuntimeError("DX12 submission/transfer failed") from self.failure
        value = self.fence.call(8, Q, ())
        if value == (1 << 64) - 1:
            self.failure = RuntimeError("DX12 device removed")
            check(self.native.call(37, H, ()), "GetDeviceRemovedReason")
            raise self.failure
        return value

    def wait(self, value: int, timeout: int | None = None) -> None:
        """Wait for one fence point in milliseconds without holding the submission lock."""
        with self.lock:
            if self.pending_commands and value >= self.pending_commands[0].fence:
                self._flush()
        if self.completed() >= value:
            return
        kernel = self.loader.kernel32
        event = kernel.CreateEventW(None, False, False, None)
        if not event:
            raise C.WinError()
        try:
            check(self.fence.call(9, H, (Q, P), value, event), "SetEventOnCompletion")
            deadline = time.monotonic() + (timeout or self.wait_timeout_ms) / 1000
            while self.completed() < value:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"{self.device} fence {value} timed out")
                status = kernel.WaitForSingleObject(event, min(100, max(1, int(remaining * 1000))))
                if status == 0xFFFFFFFF:
                    raise C.WinError()
        finally:
            kernel.CloseHandle(event)

    def synchronize(self, timeout: int | None = None) -> None:
        """Wait for this device's submitted work and its explicit transfer obligations."""
        with self.lock:
            value, transfers = self.fence_value, tuple(self.transfers)
        self.wait(value, timeout)
        for transfer in transfers:
            transfer.wait(timeout)
        with self.lock:
            self.retire()

    def retire(self) -> int:
        """Drop completed references under the device lock and return their completion snapshot."""
        done = self.completed()
        ready = [callback for f, _, callback in self.inflight if f <= done and callback is not None]
        self.inflight = [(f, owners, callback) for f, owners, callback in self.inflight if f > done]
        for callback in ready:
            callback()
        self.transfers = [t for t in self.transfers if not t.complete()]
        return done

    def command(self, pool: bool = True) -> DX12Command:
        """Reserve a free command pair, creating another when existing pairs are active."""
        with self.lock:
            if pool:
                self._flush()
            done = self.retire()
            if pool:
                for command in self.commands:
                    if not command.busy and command.fence <= done:
                        command.reset()
                        return command
            command = DX12Command(self, pool)
            if pool:
                self.commands.append(command)
            return command

    def _flush(self) -> None:
        """Submit pending compute lists under the device lock and signal their shared fence.

        Every eager list begins with a compute barrier. Copies, graphs, explicit
        waits and synchronization flush pending work before crossing that boundary.
        Owners are retained from enqueue through completion, including on failure.
        """
        if not self.pending_commands:
            return
        if self.failure is not None:
            raise RuntimeError("DX12 submission/transfer failed") from self.failure
        lists = (P * len(self.pending_commands))(*(command.list.ptr.value for command in self.pending_commands))
        try:
            self.queue.call(10, None, (U, P), len(self.pending_commands), lists)
            check(self.queue.call(14, H, (P, Q), self.fence.ptr, self.fence_value), "Signal queue fence")
        except BaseException as exc:
            self.failure = exc
            raise
        self.pending_commands.clear()

    def submit(self, command: DX12Command, owners: tuple = (), callback: Callable[[], None] | None = None, batch: bool = False) -> int:
        """Enqueue a closed list with fence-owned resources, batching only ordinary eager compute.

        Timed calls, copies and graphs submit immediately. Untimed eager compute
        shares a submission of at most sixteen lists; all lists in a batch retain
        one completion point, which wait/synchronize submit before waiting on it.
        """
        with self.lock:
            if self.failure is not None:
                raise RuntimeError("DX12 submission/transfer failed") from self.failure
            if not batch:
                self._flush()
            if not self.pending_commands:
                self.fence_value += 1
            command.fence, command.busy = self.fence_value, False
            for owner in owners:
                if isinstance(owner, DX12Buffer):
                    owner.owner.uses[self] = self.fence_value
            self.inflight.append((self.fence_value, (command, *owners), callback))
            self.pending_commands.append(command)
            if not batch or len(self.pending_commands) >= 16:
                self._flush()
            return self.fence_value

    def buffer(self, size: int, heap: int = 1) -> DX12Buffer:
        """Allocate padded raw bytes in DEFAULT, UPLOAD, or READBACK memory."""
        physical = round_up(max(size, 4) + 4, 256)
        props = d.D3D12_HEAP_PROPERTIES(heap, 0, 0, 1, 1)
        desc = d.D3D12_RESOURCE_DESC(1, 0, physical, 1, 1, 1, 0, d.DXGI_SAMPLE_DESC(1, 0), 1, 4 if heap == 1 else 0)
        state = 0 if heap == 1 else 0xAC3 if heap == 2 else 0x400
        placement = None
        if size > 2 * WINDOW - 4:
            heap_desc = d.D3D12_HEAP_DESC(round_up(physical, 65536), props, 0, 0xC0)
            placement = create(self.native, 28, d.IID_ID3D12Heap, (P,), C.byref(heap_desc))
            resource = create(self.native, 29, d.IID_ID3D12Resource, (P, Q, P, U, P), placement.ptr, 0, C.byref(desc), state, None)
        else:
            resource = create(self.native, 27, d.IID_ID3D12Resource, (P, U, P, U, P), C.byref(props), 0, C.byref(desc), state, None)
        address = resource.call(11, Q, ())
        mapped = None
        if heap != 1:
            pointer = P()
            read_range = d.D3D12_RANGE(0, 0 if heap == 2 else physical)
            check(resource.call(8, H, (U, P, P), 0, C.byref(read_range), C.byref(pointer)), "Map staging buffer")
            mapped = memoryview((C.c_ubyte * physical).from_address(unwrap(pointer.value))).cast("B")
        return DX12Buffer(resource, address, size, self, mapped=mapped, heap=placement, initial_state=state)

    def upload(self, size: int) -> DX12Buffer:
        """Allocate persistently mapped CPU-write staging memory."""
        return self.buffer(size, 2)

    def timestamp(self, command: DX12Command, count: int = 2) -> DX12Timing:
        """Start a slot-local timestamp sequence."""
        return DX12Timing(self, command, count)


class DX12Transfer:
    """An ordered cross-adapter staging copy retaining both completion points."""

    def __init__(self, src: DX12Buffer, dest: DX12Buffer, size: int) -> None:
        """Enqueue both queues before returning and start an independent readiness worker."""
        self.src, self.dest = src, dest
        self.readback, self.upload = src.dev.buffer(size, 3), dest.dev.upload(size)
        self.ready = dest.dev.new_fence()
        self.error: BaseException | None = None
        first, second = sorted((src.dev, dest.dev), key=id)
        with first.lock, second.lock:
            source = src.dev.command()
            source.copy(self.readback, src, size)
            source.close()
            self.source_fence = src.dev.submit(source, (src, self.readback, self))
            destination = dest.dev.command()
            destination.copy(dest, self.upload, size)
            destination.close()
            try:
                check(dest.dev.queue.call(15, H, (P, Q), self.ready.ptr, 1), "Wait for transfer staging")
                self.dest_fence = dest.dev.submit(destination, (dest, self.upload, self))
            except BaseException as exc:
                self.error = self.src.dev.failure = self.dest.dev.failure = exc
                check(self.ready.call(10, H, (Q,), 1), "Release failed transfer submission")
                raise
            src.owner.uses[dest.dev] = self.dest_fence
            dest.owner.uses[src.dev] = self.source_fence
            src.dev.transfers.append(self)
            dest.dev.transfers.append(self)
        try:
            threading.Thread(target=self._run, name="DX12 transfer", daemon=True).start()
        except BaseException as exc:
            self.error = self.src.dev.failure = self.dest.dev.failure = exc
            check(self.ready.call(10, H, (Q,), 1), "Release failed transfer readiness")
            raise

    def _run(self) -> None:
        """Make this slot ready independently of other pending transfers."""
        try:
            self.src.dev.wait(self.source_fence)
            unwrap(self.upload.mapped)[: self.upload.size] = unwrap(self.readback.mapped)[: self.readback.size]
        except BaseException as exc:
            self.error = exc
            self.src.dev.failure = self.dest.dev.failure = exc
        finally:
            try:
                check(self.ready.call(10, H, (Q,), 1), "Signal transfer readiness")
            except BaseException as exc:
                self.error = self.src.dev.failure = self.dest.dev.failure = exc

    def complete(self) -> bool:
        """Report whether both queue uses have completed; propagate worker errors."""
        if self.error is not None:
            raise RuntimeError("DX12 transfer failed") from self.error
        return self.src.dev.completed() >= self.source_fence and self.dest.dev.completed() >= self.dest_fence

    def wait(self, timeout: int | None = None) -> None:
        """Wait only for this copy's source and destination fence points."""
        self.src.dev.wait(self.source_fence, timeout)
        self.dest.dev.wait(self.dest_fence, timeout)
        if self.error is not None:
            raise RuntimeError("DX12 transfer failed") from self.error


class DX12Allocator(Allocator[DX12Device]):
    """Raw DEFAULT storage with asynchronous owned upload and blocking readback."""

    def __init__(self, dev: DX12Device) -> None:
        """Enable backend transfers and defer allocation-cache reuse until completion."""
        super().__init__(dev, supports_copy_from_disk=False, supports_transfer=True)
        self.retired: list[tuple[BufferStorage, int, BufferSpec | None]] = []

    def alloc(self, size: int, options: BufferSpec | None = None) -> BufferStorage:
        """Return safe cached storage, collecting completed frees first."""
        with self.dev.lock:
            self._collect()
            return super().alloc(size, options)

    def _collect(self) -> None:
        """Move completed frees into tinygrad's cache while holding the device lock."""
        pending = []
        for storage, size, spec in self.retired:
            if all(dev.completed() >= fence for dev, fence in storage.buf.owner.uses.items()):
                super().free(storage, size, spec)
            else:
                pending.append((storage, size, spec))
        self.retired = pending

    def free_cache(self) -> None:
        """Release completed cached storage, retaining allocations still referenced by GPU work."""
        with self.dev.lock:
            self._collect()
            super().free_cache()

    def free(self, storage: BufferStorage, size: int, options: BufferSpec | None = None) -> None:
        """Keep pending allocations out of tinygrad's reusable cache."""
        with self.dev.lock:
            self.retired.append((storage, size, options))

    def _alloc(self, size: int, options: BufferSpec) -> BufferStorage:
        """Allocate tensor bytes; native external pointers are not a portable resource handle."""
        if options.external_ptr is not None:
            raise ValueError("DX12 external_ptr requires an ID3D12Resource import API")
        buffer = self.dev.buffer(size)
        if options.zero:
            self._copyin(buffer, memoryview(bytes(size)))
        return BufferStorage(buffer)

    def _offset(self, buf: DX12Buffer, size: int, offset: int) -> DX12Buffer:
        """Return a bounds-checked byte view retaining its original resource."""
        if offset < 0 or size < 0 or offset + size > buf.size:
            raise ValueError("DX12 view outside allocation")
        return DX12Buffer(buf.resource, buf.gpu_address, size, self.dev, buf.offset + offset, buf.owner)

    def _copyin(self, dest: DX12Buffer, src: memoryview) -> None:
        """Own caller data before returning; submit the upload on the ordered compute queue."""
        if src.nbytes > dest.size:
            raise ValueError("DX12 upload exceeds view size")
        upload = self.dev.upload(src.nbytes)
        unwrap(upload.mapped)[: src.nbytes] = src.cast("B")
        self._copy(dest, upload, src.nbytes)

    def _copyout(self, dest: memoryview, src: DX12Buffer) -> None:
        """Wait for readback, then copy the requested bytes into caller storage."""
        if dest.nbytes > src.size:
            raise ValueError("DX12 readback exceeds view size")
        readback = self.dev.buffer(dest.nbytes, 3)
        fence = self._copy(readback, src, dest.nbytes)
        self.dev.wait(fence)
        dest.cast("B")[:] = unwrap(readback.mapped)[: dest.nbytes]

    def _copy(self, dest: DX12Buffer, src: DX12Buffer, size: int) -> int:
        """Record one copy in its own ExecuteCommandLists execution boundary."""
        command = self.dev.command()
        command.copy(dest, src, size)
        command.close()
        return self.dev.submit(command, (dest, src))

    def _transfer(self, dest: DX12Buffer, src: DX12Buffer, sz: int, src_dev: DX12Device, dest_dev: DX12Device) -> None:
        """Order same-device or asynchronous cross-adapter copies, including byte views."""
        if sz > src.size or sz > dest.size or sz < 0:
            raise ValueError("DX12 transfer exceeds view size")
        if sz == 0:
            return
        if src_dev is dest_dev:
            if src.resource is dest.resource:
                if src.offset == dest.offset:
                    return
                if src.offset < dest.offset + sz and dest.offset < src.offset + sz:
                    temporary = self.dev.buffer(sz)
                    self._copy(temporary, src, sz)
                    self._copy(dest, temporary, sz)
                    return
            self._copy(dest, src, sz)
        else:
            from tinygrad_dx12.runtime.support.transfer import shared_transfer

            if shared_transfer(src, dest, sz) is None:
                DX12Transfer(src, dest, sz)


class DX12Descriptors:
    """Shader-visible descriptor storage private to one inactive execution slot."""

    def __init__(self, program: DX12Program, bufs: tuple[DX12Buffer, ...]) -> None:
        """Allocate one table; callers must wait before updating its descriptors."""
        self.program = program
        desc = d.D3D12_DESCRIPTOR_HEAP_DESC(0, sum(b.views for b in program.layout.buffers), 1, 0)
        self.heap = create(program.dev.native, 14, d.IID_ID3D12DescriptorHeap, (P,), C.byref(desc))
        self.cpu, self.gpu = d.D3D12_CPU_DESCRIPTOR_HANDLE(), d.D3D12_GPU_DESCRIPTOR_HANDLE()
        self.heap.call(9, P, (P,), C.byref(self.cpu))
        self.heap.call(10, P, (P,), C.byref(self.gpu))
        self.increment = program.dev.native.call(15, U, (U,), 0)
        self.update(bufs)

    def update(self, bufs: tuple[DX12Buffer, ...]) -> None:
        """Replace raw views in a completed slot's descriptor region."""
        windows = ((binding, buf, view * WINDOW) for binding, buf in zip(self.program.layout.buffers, bufs) for view in range(binding.views))
        for i, (binding, buf, start) in enumerate(windows):
            handle = d.D3D12_CPU_DESCRIPTOR_HANDLE(self.cpu.ptr + i * self.increment)
            size = max(0, min(2 * WINDOW - 4, buf.size + (buf.offset & 15) - start))
            native, _, offset = buf.binding((buf.offset & ~15) + start) if size else (buf.resource, 0, 0)
            length, resource = (round_up(size, 4), native.ptr) if size else (16, None)
            if binding.writable:
                uav = d.D3D12_UNORDERED_ACCESS_VIEW_DESC(Format=39, ViewDimension=1)
                uav.Buffer = d.D3D12_BUFFER_UAV(offset // 4, length // 4, 0, 0, 1)
                self.program.dev.native.call(19, None, (P, P, P, d.D3D12_CPU_DESCRIPTOR_HANDLE), resource, None, C.byref(uav), handle)
            else:
                srv = d.D3D12_SHADER_RESOURCE_VIEW_DESC(Format=39, ViewDimension=1, Shader4ComponentMapping=5768)
                srv.Buffer = d.D3D12_BUFFER_SRV(offset // 4, length // 4, 0, 1)
                self.program.dev.native.call(18, None, (P, P, d.D3D12_CPU_DESCRIPTOR_HANDLE), resource, C.byref(srv), handle)


class DX12Program(Program[DX12Device]):
    """Device-local root signature, PSO specializations, bindings, and eager dispatch."""

    def __init__(self, dev: DX12Device, obj: TinyELF, graph: bool = False) -> None:
        """Build root bindings; graphs use mutable CBVs and FP64 keeps root buffers.

        Both layouts bind the same shader registers, so graph replay reuses the
        compiled DXIL without a second shader or indirect root-argument commands.
        """
        self.dev, self.obj, self.name, self.profile_key = dev, obj, obj.name, obj.profile_key
        self.source, self.dxil = unpack_shader(obj.lib)
        self.layout = parse_source(self.source)
        if graph:
            table = bool(self.layout.buffers) and (self.layout.table or not self.layout.has_fp64)
            self.layout = replace(
                self.layout,
                buffers=tuple(replace(b, root_index=0 if table else i) for i, b in enumerate(self.layout.buffers)),
                table=table,
                scalar_root=1 if table else len(self.layout.buffers),
                scalar_cbv=bool(self.layout.scalar_words),
            )
        self.root_signature = self._root_signature()
        self.pipelines: dict[tuple[tuple[int, int, int], bool], Com] = {}
        self.commands: list[DX12Command] = []
        self._signature: Com | None = None
        self.specialize(self.layout.local_size)

    def _root_signature(self) -> Com:
        """Serialize root-signature 1.1 from the shader's single binding description."""
        layout = self.layout
        count = (1 if layout.table else len(layout.buffers)) + bool(layout.scalar_words)
        params = (d.D3D12_ROOT_PARAMETER1 * count)()
        if layout.table:
            ranges = (d.D3D12_DESCRIPTOR_RANGE1 * len(layout.buffers))(*[
                d.D3D12_DESCRIPTOR_RANGE1(int(b.writable), b.views, b.register, 0, 3, 0xFFFFFFFF) for b in layout.buffers
            ])
            params[0].ParameterType = 0
            params[0].DescriptorTable = d.D3D12_ROOT_DESCRIPTOR_TABLE1(len(ranges), ranges)
        else:
            for binding in layout.buffers:
                params[binding.root_index].ParameterType = 4 if binding.writable else 3
                params[binding.root_index].Descriptor = d.D3D12_ROOT_DESCRIPTOR1(binding.register, 0, 2)
        if layout.scalar_words:
            param = params[layout.scalar_root]
            param.ParameterType = 2 if layout.scalar_cbv else 1
            if layout.scalar_cbv:
                param.Descriptor = d.D3D12_ROOT_DESCRIPTOR1(0, 0, 2)
            else:
                param.Constants = d.D3D12_ROOT_CONSTANTS(0, 0, layout.scalar_words)
        desc = d.D3D12_VERSIONED_ROOT_SIGNATURE_DESC(Version=2)
        desc.Desc_1_1 = d.D3D12_ROOT_SIGNATURE_DESC1(count, params, 0, None, 0)
        data = self.dev.loader.serialize_root_signature(desc)
        return create(self.dev.native, 16, d.IID_ID3D12RootSignature, (U, P, C.c_size_t), 0, C.create_string_buffer(data), len(data))

    def specialize(self, local_size: tuple[int, int, int], aligned: bool = False) -> Com:
        """Cache fixed numthreads and known zero binding offsets without changing byte views.

        A 16-byte-aligned view has no residual offset after binding its GPU address.
        Specialize only that offset; DXC derives alignment for each actual access,
        including native halfword loads/stores and byte-packed operations.
        """
        local_size = (*local_size,)
        key = (local_size, aligned and self.layout.alignable)
        with self.dev.lock:
            if key not in self.pipelines:
                source = specialize_source(self.source, local_size)
                if key[1]:
                    for binding in self.layout.buffers:
                        source = source.replace(f"#define _offset_{binding.slot} _p{binding.offset_word}", f"#define _offset_{binding.slot} 0u")
                dxil = (
                    self.dxil if local_size == self.layout.local_size and not key[1] else unpack_shader(self.dev.compiler.compile_cached(source))[1]
                )
                data = C.create_string_buffer(dxil)
                desc = d.D3D12_COMPUTE_PIPELINE_STATE_DESC(
                    pRootSignature=self.root_signature.ptr, CS=d.D3D12_SHADER_BYTECODE(C.cast(data, P), len(dxil))
                )
                self.pipelines[key] = create(self.dev.native, 11, d.IID_ID3D12PipelineState, (P,), C.byref(desc))
            return self.pipelines[key]

    def descriptors(self, bufs: tuple[DX12Buffer, ...]) -> DX12Descriptors | None:
        """Allocate descriptors for graph bindings or root-signature overflow."""
        return DX12Descriptors(self, bufs) if self.layout.table else None

    def bind_buffers(self, command: DX12Command, bufs: tuple[DX12Buffer, ...], descriptors: DX12Descriptors | None) -> None:
        """Bind raw buffer addresses or their descriptor table on an open command list."""
        if descriptors is not None:
            heaps = (P * 1)(descriptors.heap.ptr.value)
            command.list.call(28, None, (U, P), 1, heaps)
            command.list.call(31, None, (U, d.D3D12_GPU_DESCRIPTOR_HANDLE), 0, descriptors.gpu)
        else:
            for binding, buf in zip(self.layout.buffers, bufs):
                command.list.call(41 if binding.writable else 39, None, (U, Q), binding.root_index, buf.binding_address & ~15)

    def indirect_signature(self) -> Com:
        """Return the layout's native indirect signature for explicit callers and diagnostics.

        Ordinary graph replay uses direct dispatch. This remains paired with
        BindingLayout.pack_record for callers that explicitly submit indirect work.
        """
        with self.dev.lock:
            if self._signature is None:
                arguments = [
                    d.D3D12_INDIRECT_ARGUMENT_DESC(
                        Type=8 if b.writable else 7,
                        **{"UnorderedAccessView" if b.writable else "ShaderResourceView": d.D3D12_INDIRECT_ROOT_DESCRIPTOR(b.root_index)},
                    )
                    for b in self.layout.buffers
                    if not self.layout.table
                ]
                if self.layout.scalar_words:
                    arguments.append(
                        d.D3D12_INDIRECT_ARGUMENT_DESC(Type=6, ConstantBufferView=d.D3D12_INDIRECT_ROOT_DESCRIPTOR(self.layout.scalar_root))
                        if self.layout.scalar_cbv
                        else d.D3D12_INDIRECT_ARGUMENT_DESC(
                            Type=5, Constant=d.D3D12_INDIRECT_CONSTANT(self.layout.scalar_root, 0, self.layout.scalar_words)
                        )
                    )
                arguments.append(d.D3D12_INDIRECT_ARGUMENT_DESC(Type=2))
                array = (d.D3D12_INDIRECT_ARGUMENT_DESC * len(arguments))(*arguments)
                desc = d.D3D12_COMMAND_SIGNATURE_DESC(self.layout.stride, len(arguments), array, 0)
                self._signature = create(
                    self.dev.native, 41, d.IID_ID3D12CommandSignature, (P, P), C.byref(desc), self.root_signature.ptr if len(arguments) > 1 else None
                )
            return self._signature

    def __call__(
        self,
        *bufs: DX12Buffer,
        global_size: tuple[int, int, int] = (1, 1, 1),
        local_size: tuple[int, int, int] = (1, 1, 1),
        vals: tuple[int, ...] = (),
        wait: bool = False,
        timeout: int | None = None,
    ) -> float | None:
        """Dispatch group counts, returning GPU seconds only when requested.

        Zero-work launches are skipped. Buffers must belong to this device and match
        the compiled signature. Local dimensions select a fixed-numthreads PSO.
        Cached recordings reuse identical bindings and launch sizes; changed
        arguments rebuild only completed recordings. Active evictions remain owned
        by the device's submission fence. Timing belongs to the recording key, so
        ordinary launches never inherit timestamp commands.
        """
        global_size, local_size = (*(global_size or (1, 1, 1)),), (*(local_size or (1, 1, 1)),)
        if len(global_size) != 3 or any(x < 0 or x > 65535 for x in global_size):
            raise ValueError("invalid DX12 dispatch dimensions")
        if len(bufs) != len(self.layout.buffers) or any(b.dev is not self.dev for b in bufs):
            raise ValueError("DX12 buffer device/signature mismatch")
        if 0 in global_size:
            return 0.0 if wait else None
        if len(vals) != len(self.layout.scalars):
            raise ValueError("DX12 scalar argument count mismatch")
        scalar_data = self.layout.scalar_data(tuple(b.offset & 15 for b in bufs), vals) if self.layout.scalars else None
        key = (local_size, tuple((b.resource, b.offset, b.size) for b in bufs), scalar_data, global_size, bool(wait or PROFILE))
        with self.dev.lock:
            done = self.dev.retire()
            command = next((c for c in self.commands if not c.busy and c.fence <= done and c.key == key), None)
            if command is None:
                pipeline = self.specialize(local_size, aligned=all(buf.offset % 16 == 0 for buf in bufs))
                if scalar_data is None:
                    scalar_data = self.layout.scalar_data(tuple(b.offset & 15 for b in bufs), vals)
                command = next((c for c in self.commands if not c.busy and c.fence <= done), None)
                if command is None:
                    command = self.dev.command(pool=False)
                    self.commands.append(command)
                else:
                    command.reset()
                descriptors = self.descriptors(bufs)
                scalar_buffer = self.dev.upload(self.layout.cbv_size) if self.layout.scalar_cbv else None
                if scalar_buffer is not None:
                    unwrap(scalar_buffer.mapped)[: len(scalar_data)] = scalar_data
                access = d.D3D12_BARRIER_ACCESS_UNORDERED_ACCESS | d.D3D12_BARRIER_ACCESS_SHADER_RESOURCE
                barrier = d.D3D12_GLOBAL_BARRIER(d.D3D12_BARRIER_SYNC_COMPUTE_SHADING, d.D3D12_BARRIER_SYNC_COMPUTE_SHADING, access, access)
                group = d.D3D12_BARRIER_GROUP(Type=d.D3D12_BARRIER_TYPE_GLOBAL, NumBarriers=1, pGlobalBarriers=C.pointer(barrier))
                command.list.call(80, None, (U, C.POINTER(d.D3D12_BARRIER_GROUP)), 1, C.byref(group))
                timing = self.dev.timestamp(command) if wait or PROFILE else None
                command.list.call(29, None, (P,), self.root_signature.ptr)
                command.list.call(25, None, (P,), pipeline.ptr)
                self.bind_buffers(command, bufs, descriptors)
                if scalar_buffer is not None:
                    command.list.call(37, None, (U, Q), self.layout.scalar_root, scalar_buffer.gpu_address)
                elif self.layout.scalar_words:
                    data = C.create_string_buffer(scalar_data)
                    command.list.call(35, None, (U, U, P, U), self.layout.scalar_root, self.layout.scalar_words, data, 0)
                command.list.call(14, None, (U, U, U), *global_size)
                if timing is not None:
                    timing.finish(command)
                command.close()
                command.key, command.owners, command.timing = key, (pipeline, descriptors, scalar_buffer, timing), timing
            timing = command.timing

            def profile() -> None:
                """Publish completed GPU timestamps through tinygrad's profiling interface."""
                if timing is not None:
                    ticks = timing.timestamps()
                    scale = decimal.Decimal(1_000_000) / self.dev.frequency
                    Compiled.profile_events.append(
                        ProfileRangeEvent(self.dev.device, self.name, ticks[0] * scale, ticks[-1] * scale, self.profile_key)
                    )

            fence = self.dev.submit(command, (*bufs, self, *command.owners), profile if PROFILE else None, batch=not (wait or PROFILE))
            # A completed GPU fence does not mean a concurrent timing caller has read its result.
            command.busy = wait
        if wait:
            try:
                self.dev.wait(fence, timeout)
                return unwrap(timing).seconds()
            finally:
                with self.dev.lock:
                    command.busy = False
        return None
