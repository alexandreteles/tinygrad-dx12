"""Direct-dispatch graphs with mutable descriptors and fence-owned replay slots."""

import ctypes as C
import decimal, functools, operator, struct, threading, weakref
from dataclasses import dataclass, field
from tinygrad.device import Buffer, ProfileGraphEntry, ProfileGraphEvent
from tinygrad.engine.jit import GraphRunner, GraphException
from tinygrad.engine.realize import get_call_arg_uops
from tinygrad.helpers import PROFILE
from tinygrad.uop.ops import Ops, UOp
from tinygrad_dx12.compat import resolve_graph_input
from tinygrad_dx12.runtime.autogen import d3d12 as d3d
from tinygrad_dx12.runtime.ops_dx12 import DX12Buffer, DX12Command, DX12Timing, DX12Descriptors, DX12Device, DX12Program


def barrier_access(dependencies: list[int], accesses: list[int], covered: dict[int, int], after: int) -> tuple[int, int]:
    """Return the merged compute barrier masks for unresolved producer accesses.

    A global barrier covers earlier accesses with the same source masks. Track
    its destination masks separately: publishing a UAV write for an SRV read
    does not satisfy a subsequent UAV consumer.
    """
    pending = [p for p in dependencies if after & ~covered.get(p, 0)]
    before = functools.reduce(operator.or_, (accesses[p] for p in pending), 0)
    if before:
        for p, access in enumerate(accesses):
            if not access & ~before:
                covered[p] = covered.get(p, 0) | after
    return before, after if before else 0


@dataclass
class GraphSlot:
    """Resources that may only be patched or reset after their fence completes."""

    command: DX12Command
    arguments: DX12Buffer
    descriptors: list[DX12Descriptors | None]
    timing: DX12Timing | None = None
    query_count: int = 0
    local_sizes: tuple[tuple[int, int, int], ...] = ()
    global_sizes: tuple[tuple[int, int, int], ...] = ()
    aligned: tuple[bool, ...] = ()
    fence: int = 0
    owners: tuple[DX12Buffer, ...] = ()
    buffers: list[list[DX12Buffer]] = field(default_factory=list)


class DX12Graph(GraphRunner):
    """Record a single-device kernel sequence with direct compute dispatches.

    Three independent lists and upload regions allow queued replays without
    modifying active arguments. Launch-size, alignment, or timing-mode changes
    rebuild an idle list; FP64 root-buffer changes also rebuild that idle list.
    Other buffer/scalar changes patch descriptors/data.
    Timestamp storage is allocated lazily for endpoints or per-kernel profiling.
    """

    def __init__(self, linear: UOp, input_uops: tuple[UOp, ...] = ()) -> None:
        """Resolve tinygrad programs, plan their barriers, and record replay slots."""
        super().__init__(linear, input_uops)
        self.programs = tuple(rt for rt in self.runtimes if isinstance(rt, DX12Program))
        if not self.programs or len(self.programs) != len(self.runtimes):
            raise GraphException("DX12 graphs require DX12 program calls")
        self.dev = self.programs[0].dev
        self.device = self.dev.device
        if any(rt.dev is not self.dev for rt in self.programs):
            raise GraphException("DX12 graphs require one device")
        self.programs = tuple(DX12Program(rt.dev, rt.obj, graph=True) for rt in self.programs)
        self.lock, self.next_slot = threading.RLock(), 0
        self.layouts = [rt.layout for rt in self.programs]
        self.cbv_offsets, size = [], 0
        for layout in self.layouts:
            self.cbv_offsets.append(((size + 255) & ~255) if layout.scalar_cbv else 0)
            if layout.scalar_cbv:
                size = self.cbv_offsets[-1] + layout.cbv_size
        self.argument_size = max(size, 4)
        defaults = {v.expr: int(v.vmin) for _, ast, _, dv in self.calls for v in ast.arg.vars if v.expr not in dv}
        self.launches = [ast.arg.launch_dims({**defaults, **dv}) for _, ast, _, dv in self.calls]
        self.values = [ast.arg.vals({**defaults, **dv}) for _, ast, _, dv in self.calls]
        # Resolve full view templates, including MSELECT(PARAM) after tuple-device splitting.
        self.buffer_patches = [
            (j, i, source, self.cbv_offsets[j] + 4 * binding.offset_word)
            for j, (call, layout) in enumerate(zip(self.linear.src, self.layouts))
            for i, binding in enumerate(layout.buffers)
            for source in (get_call_arg_uops(call)[binding.slot],)
            if any(u.op is Ops.PARAM for u in source.toposort())
        ]
        self.input_refs: tuple[weakref.ReferenceType[UOp], ...] = ()
        self.resolved_inputs: list[Buffer] = []
        self.scalar_patches = {
            (j, i): (
                self.cbv_offsets[j] + 4 * scalar.word,
                ("<f" if scalar.words == 1 else "<d") if scalar.dtype in ("float", "double") else ("<I" if scalar.words == 1 else "<Q"),
                (1 << (32 * scalar.words)) - 1,
            )
            for j, layout in enumerate(self.layouts)
            for i, scalar in enumerate(layout.scalars)
        }
        self.barriers, self.dependencies = [], []
        accesses: list[int] = []
        covered: dict[int, int] = {}
        for j, ((_, ast, bufs, _), layout) in enumerate(zip(self.calls, self.layouts)):
            access = functools.reduce(
                operator.or_,
                (d3d.D3D12_BARRIER_ACCESS_UNORDERED_ACCESS if b.writable else d3d.D3D12_BARRIER_ACCESS_SHADER_RESOURCE for b in layout.buffers),
                0,
            )
            dependencies = self._access_resources(bufs, ast.arg.outs, j)
            self.dependencies.append(dependencies)
            self.barriers.append(barrier_access(dependencies, accesses, covered, access))
            accesses.append(access)
        self.profile_entries = [ProfileGraphEntry(self.device, rt.name, j, j + 1, rt.profile_key) for j, rt in enumerate(self.programs)]
        self.slots = [self._new_slot() for _ in range(3)]

    def _new_slot(self) -> GraphSlot:
        """Allocate slot-local arguments, descriptors, and an untimed native list."""
        command, arguments = self.dev.command(pool=False), self.dev.upload(self.argument_size)
        buffers = [[bufs[b.slot]._buf for b in layout.buffers] for (_, _, bufs, _), layout in zip(self.calls, self.layouts)]
        slot = GraphSlot(
            command,
            arguments,
            [rt.descriptors(tuple(bufs)) for rt, bufs in zip(self.programs, buffers)],
            global_sizes=tuple(groups for groups, _ in self.launches),
            buffers=buffers,
        )
        mapped = arguments.mapped
        assert mapped is not None
        for j, (layout, bufs) in enumerate(zip(self.layouts, buffers)):
            offsets = tuple(b.binding_address & 15 for b in bufs)
            data = layout.scalar_data(offsets, self.values[j])
            mapped[self.cbv_offsets[j] : self.cbv_offsets[j] + len(data)] = data
        self._record(slot, tuple(local for _, local in self.launches), initial=True)
        return slot

    def _record(self, slot: GraphSlot, local_sizes: tuple[tuple[int, int, int], ...], initial: bool = False, query_count: int = 0) -> None:
        """Record an idle slot with optional endpoint or per-kernel timestamps."""
        if not initial:
            slot.command.reset()
        if query_count:
            if slot.timing is None or slot.timing.count != query_count:
                slot.timing = self.dev.timestamp(slot.command, count=query_count)
            else:
                slot.timing.mark(slot.command, 0)
        slot.aligned = tuple(rt.layout.alignable and all(buf.offset % 16 == 0 for buf in bufs) for rt, bufs in zip(self.programs, slot.buffers))
        for j, rt in enumerate(self.programs):
            if before := self.barriers[j][0]:
                barrier = d3d.D3D12_GLOBAL_BARRIER(
                    d3d.D3D12_BARRIER_SYNC_COMPUTE_SHADING, d3d.D3D12_BARRIER_SYNC_COMPUTE_SHADING, before, self.barriers[j][1]
                )
                group = d3d.D3D12_BARRIER_GROUP(Type=d3d.D3D12_BARRIER_TYPE_GLOBAL, NumBarriers=1, pGlobalBarriers=C.pointer(barrier))
                slot.command.list.call(80, None, (C.c_uint32, C.POINTER(d3d.D3D12_BARRIER_GROUP)), 1, C.byref(group))
            pso = rt.specialize(local_sizes[j], aligned=slot.aligned[j])
            slot.command.list.call(29, None, (C.c_void_p,), rt.root_signature.ptr)
            slot.command.list.call(25, None, (C.c_void_p,), pso.ptr)
            rt.bind_buffers(slot.command, tuple(slot.buffers[j]), slot.descriptors[j])
            if rt.layout.scalar_words:
                slot.command.list.call(37, None, (C.c_uint32, C.c_uint64), rt.layout.scalar_root, slot.arguments.gpu_address + self.cbv_offsets[j])
            slot.command.list.call(14, None, (C.c_uint32, C.c_uint32, C.c_uint32), *slot.global_sizes[j])
            if query_count > 2 and j + 1 < len(self.calls) and slot.timing is not None:
                slot.timing.mark(slot.command, j + 1)
        if query_count and slot.timing is not None:
            slot.timing.finish(slot.command)
        slot.command.close()
        slot.local_sizes, slot.query_count = local_sizes, query_count

    def _acquire(self) -> GraphSlot:
        """Return a completed ring slot, waiting only when every slot is active."""
        with self.dev.lock:
            completed = self.dev.retire()
        index = next((i for n in range(len(self.slots)) if self.slots[i := (self.next_slot + n) % len(self.slots)].fence <= completed), None)
        if index is None:
            index = min(range(len(self.slots)), key=lambda i: self.slots[i].fence)
            self.dev.wait(self.slots[index].fence)
            with self.dev.lock:
                self.dev.retire()
        self.next_slot = (index + 1) % len(self.slots)
        return self.slots[index]

    def __call__(self, input_uops: tuple[UOp, ...], var_vals: dict[str, int], wait: bool = False) -> float | None:
        """Reuse view geometry, refresh current storage and submit once.

        Input identity changes resolve the complete templates again. Unchanged
        inputs still check Buffer storage versions, including native reallocation.
        Cached views belong to this graph; input UOps are referenced weakly.
        Return GPU seconds only when requested.
        """
        with self.lock:
            slot = self._acquire()
            mapped = slot.arguments.mapped
            assert mapped is not None
            changed_buffers = set()
            if (
                not self.resolved_inputs
                or len(self.input_refs) != len(input_uops)
                or any(ref() is not u for ref, u in zip(self.input_refs, input_uops))
            ):
                self.resolved_inputs = [resolve_graph_input(source, input_uops) for _, _, source, _ in self.buffer_patches]
                self.input_refs = tuple(weakref.ref(u) for u in input_uops)
            for (j, i, _, view_offset), raw in zip(self.buffer_patches, self.resolved_inputs):
                buf = raw.ensure_allocated()._buf
                if buf.dev is not self.dev:
                    raise RuntimeError("foreign device buffer in DX12 graph")
                previous = slot.buffers[j][i]
                if buf.resource is not previous.resource or buf.offset != previous.offset or buf.size != previous.size:
                    changed_buffers.add(j)
                slot.buffers[j][i] = buf
                address = buf.binding_address
                struct.pack_into("<I", mapped, view_offset, address & 15)
            for j, i, value in self.updated_vars(var_vals):
                offset, fmt, mask = self.scalar_patches[j, i]
                struct.pack_into(fmt, mapped, offset, float(value) if fmt in ("<f", "<d") else int(value) & mask)
            locals_ = list(slot.local_sizes)
            globals_ = list(slot.global_sizes)
            for j, groups, local in self.updated_launch_dims(var_vals):
                if len(groups) != 3 or any(not isinstance(n, int) or n < 0 or n > 65535 for n in groups):
                    raise ValueError(f"invalid DX12 dispatch groups: {groups}")
                globals_[j] = (int(groups[0]), int(groups[1]), int(groups[2]))
                locals_[j] = (local[0], local[1], local[2])
            for j in changed_buffers:
                descriptor = slot.descriptors[j]
                if descriptor is not None:
                    descriptor.update(tuple(slot.buffers[j]))
            query_count = len(self.calls) + 1 if PROFILE else 2 if wait else 0
            alignment_changed = any(
                slot.aligned[j] != (self.layouts[j].alignable and all(buf.offset % 16 == 0 for buf in slot.buffers[j])) for j in changed_buffers
            )
            if (
                tuple(locals_) != slot.local_sizes
                or tuple(globals_) != slot.global_sizes
                or alignment_changed
                or query_count != slot.query_count
                or any(not self.layouts[j].table for j in changed_buffers)
            ):
                slot.global_sizes = tuple(globals_)
                self._record(slot, tuple(locals_), query_count=query_count)
            slot.owners = tuple(buf for bufs in slot.buffers for buf in bufs)
            callback = None
            if PROFILE:
                assert slot.timing is not None
                callback = functools.partial(self._profile, slot.timing, self.dev, self.profile_entries, self.dependencies)
            slot.fence = self.dev.submit(slot.command, owners=(*slot.owners, slot, *self.programs), callback=callback)
            if wait:
                self.dev.wait(slot.fence)
                assert slot.timing is not None
                return slot.timing.seconds()
            return None

    @staticmethod
    def _profile(timing: DX12Timing, dev: DX12Device, entries: list[ProfileGraphEntry], dependencies: list[list[int]]) -> None:
        """Publish real per-kernel query values after a submission completes."""
        signals = [decimal.Decimal(tick) * 1000000 / dev.frequency for tick in timing.timestamps()]
        dev.profile_events.append(ProfileGraphEvent(entries, dependencies, signals))
