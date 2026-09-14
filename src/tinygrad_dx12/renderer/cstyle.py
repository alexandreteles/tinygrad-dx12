"""HLSL source and the common root-signature/indirect binding description."""

import json, math, struct
from dataclasses import asdict, dataclass, replace
from threading import RLock
from tinygrad.dtype import AddrSpace, DType, dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import CStyleLanguage, base_rewrite
from tinygrad.codegen.decomp.transcendental import xexp2, xlog2, xpow
from tinygrad.uop.ops import Ops, UOp, UPat, PatternMatcher
from tinygrad.uop import GroupOp
from tinygrad_dx12.codegen.decomp.late import decompose_late_int64

_DOUBLE_WORDS = "uint2 _double_words(double v) { uint lo,hi; asuint(v,lo,hi); return uint2(lo,hi); }"
WINDOW = 1 << 30


def dtype_of(u: UOp) -> DType:
    """Resolve tinygrad's recursively inferred dtype descriptor to a concrete dtype."""
    dt = getattr(u, "dtype")
    if not isinstance(dt, DType):
        raise TypeError("tinygrad UOp has no concrete dtype")
    return dt


def buffer_views(u: UOp) -> int:
    """Cover the declared byte span, including rebinding alignment, with overlapping raw views.

    Each view starts 1 GiB after the preceding view and addresses almost 2 GiB.
    The overlap lets a vector or matrix tile cross a window without splitting its
    native load/store. Small parameters retain one ordinary root/table binding.
    """
    size = u.max_numel() * dtype_of(u).itemsize + 15
    return (size + WINDOW - 1) // WINDOW if size > 2 * WINDOW - 4 else 1


def buffer_access(name: str, byte: str, wide: bool, uniform: bool = False) -> tuple[str, str]:
    """Select a raw descriptor and its uint32 offset from a logical byte address."""
    index = f"uint(({byte})>>30)"
    return (f"{name}[{index if uniform else f'NonUniformResourceIndex({index})'}]", f"(uint({byte})&0x3fffffffu)") if wide else (name, byte)


def pack_byte_stores(uops: list[UOp]) -> dict[UOp, tuple[UOp, ...]]:
    """Combine four adjacent byte stores within a straight sequence of stores/indexes.

    Never cross a load, another buffer, control flow, or any other instruction.
    All values are available at the last store. A merged store owns the same four
    bytes; runtime alignment selects one DWORD write or two masked writes.
    """
    packed: dict[UOp, tuple[UOp, ...]] = {}
    pending: list[UOp] = []

    def flush() -> None:
        """Pack consecutive addresses and leave all other stores in their original order."""
        if len(pending) >= 4:
            offsets = [(store.src[0].src[1] - pending[0].src[0].src[1]).ssimplify() for store in pending]
            if all(isinstance(offset, int) for offset in offsets):
                ordered = sorted(((offset, store) for offset, store in zip(offsets, pending) if isinstance(offset, int)), key=lambda item: item[0])
                for start in range(0, len(ordered) - 3, 4):
                    group = ordered[start : start + 4]
                    if [offset for offset, _ in group] != list(range(group[0][0], group[0][0] + 4)):
                        continue
                    stores = tuple(store for _, store in group)
                    packed.update((store, ()) for store in stores)
                    packed[max(stores, key=pending.index)] = stores
        pending.clear()

    for u in uops:
        if u.op is Ops.INDEX:
            continue
        if (
            u.op is Ops.STORE
            and len(u.src) == 2
            and u.src[0].op is Ops.INDEX
            and u.src[0].src[0].op is Ops.PARAM
            and not u.src[0].src[0].arg.volatile
            and u.src[0].addrspace is AddrSpace.GLOBAL
            and dtype_of(u.src[1]).itemsize == 1
            and u.src[1].max_numel() == 1
        ):
            if pending and u.src[0].src[0] is not pending[0].src[0].src[0]:
                flush()
            pending.append(u)
        else:
            flush()
    flush()
    return packed


@dataclass(frozen=True)
class BufferBinding:
    """One buffer, in tinygrad's sorted parameter-slot order; offsets are bytes."""

    slot: int
    name: str
    writable: bool
    register: int
    root_index: int
    offset_word: int
    views: int = 1


@dataclass(frozen=True)
class ScalarBinding:
    """One scalar parameter with an explicit little-endian DWORD location."""

    slot: int
    name: str
    dtype: str
    word: int
    words: int


@dataclass(frozen=True)
class BindingLayout:
    """Single source of truth for HLSL, root parameters, and indirect records.

    Buffers and scalars use their sorted tinygrad parameter order. Every buffer
    contributes a byte-offset word, allowing raw bindings to be aligned down to 16
    bytes. Large parameter sets use a CBV, then a descriptor table when required.
    Indirect records are tightly packed and their final stride is rounded to eight.
    ``strict_math`` preserves evaluation order across compilation and specialization
    for pointwise kernels and shared floating expansions that construct values from bits.
    """

    buffers: tuple[BufferBinding, ...]
    scalars: tuple[ScalarBinding, ...]
    scalar_words: int
    scalar_root: int
    scalar_cbv: bool
    table: bool
    local_size: tuple[int, int, int] = (1, 1, 1)
    alignable: bool = False
    has_fp64: bool = False
    strict_math: bool = False

    def declarations(self) -> list[str]:
        """Share raw resource arrays, scalar packing, and residual-offset macros across shaders."""
        return (
            [
                f"{'RW' if b.writable else ''}ByteAddressBuffer {b.name}{f'[{b.views}]' if b.views > 1 else ''}"
                f" : register({'u' if b.writable else 't'}{b.register});"
                for b in self.buffers
            ]
            + (
                [
                    "cbuffer DX12Params : register(b0) {",
                    *(f"  uint _p{i} : packoffset(c{i // 4}.{'xyzw'[i % 4]});" for i in range(self.scalar_words)),
                    "};",
                ]
                if self.scalar_words
                else []
            )
            + [f"#define _offset_{b.slot} _p{b.offset_word}" for b in self.buffers]
        )

    @property
    def root_dwords(self) -> int:
        """Return the complete root-signature cost, including parameter storage."""
        return (1 if self.table else 2 * len(self.buffers)) + (2 if self.scalar_cbv else self.scalar_words)

    @property
    def scalar_offset(self) -> int:
        """Return the scalar constants/CBV address offset within an indirect record."""
        return 0 if self.table else 8 * len(self.buffers)

    @property
    def dispatch_offset(self) -> int:
        """Return the dispatch XYZ DWORD offset in bytes."""
        return self.scalar_offset + (8 if self.scalar_cbv else 4 * self.scalar_words)

    @property
    def stride(self) -> int:
        """Return the indirect command signature's eight-byte-rounded byte stride."""
        return (self.dispatch_offset + 12 + 7) & ~7

    @property
    def cbv_size(self) -> int:
        """Return the 256-byte-aligned size of a scalar constant buffer, or zero."""
        return ((4 * self.scalar_words + 255) & ~255) if self.scalar_cbv else 0

    def scalar_data(self, offsets: tuple[int, ...], vals: tuple[int | float, ...]) -> bytes:
        """Pack view offsets and scalar values without relying on C buffer packing."""
        if len(offsets) != len(self.buffers) or len(vals) != len(self.scalars):
            raise ValueError("DX12 argument count mismatch")
        data = bytearray(4 * self.scalar_words)
        for binding, offset in zip(self.buffers, offsets):
            struct.pack_into("<I", data, 4 * binding.offset_word, offset)
        for scalar, value in zip(self.scalars, vals):
            if scalar.dtype in ("float", "double"):
                struct.pack_into("<f" if scalar.words == 1 else "<d", data, 4 * scalar.word, value)
            else:
                struct.pack_into("<I" if scalar.words == 1 else "<Q", data, 4 * scalar.word, int(value) & ((1 << (32 * scalar.words)) - 1))
        return bytes(data)

    def pack_record(
        self, addresses: tuple[int, ...], offsets: tuple[int, ...], vals: tuple[int | float, ...], groups: tuple[int, int, int], cbv_gpuva: int = 0
    ) -> bytes:
        """Create a record; CBV layouts require an aligned external scalar-buffer address."""
        if len(addresses) != len(self.buffers):
            raise ValueError("DX12 buffer count mismatch")
        if any(x < 0 or x > 65535 for x in groups):
            raise ValueError(f"invalid DX12 dispatch group count: {groups}")
        scalars = self.scalar_data(offsets, vals)
        data = bytearray(self.stride)
        if not self.table:
            for i, address in enumerate(addresses):
                if address & 15:
                    raise ValueError("DX12 raw buffer addresses require 16-byte alignment")
                struct.pack_into("<Q", data, 8 * i, address)
        if self.scalar_cbv:
            if not cbv_gpuva or cbv_gpuva & 255:
                raise ValueError("DX12 root CBV requires a 256-byte-aligned address")
            struct.pack_into("<Q", data, self.scalar_offset, cbv_gpuva)
        else:
            data[self.scalar_offset : self.dispatch_offset] = scalars
        struct.pack_into("<III", data, self.dispatch_offset, *groups)
        return bytes(data)

    def to_json(self) -> str:
        """Serialize stable metadata for source and compilation cache keys."""
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @staticmethod
    def from_json(data: str) -> BindingLayout:
        """Read metadata emitted by this renderer."""
        obj = json.loads(data)
        return BindingLayout(
            tuple(BufferBinding(**x) for x in obj["buffers"]),
            tuple(ScalarBinding(**x) for x in obj["scalars"]),
            obj["scalar_words"],
            obj["scalar_root"],
            obj["scalar_cbv"],
            obj["table"],
            tuple(obj["local_size"]),
            obj.get("alignable", False),
            obj.get("has_fp64", False),
            obj.get("strict_math", False),
        )


def parse_source(source: str) -> BindingLayout:
    """Extract the shader's binding description; reject unrelated HLSL."""
    if not source.startswith("// DX12:"):
        raise ValueError("shader has no DX12 binding metadata")
    return BindingLayout.from_json(source.splitlines()[0][8:])


def specialize_source(source: str, local_size: tuple[int, int, int]) -> str:
    """Specialize fixed HLSL numthreads, preserving the original binding layout."""
    if len(local_size) != 3 or any(x < 1 for x in local_size) or math.prod(local_size) > 1024 or local_size[2] > 64:
        raise ValueError(f"invalid DX12 local size: {local_size}")
    layout = replace(parse_source(source), local_size=local_size)
    lines = source.splitlines()
    lines[0] = "// DX12:" + layout.to_json()
    for axis, size in zip("XYZ", local_size):
        prefix = f"#define DX12_LOCAL_{axis} "
        lines = [prefix + str(size) if line.startswith(prefix) else line for line in lines]
    return "\n".join(lines)


def make_layout(bufs: list[tuple[str, tuple[UOp, bool]]], local_size: tuple[int, int, int]) -> BindingLayout:
    """Choose root descriptors/constants, CBV, or descriptor-table overflow."""
    ordered = sorted(bufs, key=lambda x: x[1][0].arg.slot)
    buffers = [(name, u, mutable) for name, (u, mutable) in ordered if u.addrspace is AddrSpace.GLOBAL]
    word, scalars = len(buffers), []
    for name, (u, _) in ordered:
        if u.addrspace is not AddrSpace.ALU:
            continue
        words = max(1, dtype_of(u).itemsize // 4)
        scalars.append(ScalarBinding(u.arg.slot, name, dtype_of(u).name, word, words))
        word += words
    cbv = 2 * len(buffers) + word > 64
    table = 2 * len(buffers) + (2 if cbv else word) > 64 or any(buffer_views(u) > 1 for _, u, _ in buffers)
    counts = {False: 0, True: 0}
    binds = []
    for i, (name, u, mutable) in enumerate(buffers):
        views = buffer_views(u)
        binds.append(BufferBinding(u.arg.slot, name, mutable, counts[mutable], 0 if table else i, i, views))
        counts[mutable] += views
    return BindingLayout(tuple(binds), tuple(scalars), word, 1 if table else len(buffers), cbv, table, local_size)


dx12_rewrite = (
    PatternMatcher([
        (
            UPat.var("a", dtypes.floats) * UPat.var("b") + UPat.var("c"),
            lambda ctx, a, b, c: f"{'fma' if dtype_of(a) is dtypes.double else 'mad'}({ctx[a]},{ctx[b]},{ctx[c]})" if ctx._strict_math else None,
        ),
        (UPat(Ops.CMPNE, src=(UPat.var("x", dtypes.floats), UPat.var("x"))), lambda ctx, x: f"isnan({ctx[x]})"),
        (UPat(Ops.CMPEQ, src=(UPat.var("x", dtypes.floats), UPat.var("x"))), lambda ctx, x: f"(!isnan({ctx[x]}))"),
        (UPat(GroupOp.ALU, (dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16), name="x"), lambda ctx, x: ctx.render_narrow_alu(x)),
        (UPat(Ops.LOAD, name="x"), lambda ctx, x: ctx.render_load(x)),
        (UPat(Ops.STORE, name="x"), lambda ctx, x: ctx.render_store(x)),
        (UPat(Ops.BITCAST, name="x"), lambda ctx, x: ctx.render_bitcast(x)),
        (UPat(Ops.TRUNC, dtypes.float64, name="x"), lambda ctx, x: ctx.render_double_trunc(x)),
        (UPat(Ops.CAST, name="x"), lambda ctx, x: ctx.render_constant(x) if x.src[0].op is Ops.CONST else f"({ctx.render_cast(x, ctx[x.src[0]])})"),
    ])
    + base_rewrite
)


class DX12Renderer(CStyleLanguage):
    """Render tinygrad scalar/vector programs to raw-buffer HLSL.

    The target architecture is JSON containing queried device capabilities. This
    keeps tinygrad's program cache separate across lowering capabilities. Unsupported
    FP8/BF16/FP16/integer64 operations use tinygrad's dtype decomposition pass.
    Packed stores use DWORD compare/exchange so concurrent neighboring elements
    cannot overwrite each other; aligned native halfwords use direct stores.
    Workgroup dimensions and zero residual binding offsets are specialized in source.
    """

    global_max = (65535, 65535, 65535)
    supports_float4 = False  # Let DXC pack raw memory accesses instead of constraining them to tinygrad vectors.
    local_max = (1024, 1024, 64)
    shared_max = 32768
    float4 = "float4"
    barrier = "GroupMemoryBarrierWithGroupSync();"
    infinity = "asfloat(0x7f800000u)"
    nan = "asfloat(0x7fc00000u)"
    code_for_workitem = {"g": lambda axis: f"group_id[{axis}]", "l": lambda axis: f"local_id[{axis}]"}
    code_for_op = {
        **{op: render for op, render in CStyleLanguage.code_for_op.items() if op is not Ops.SIN},
        Ops.MAX: lambda a, b, dtype: f"max({a},{b})",
        Ops.FDIV: lambda a, b, dtype: f"({a}/{b})",
    }
    string_rewrite = dx12_rewrite
    extra_matcher = PatternMatcher([
        (UPat(Ops.EXP2, dtypes.float64, src=(UPat.var("d"),)), xexp2),
        (UPat(Ops.LOG2, dtypes.float64, src=(UPat.var("d"),)), xlog2),
        (UPat(Ops.SQRT, dtypes.float64, src=(UPat.var("d"),)), lambda d: xpow(d, d.const_like(0.5))),
        (
            UPat(Ops.SINK, name="sink"),
            lambda ctx, sink: (
                decompose_late_int64(sink)
                if not ctx.capabilities.get("int64", False) and any(dtype_of(u) in dtypes.int64s for u in sink.toposort())
                else None
            ),
        ),
    ])

    def __init__(self, target: Target) -> None:
        """Create a device-local renderer/compiler from serialized feature queries."""
        super().__init__(target)
        self._render_lock = RLock()
        self.capabilities = json.loads(target.arch) if target.arch else {}
        self.native16 = bool(self.capabilities.get("native16", False))
        self.matrix_failures: dict[str, str] = {}
        self.matrix_support: dict[tuple, dict | None] = {}
        self.matrix_choices: dict[tuple, tuple[str, tuple[int, int, int], int]] = {}
        self.type_map = {
            dtypes.bool: "bool",
            dtypes.float: "float",
            dtypes.double: "double",
            dtypes.half: "float16_t",
            dtypes.int8: "int",
            dtypes.uint8: "uint",
            dtypes.int16: "int16_t" if self.native16 else "int",
            dtypes.uint16: "uint16_t" if self.native16 else "uint",
            dtypes.int32: "int",
            dtypes.uint32: "uint",
            dtypes.int64: "int64_t",
            dtypes.uint64: "uint64_t",
        }
        from tinygrad_dx12.runtime.support.compiler_dx12 import DX12Compiler

        sm = self.capabilities.get("shader_model", 0x6A)
        self.compiler = DX12Compiler(target=f"cs_{sm >> 4}_{sm & 15}", native16=self.native16, capabilities=target.arch)

    def supported_dtypes(self) -> set[DType]:
        """Report arithmetic support separately from packed storage and LinAlg types."""
        unsupported = {*dtypes.fp8s, dtypes.bfloat16}
        if not self.native16:
            unsupported.add(dtypes.half)
        if not self.capabilities.get("int64", False):
            unsupported.update((dtypes.int64, dtypes.uint64))
        if not self.capabilities.get("doubles", False):
            unsupported.add(dtypes.double)
        return super().supported_dtypes() - unsupported

    def _render_dtype(
        self,
        dtype: DType,
        sz: int = 1,
        addrspace: AddrSpace = AddrSpace.ALU,
        mutable: bool = True,
        override_ptr: bool = False,
        shape: tuple[int, ...] | None = None,
    ) -> str:
        """Name HLSL value types; raw resources never become C-style pointers."""
        if dtype not in self.type_map and dtype is not dtypes.void:
            raise NotImplementedError(f"DX12 scalar dtype {dtype}")
        name = self.type_map.get(dtype, "void")
        return name if sz == 1 else f"vector<{name},{sz}>"

    def render_constant(self, u: UOp) -> str:
        """Render width-committed constants, including nonfinite floats and 64-bit integers."""
        value = u.src[0].val
        if dtype_of(u) in dtypes.floats:
            if math.isnan(value):
                value = self.nan
            elif math.isinf(value):
                value = self.infinity if value > 0 else f"-{self.infinity}"
            else:
                value = repr(float(value)) + ("L" if dtype_of(u) is dtypes.double else "f")
        elif dtype_of(u) is dtypes.bool:
            value = "true" if value else "false"
        elif dtype_of(u) is dtypes.uint64:
            value = f"{int(value) & ((1 << 64) - 1)}ull"
        elif dtype_of(u) is dtypes.int64:
            value = f"{value}ll" if value != -(1 << 63) else "(-9223372036854775807ll-1ll)"
        elif dtype_of(u) in dtypes.uints:
            value = f"{int(value) & ((1 << dtype_of(u).bitsize) - 1)}u"
        else:
            value = str(value)
        return self.render_cast(u, str(value))

    def render_cast(self, u: UOp, val: str) -> str:
        """Preserve narrow integer truncation despite wider HLSL arithmetic types."""
        if dtype_of(u) in (dtypes.int8, dtypes.int16) and self.type_map[dtype_of(u)] == "int":
            shift = 32 - dtype_of(u).bitsize
            return f"((({self.render_type(u)})({val}) << {shift}) >> {shift})"
        if dtype_of(u) in (dtypes.uint8, dtypes.uint16) and self.type_map[dtype_of(u)] == "uint":
            return f"(({self.render_type(u)})({val}) & {(1 << dtype_of(u).bitsize) - 1}u)"
        return f"({self.render_type(u)})({val})"

    def render_narrow_alu(self, u: UOp) -> str:
        """Commit narrow arithmetic before consumers, preserving grouping across chained ALU operations."""
        value = self.code_for_op[u.op](*(self[source] for source in u.src), dtype_of(u))
        return self.render_cast(u, value)

    def render_bitcast(self, u: UOp) -> str:
        """Reinterpret equal-size values through HLSL's scalar/vector intrinsics."""
        src, val, dt = u.src[0], self[u.src[0]], dtype_of(u)
        if u.addrspace in (AddrSpace.GLOBAL, AddrSpace.LOCAL):
            return val
        if dt == dtype_of(src):
            return val
        if self.native16 and dt.itemsize == 2:
            return f"{'asfloat16' if dt is dtypes.half else 'asint16' if dt is dtypes.int16 else 'asuint16'}({val})"
        if dt is dtypes.float:
            return f"asfloat({val})"
        if dt in (dtypes.int32, dtypes.uint32):
            return f"{'asint' if dt is dtypes.int32 else 'asuint'}({val})"
        if dt is dtypes.double:
            return f"asdouble((uint)({val}), (uint)((uint64_t)({val}) >> 32))"
        if dtype_of(src) is dtypes.double:
            self._helpers.add("uint64_t _double_bits(double v) { uint lo, hi; asuint(v, lo, hi); return (uint64_t(hi)<<32)|lo; }")
            return f"({self.render_type(u)})_double_bits({val})"
        return self.render_cast(u, val)

    def render_double_trunc(self, u: UOp) -> str:
        """Truncate the FP64 mantissa using two DWORDs, independently of integer64 support."""
        self._helpers.add(
            "double _trunc_double(double v) { uint lo,hi; asuint(v,lo,hi); int e=int((hi>>20)&2047u)-1023; "
            "if (e<0) return asdouble(0u,hi&0x80000000u); if (e<=20) return asdouble(0u,hi&~(0xfffffu>>e)); "
            "if (e<52) return asdouble(lo&(0xffffffffu<<(52-e)),hi); return v; }"
        )
        return f"_trunc_double({self[u.src[0]]})"

    def _address(self, u: UOp) -> str:
        """Compute raw byte offsets through buffer aliases and dtype-changing views."""
        if u.op is Ops.PARAM:
            return f"_offset_{u.arg.slot}"
        if u.op in (Ops.AFTER, Ops.CAST, Ops.BITCAST):
            return self._address(u.src[0])
        if u.op in (Ops.INDEX, Ops.SHRINK):
            integer = "uint64_t" if buffer_views(u.buf_uop) > 1 else "uint"
            return f"({self._address(u.src[0])} + {integer}({self[u.src[1]]})*{dtype_of(u.src[0]).itemsize}u)"
        raise NotImplementedError(f"DX12 address operation {u.op}")

    def _raw(self, name: str, byte: str, method: str, *args: str) -> str:
        """Emit one native buffer operation, widening only parameters that span multiple views."""
        resource, offset = buffer_access(name, byte, name in self._wide)
        return f"{resource}.{method}({','.join((offset, *args))})"

    def render_index(self, x: UOp, buf: UOp, idx: UOp) -> str:
        """Render a raw byte address or a local/register/vector element index."""
        if x.addrspace is AddrSpace.GLOBAL:
            return self._address(x)
        if buf.addrspace is AddrSpace.ALU:
            return f"({self[buf]})[{self[idx]}]"
        return f"{self[buf]}[{self[idx]}]"

    def render_buffer(self, x: UOp) -> str:
        """Place shared arrays at global scope and register arrays inside the entry point."""
        declaration = f"{self.render_dtype(dtype_of(x))} {self[x]}[{x.max_numel()}];"
        if x.addrspace is AddrSpace.LOCAL:
            self._shared.append("groupshared " + declaration)
            return ""
        return declaration

    def _load_word(self, name: str, byte: str, bits: int = 32) -> str:
        """Read only DWORDs touched by the element, including the physical allocation tail."""
        self._helpers.add(
            f"uint _ld{bits}_{name}({'uint64_t' if name in self._wide else 'uint'} p) {{ "
            f"uint s=uint(p&3u)*8u, a={self._raw(name, 'p-(p&3u)', 'Load')}; "
            f"if (s<={32 - bits}u) return a>>s; return (a>>s)|({self._raw(name, 'p-(p&3u)+4u', 'Load')}<<(32u-s)); }}"
        )
        return f"_ld{bits}_{name}({byte})"

    def _load_scalar(self, dt: DType, name: str, byte: str) -> str:
        """Load the stored representation and sign-extend or reinterpret it as required."""
        word = self._load_word(name, byte, min(32, dt.itemsize * 8))
        if dt is dtypes.float:
            return f"asfloat({word})"
        if dt is dtypes.double:
            return f"asdouble({word}, {self._load_word(name, f'({byte}+4u)')})"
        if dt in (dtypes.int64, dtypes.uint64):
            return f"({self.type_map[dt]})((uint64_t({self._load_word(name, f'({byte}+4u)')})<<32)|{word})"
        if dt is dtypes.half:
            return f"asfloat16(uint16_t({word}))"
        if dt is dtypes.bool:
            return f"(({word}&255u)!=0u)"
        if dt in dtypes.sints:
            shift = 32 - dt.bitsize
            return f"({self.type_map[dt]})(int({word} << {shift}) >> {shift})" if shift else f"asint({word})"
        return f"({self.type_map[dt]})({word}&{(1 << dt.bitsize) - 1}u)"

    def render_load(self, u: UOp) -> str:
        """Render scalar/vector raw loads and preserve tinygrad's predicate default."""
        idx = u.src[0]
        if idx.addrspace is not AddrSpace.GLOBAL:
            lanes = [self._local_access(idx, i) for i in range(u.max_numel())]
            value = lanes[0] if len(lanes) == 1 else f"{self.render_type(u)}({','.join(lanes)})"
        else:
            name, byte = self[idx.buf_uop], self._address(idx)
            if buffer_views(idx.buf_uop) > 1:
                self._wide.add(name)
            dt = dtype_of(u)
            lanes = [self._load_scalar(dt, name, f"({byte}+{i * dt.itemsize}u)") for i in range(u.max_numel())]
            value = lanes[0] if len(lanes) == 1 else f"{self.render_type(u)}({','.join(lanes)})"
            if dt.itemsize >= 4 or dt.itemsize == 2 and self.native16:
                value = f"((({byte})&{dt.itemsize - 1}u)==0u ? {self._raw(name, byte, f'Load<{self.render_type(u)} >')} : {value})"
        return f"({self[u.src[2]]}?{value}:{self[u.src[1]]})" if len(u.src) == 3 else value

    def _local_access(self, u: UOp, lane: int = 0) -> str:
        """Index contiguous shared/register vector lanes using their scalar array storage."""
        if u.op in (Ops.AFTER, Ops.CAST, Ops.BITCAST):
            return self._local_access(u.src[0], lane)
        if u.op in (Ops.INDEX, Ops.SHRINK):
            return f"{self[u.src[0]]}[{self[u.src[1]]}+{lane}]"
        return f"{self[u]}[{lane}]"

    def _store_word(self, name: str, byte: str, value: str, bits: int) -> str:
        """Emit an atomic masked store when distinct elements can share one DWORD."""
        self._helpers.add(
            f"void _masked_{name}({'uint64_t' if name in self._wide else 'uint'} p,uint v,uint m) {{ "
            f"uint old={self._raw(name, 'p', 'Load')},seen; "
            f"[allow_uav_condition] for (;;) {{ {self._raw(name, 'p', 'InterlockedCompareExchange', 'old', '(old&~m)|(v&m)', 'seen')}; "
            "if (seen==old) break; old=seen; } }"
        )
        self._helpers.add(
            f"void _st_{name}({'uint64_t' if name in self._wide else 'uint'} p,uint v,uint bits) {{ "
            f"uint s=uint(p&3u)*8u, m=0xffffffffu>>(32u-bits); "
            f"if (!s && bits==32u) {{ {self._raw(name, 'p', 'Store', 'v')}; return; }} "
            f"_masked_{name}(p-(p&3u),v<<s,m<<s); if (s+bits>32u) _masked_{name}(p-(p&3u)+4u,v>>(32u-s),m>>(32u-s)); }}"
        )
        return f"_st_{name}({byte}, {value}, {bits}u);"

    def _store_scalar(self, dt: DType, name: str, byte: str, value: str) -> str:
        """Store a scalar with exact byte ownership for packed types and views."""
        if dt is dtypes.float:
            value = f"asuint({value})"
        elif dt is dtypes.half:
            value = f"asuint16({value})"
        elif dt is dtypes.double:
            self._helpers.add(_DOUBLE_WORDS)
            return (
                self._store_word(name, byte, f"_double_words({value}).x", 32)
                + "\n"
                + self._store_word(name, f"({byte}+4u)", f"_double_words({value}).y", 32)
            )
        if dt.itemsize == 8:
            return (
                self._store_word(name, byte, f"uint({value})", 32)
                + "\n"
                + self._store_word(name, f"({byte}+4u)", f"uint((uint64_t)({value})>>32)", 32)
            )
        return self._store_word(name, byte, f"uint({value})", dt.itemsize * 8)

    def render_store(self, u: UOp) -> str:
        """Use typed stores at natural alignment, preserving byte ownership for other views."""
        if (group := self._packed_stores.get(u)) is not None:
            if not group:
                return ""
            idx = group[0].src[0]
            if buffer_views(idx.buf_uop) > 1:
                self._wide.add(self[idx.buf_uop])
            value = "|".join(f"((uint({self[store.src[1]]})&255u)<<{8 * i})" for i, store in enumerate(group))
            return self._store_word(self[idx.buf_uop], self._address(idx), f"({value})", 32)
        idx, value = u.src[:2]
        if idx.addrspace is not AddrSpace.GLOBAL:
            return "\n".join(
                f"{self._local_access(idx, i)} = " + (self[value] if value.max_numel() == 1 else f"({self[value]})[{i}]") + ";"
                for i in range(value.max_numel())
            )
        name, byte = self[idx.buf_uop], self._address(idx)
        if buffer_views(idx.buf_uop) > 1:
            self._wide.add(name)
        dt, lanes = dtype_of(value), value.max_numel()
        fallback = "\n".join(
            self._store_scalar(dt, name, f"({byte}+{i * dt.itemsize}u)", self[value] if lanes == 1 else f"({self[value]})[{i}]") for i in range(lanes)
        )
        # Typed stores preserve the scalar/vector storage representation and own its exact bytes.
        if dt.itemsize >= 4 or dt.itemsize == 2 and self.native16:
            return (
                f"if (({byte}&{dt.itemsize - 1}u)==0u) {{ {self._raw(name, byte, f'Store<{self.render_type(value)} >', self[value])}; }} "
                f"else {{\n{fallback}\n}}"
            )
        # A contiguous vector owns complete DWORDs, so its aligned path needs no read/modify/write.
        if lanes == 1 or lanes * dt.itemsize % 4:
            return fallback
        words, per_word = [], 4 // dt.itemsize
        for start in range(0, lanes, per_word):
            components = []
            for lane in range(per_word):
                val = f"uint(({self[value]})[{start + lane}])"
                components.append(f"(({val}&{(1 << (dt.itemsize * 8)) - 1}u)<<{lane * dt.itemsize * 8})")
            words.append("(" + "|".join(components) + ")")
        packed = words[0] if len(words) == 1 else f"uint{len(words)}({','.join(words)})"
        call = self._raw(name, byte, f"Store{len(words) if len(words) > 1 else ''}", packed) + ";"
        return f"if (({byte}&3u)==0u) {{ {call} }} else {{\n{fallback}\n}}"

    def render_kernel(
        self, function_name: str, kernel: list[str], bufs: list[tuple[str, tuple[UOp, bool]]], uops: list[UOp], prefix: list[str] | None = None
    ) -> str:
        """Assemble resource declarations, explicit DWORD scalar fields, and entry point."""
        local = [1, 1, 1]
        for u in uops:
            if u.op is Ops.SPECIAL and u.arg.startswith("l"):
                local[int(u.arg[-1])] = int(u.src[0].vmax)
        local_size = (local[0], local[1], local[2])
        layout = replace(
            make_layout(bufs, local_size),
            alignable=True,
            has_fp64=any(dtype_of(u) is dtypes.float64 for u in uops),
            strict_math=self._strict_math,
        )
        lines = ["// DX12:" + layout.to_json()]
        lines += [f"#define DX12_LOCAL_{axis} {size}" for axis, size in zip("XYZ", local)]
        lines += layout.declarations()
        for scalar in layout.scalars:
            value = f"_p{scalar.word}"
            if scalar.words == 2:
                value = f"((uint64_t(_p{scalar.word + 1})<<32)|{value})"
            if scalar.dtype == "float":
                value = f"asfloat({value})"
            elif scalar.dtype == "double":
                value = f"asdouble(_p{scalar.word},_p{scalar.word + 1})"
            else:
                dt = next(dt for dt in self.type_map if dt.name == scalar.dtype)
                value = f"({self.type_map[dt]})({value})"
            lines.append(f"#define {scalar.name} ({value})")
        lines += sorted(self._helpers, key=lambda x: ("_st_" in x, x)) + self._shared + (prefix or [])
        lines += [
            "[numthreads(DX12_LOCAL_X,DX12_LOCAL_Y,DX12_LOCAL_Z)]",
            "void main(uint3 group_id : SV_GroupID, uint3 local_id : SV_GroupThreadID) {",
            *kernel,
            "}",
        ]
        return specialize_source("\n".join(lines), local_size)

    def render(self, uops: list[UOp]) -> str:
        """Render one complete shader while owning this renderer's temporary state."""
        with self._render_lock:
            # Preserve values across pointwise fusion and shared bit-level expansions; let DXC optimize arithmetic loops.
            self._strict_math = not any(u.op is Ops.RANGE for u in uops) or any(
                u.op is Ops.BITCAST and u.addrspace is AddrSpace.ALU and dtype_of(u) in dtypes.floats for u in uops
            )
            self._packed_stores = pack_byte_stores(uops)
            self._helpers: set[str] = set()
            self._wide: set[str] = set()
            if any(u.op is Ops.CUSTOM and "_double_words" in u.arg[0] for u in uops):
                self._helpers.add(_DOUBLE_WORDS)
            self._shared: list[str] = []
            return super().render(uops)
