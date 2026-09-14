"""Reuse tinygrad's integer word arithmetic for late transcendental expansions."""

from tinygrad.codegen.decomp.dtype import l2i, l2i_dt
from tinygrad.dtype import dtypes, truncate
from tinygrad.uop import GroupOp
from tinygrad.uop.ops import Ops, UOp


def decompose_late_int64(sink: UOp) -> UOp:
    """Lower late integer64 values to DWORDs while preserving native double arithmetic.

    tinygrad decomposes unsupported dtypes before its transcendental expansions.
    Those expansions can introduce new integer64 intermediates, including double
    bitcasts not covered by its integer-buffer decomposition. Reuse ``l2i`` for all
    arithmetic and add only the HLSL double/word bitcast boundary. No buffers or
    extra dispatches are introduced.
    """
    from tinygrad_dx12.renderer.cstyle import dtype_of

    lowered: dict[UOp, UOp | tuple[UOp, UOp]] = {}

    def scalar(value: UOp) -> UOp:
        """Return an ordinary lowered node, rejecting an unexpected unsplit consumer."""
        result = lowered[value]
        if isinstance(result, tuple):
            raise RuntimeError(f"unsupported late integer64 consumer of {value.op}")
        return result

    def words(value: UOp) -> tuple[UOp, UOp]:
        """Return a lowered integer's low/high words in its signedness."""
        result = lowered[value]
        if value.op is Ops.CONST:
            low, high = (UOp.const(truncate[word_dtype](int(value.val) >> shift), word_dtype) for shift in (0, 32))
            return low, high
        if not isinstance(result, tuple):
            raise RuntimeError(f"expected late integer64 value, got {value.op}")
        return result

    for value in sink.toposort():
        dt = dtype_of(value)
        word_dtype = l2i_dt.get(dt, next((l2i_dt[dtype_of(source)] for source in value.src if dtype_of(source) in l2i_dt), dtypes.uint32))
        if dt in l2i_dt:
            if value.op is Ops.CAST and value.src[0].op is Ops.CONST:
                low, high = (UOp.const(truncate[word_dtype](int(value.val) >> shift), word_dtype) for shift in (0, 32))
                lowered[value] = low, high
            elif value.op in (Ops.CAST, Ops.BITCAST) and value.src[0].dtype in l2i_dt:
                lowered[value] = l2i(Ops.BITCAST, word_dtype, *words(value.src[0]))
            elif value.op is Ops.CAST:
                lowered[value] = l2i(Ops.CAST, dt, scalar(value.src[0]))
            elif value.op is Ops.BITCAST and value.src[0].dtype is dtypes.float64:
                low, high = (
                    UOp(Ops.CUSTOM, src=(scalar(value.src[0]),), arg=(f"_double_words({{0}}).{lane}", dtypes.uint32)).bitcast(word_dtype)
                    for lane in ("x", "y")
                )
                lowered[value] = low, high
            elif value.op in (Ops.SHL, Ops.SHR):
                shift = words(value.src[1])[0] if dtype_of(value.src[1]) in l2i_dt else scalar(value.src[1])
                lowered[value] = l2i(value.op, word_dtype, *words(value.src[0]), shift)
            elif value.op is Ops.WHERE:
                lowered[value] = l2i(value.op, word_dtype, scalar(value.src[0]), *words(value.src[1]), *words(value.src[2]))
            elif value.op is Ops.MULACC:
                product = l2i(Ops.MUL, word_dtype, *words(value.src[0]), *words(value.src[1]))
                lowered[value] = l2i(Ops.ADD, word_dtype, *product, *words(value.src[2]))
            elif value.op in GroupOp.ALU:
                lowered[value] = l2i(value.op, word_dtype, *(word for source in value.src for word in words(source)))
            else:
                raise RuntimeError(f"unsupported late integer64 operation {value.op}")
        elif value.op is Ops.BITCAST and value.src[0].dtype in l2i_dt and dt is dtypes.float64:
            lowered[value] = UOp(Ops.CUSTOM, src=words(value.src[0]), arg=("asdouble(uint({0}),uint({1}))", dtypes.float64))
        elif value.op is Ops.CAST and value.src[0].dtype in l2i_dt:
            lowered[value] = l2i(value.op, dt, *words(value.src[0]))
        elif value.op in GroupOp.Comparison and any(dtype_of(source) in l2i_dt for source in value.src):
            lowered[value] = l2i(value.op, word_dtype, *(word for source in value.src for word in words(source)))
        else:
            lowered[value] = value.replace(src=tuple(scalar(source) for source in value.src))
    return scalar(sink)
