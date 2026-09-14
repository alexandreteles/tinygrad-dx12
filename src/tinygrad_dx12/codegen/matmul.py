"""Recognize bounded scheduled matrix reductions before tinygrad scalar expansion."""

from dataclasses import dataclass
from itertools import permutations
from tinygrad import Device
from tinygrad.device import Buffer, CompileError, MultiBuffer
from tinygrad.dtype import DType
from tinygrad.engine.realize import get_runtime
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from tinygrad_dx12.renderer.cstyle import DX12Renderer, dtype_of
from tinygrad_dx12.renderer.linalg import COMPONENTS, linalg_source


@dataclass(frozen=True)
class MatmulSpec:
    """A rank-two reduction with scheduled parameter slots and native matrix layouts."""

    m: int
    n: int
    k: int
    output: UOp
    left: UOp
    right: UOp
    accumulator: DType
    reduction: UOp
    output_index: UOp
    epilogue: bool
    transposed: tuple[bool, bool] = (False, False)


def _affine(value: UOp, axes: tuple[UOp, ...]) -> tuple[int, ...] | None:
    """Recognize constant affine indexing without accepting broadcasts or data-dependent addressing."""
    if value in axes:
        return (0, *(int(value is axis) for axis in axes))
    if value.op is Ops.CONST and isinstance(value.val, int):
        return (value.val, *(0 for _ in axes))
    if value.op not in (Ops.ADD, Ops.MUL) or len(value.src) != 2:
        return None
    left, right = (_affine(source, axes) for source in value.src)
    if left is None or right is None:
        return None
    if value.op is Ops.ADD:
        return tuple(a + b for a, b in zip(left, right))
    if not any(left[1:]):
        return tuple(left[0] * x for x in right)
    if not any(right[1:]):
        return tuple(right[0] * x for x in left)
    return None


def match_matmul(ast: UOp) -> MatmulSpec | None:
    """Match row/column-major matrix products whose accumulation can use native LinAlg.

    Extra outputs, broadcast addressing, symbolic sizes,
    and explicitly optimized kernels remain on tinygrad's ordinary lowering path.
    Storage types and the scheduled accumulator are passed to DX12's exact
    operation query. LinAlg owns multiplication and accumulation behavior;
    this matcher does not impose scalar-emulation rounding on matrix hardware.
    """
    if ast.op is not Ops.SINK or len(ast.src) != 1 or not isinstance(ast.arg, KernelInfo):
        return None
    if ast.arg.applied_opts or ast.arg.opts_to_apply is not None:
        return None
    end = ast.src[0]
    if end.op is not Ops.END or len(end.src) != 3:
        return None
    store = end.src[0]
    if store.op is not Ops.STORE or len(store.src) != 2:
        return None
    output, value = store.src
    if output.op is not Ops.INDEX or len(output.src) != 2 or output.src[0].op is not Ops.PARAM:
        return None
    reductions = [u for u in value.toposort() if u.op is Ops.REDUCE]
    if len(reductions) != 1:
        return None
    reduced = reductions[0]
    if reduced.arg != (Ops.ADD, 0) or len(reduced.src) != 2:
        return None
    epilogue = value is not reduced and not (value.op is Ops.CAST and value.src[0] is reduced)
    accumulator = dtype_of(reduced)
    if accumulator not in COMPONENTS:
        return None
    product, reduction_axis = reduced.src
    if product.op is Ops.CAST and dtype_of(product) is accumulator:
        product = product.src[0]
    if product.op is not Ops.MUL or len(product.src) != 2:
        return None
    inputs = tuple(source.src[0] if source.op is Ops.CAST and dtype_of(source) is accumulator else source for source in product.src)
    if any(x.op is not Ops.INDEX or len(x.src) != 2 or x.src[0].op is not Ops.PARAM or dtype_of(x) not in COMPONENTS for x in inputs):
        return None
    axes = (*end.src[1:], reduction_axis)
    if len(set(axes)) != 3 or any(
        axis.op is not Ops.RANGE
        or len(axis.src) != 1
        or axis.src[0].op is not Ops.CONST
        or not isinstance(axis.src[0].val, int)
        or axis.src[0].val <= 0
        for axis in axes
    ):
        return None
    if reduction_axis.arg[-1] is not AxisType.REDUCE:
        return None
    for row, column in permutations(end.src[1:]):
        m, n, k = (axis.src[0].val for axis in (row, column, reduction_axis))
        if _affine(output.src[1], (row, column, reduction_axis)) != (0, n, 1, 0):
            continue
        for left, right in permutations(inputs):
            left_index = _affine(left.src[1], (row, column, reduction_axis))
            right_index = _affine(right.src[1], (row, column, reduction_axis))
            if left_index not in ((0, k, 0, 1), (0, 1, 0, m)):
                continue
            if right_index not in ((0, 0, 1, n), (0, 0, k, 1)):
                continue
            params = (output.src[0], left.src[0], right.src[0])
            if params[0] in params[1:] or any(param.max_numel() < size for param, size in zip(params, (m * n, m * k, k * n))):
                continue
            return MatmulSpec(
                m,
                n,
                k,
                *params,
                accumulator,
                reduced,
                output.src[1],
                epilogue,
                (left_index == (0, 1, 0, m), right_index == (0, 0, k, 1)),
            )
    return None


def _aligned(argument: UOp) -> bool:
    """Prove matrix-start alignment from ordinary buffer/view metadata, without reading tensor values."""
    view = argument.contiguous_view()
    if view is None:
        return False
    base, offset = view
    byte_offset = offset * dtype_of(base).itemsize
    lane = None
    if base.op is Ops.MSELECT:
        lane, base = base.arg, base.src[0]
    if base.op is Ops.PARAM:
        return byte_offset % 128 == 0
    if base.op is not Ops.BUFFER:
        return False
    storage = base.arg.buffer
    if isinstance(storage, MultiBuffer):
        buffers = [storage.bufs[lane]] if lane is not None else storage.bufs
        return all((byte_offset + buffer.offset) % 128 == 0 for buffer in buffers)
    return isinstance(storage, Buffer) and (byte_offset + storage.offset) % 128 == 0


def _lower_call(call: UOp, inputs: set[UOp]) -> tuple[UOp, ...]:
    """Replace one proven matrix call with LinAlg arithmetic and its native storage conversion."""
    from tinygrad_dx12.codegen.linalg import DX12_LINALG, _matrix_program, choose_matrix

    if not DX12_LINALG or call.op is not Ops.CALL:
        return (call,)
    spec = match_matmul(call.src[0])
    if spec is None or not isinstance(device := call.device, str) or device.split(":")[0] != "DX12":
        return (call,)
    a_dtype, b_dtype = dtype_of(spec.left), dtype_of(spec.right)
    if any(param.arg.slot + 1 >= len(call.src) for param in (spec.output, spec.left, spec.right)):
        return (call,)
    output, left, right = (call.src[param.arg.slot + 1] for param in (spec.output, spec.left, spec.right))
    staging = tuple(not _aligned(argument) or any(u in inputs for u in argument.toposort()) for argument in (left, right))
    if not spec.epilogue and not _aligned(output):
        return (call,)
    renderer = Device[device].renderer
    if not isinstance(renderer, DX12Renderer):
        return (call,)
    output_dtype = spec.accumulator if spec.epilogue else dtype_of(spec.output)
    if output_dtype not in COMPONENTS:
        return (call,)
    strides = (
        spec.n * output_dtype.itemsize,
        (spec.m if spec.transposed[0] else spec.k) * a_dtype.itemsize,
        (spec.k if spec.transposed[1] else spec.n) * b_dtype.itemsize,
    )
    if any(stride % 16 for stride in strides):
        return (call,)
    scope, tile, threads = choose_matrix(Device[device], renderer, a_dtype, b_dtype, spec.accumulator, spec.m, spec.n, spec.k)
    if scope == "tiled" or any(dimension % width for dimension, width in zip((spec.m, spec.n, spec.k), tile)):
        return (call,)
    tm, tn, _ = tile
    if (limits := renderer.global_max) is not None and (spec.n // tn > limits[0] or spec.m // tm > limits[1]):
        return (call,)
    source = ""
    try:
        source = linalg_source(
            a_dtype,
            b_dtype,
            spec.m,
            spec.n,
            spec.k,
            scope,
            renderer.capabilities.get("wave_min", 32),
            tile,
            threads,
            spec.accumulator,
            output_dtype,
            spec.transposed,
        )
        if source in renderer.matrix_failures:
            return (call,)
        params = (
            UOp.param(0, output_dtype, (spec.m, spec.n), device),
            UOp.param(1, a_dtype, (spec.m, spec.k), device),
            UOp.param(2, b_dtype, (spec.k, spec.n), device),
        )
        program = _matrix_program(*params, renderer, source, (spec.n // tn, spec.m // tm, 1))
        get_runtime(device, program)
    except (RuntimeError, CompileError, ValueError) as error:
        renderer.matrix_failures[source] = str(error)
        return (call,)
    copies, arguments = [], []
    for argument, dtype, size, copy in zip((left, right), (a_dtype, b_dtype), (spec.m * spec.k, spec.k * spec.n), staging):
        if copy:
            staged = UOp.new_buffer(device, size, dtype)
            dest, src = UOp.param(0, dtype, size, device), UOp.param(1, dtype, size, device)
            axis = UOp.range(size, 0)
            sink = dest.index(axis).store(src.index(axis)).end(axis).sink(arg=KernelInfo(name="dx12_matrix_input"))
            copies.append(call.replace(src=(sink, staged, argument)))
            argument = staged
        arguments.append(argument)
    target = UOp.new_buffer(device, spec.m * spec.n, output_dtype) if spec.epilogue else output
    product = call.replace(src=(program, target, *arguments))
    if not spec.epilogue:
        return *copies, product
    # Keep arbitrary post-matrix expressions in tinygrad instead of a second HLSL expression renderer.
    temporary = UOp.param(len(call.src) - 1, output_dtype, spec.m * spec.n, device)
    epilogue = call.src[0].substitute({spec.reduction: temporary.index(spec.output_index)})
    return *copies, product, call.replace(src=(epilogue, *call.src[1:], target))


def lower_matmul(linear: UOp, input_uops: tuple[UOp, ...] = ()) -> UOp:
    """Lower eligible scheduled calls before memory planning and graph extraction.

    Programs already compiled by this pass are preserved. LinAlg converts the
    accumulator directly into the scheduled output buffer's storage dtype.
    Fused epilogues remain ordinary tinygrad calls after a native matrix product;
    their temporary storage participates in the existing memory planner.
    Rebindable JIT inputs receive visible copies because their native alignment
    may change on replay even when their tensor shape and dtype remain identical.
    """
    inputs = set(input_uops)
    return linear.replace(src=tuple(replacement for call in linear.src for replacement in _lower_call(call, inputs)))
