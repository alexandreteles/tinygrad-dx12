"""Graph-visible custom matrix programs preserving storage until LinAlg lowering."""

from tinygrad import Tensor, Device
from tinygrad.codegen import to_program
from tinygrad.device import CompileError
from tinygrad.engine.realize import capturing, get_runtime
from tinygrad.uop.ops import UOp, Ops, ProgramInfo, KernelInfo
from tinygrad.renderer import Estimates
from tinygrad.dtype import DType
from tinygrad.helpers import CAPTURING, ContextVar
from tinygrad_dx12.codegen.decomp.dx12 import PackedTensor, matrix_intermediate
from tinygrad_dx12.renderer.cstyle import DX12Renderer, parse_source, dtype_of
from tinygrad_dx12.renderer.linalg import DATATYPES, SCALARS, _accumulator, linalg_source, thread_source

DX12_LINALG = ContextVar("DX12_LINALG", 1)


def _support(device: object, renderer: DX12Renderer, a: DType, b: DType, accumulator: DType, scope: str, tile: tuple[int, int, int]) -> dict | None:
    """Cache the optional packaged operation query separately for each device renderer."""
    from tinygrad_dx12.runtime.support.dx12 import query_matrix_support

    if any(dt not in DATATYPES for dt in (a, b, accumulator)):
        return {"flags": 0}
    wave = renderer.capabilities.get("wave_min", 32)
    key = (a, b, accumulator, scope, tile, wave)
    if key not in renderer.matrix_support:
        try:
            ta, tb, tc = (DATATYPES[dt] for dt in (a, b, accumulator))
            renderer.matrix_support[key] = query_matrix_support(
                getattr(device, "native"), ta, tb, tc, wave, (tile[0], tile[2], tile[1]), scope == "ThreadGroup"
            )
        except OSError, RuntimeError:
            renderer.matrix_support[key] = None
    return renderer.matrix_support[key]


def _matrix_program(out: UOp, a: UOp, b: UOp, renderer: DX12Renderer, source: str, groups: tuple[int, int, int], name: str = "dx12_linalg") -> UOp:
    """Create a complete tinygrad PROGRAM with explicit resources and source metadata."""
    params = tuple(x.buf_uop for x in (out, a, b))
    info = ProgramInfo(
        name=name,
        global_size=groups,
        local_size=parse_source(source).local_size,
        globals=(0, 1, 2),
        outs=(0,),
        ins=(1, 2),
        target=renderer.target,
    )
    m, n, k = a.shape[0], b.shape[1] if len(b.shape) == 2 else 1, a.shape[1]
    sink = UOp.sink(
        *params,
        arg=KernelInfo(
            name=name,
            estimates=Estimates(2 * m * n * k, m * n * dtype_of(out).itemsize + m * k * dtype_of(a).itemsize + k * n * dtype_of(b).itemsize),
        ),
    )
    program = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.LINEAR, src=params), UOp(Ops.SOURCE, arg=source)), arg=info)
    return to_program(program, renderer)


def _unaligned(value: Tensor) -> bool:
    """Detect a known matrix-start offset requiring visible alignment staging."""
    if (view := value.uop.contiguous_view()) is None:
        return False
    base = view[0].src[0] if view[0].op is Ops.MSELECT else view[0]
    if base.op is Ops.BUFFER:
        from tinygrad_dx12.codegen.matmul import _aligned

        return not _aligned(value.uop)
    return bool(view[1] * dtype_of(view[0]).itemsize % 128)


def _rebindable(value: Tensor) -> bool:
    """Identify captured input storage whose hidden native offset may change on replay."""
    if not (CAPTURING and capturing) or (view := value.uop.contiguous_view()) is None:
        return False
    base = view[0].src[0] if view[0].op is Ops.MSELECT else view[0]
    return base.op is Ops.PARAM or base.realized is not None


def _stage(value: Tensor, padding: tuple[tuple[int, int], tuple[int, int]]) -> Tensor:
    """Pad tails or copy possibly rebound matrix inputs as ordinary graph-visible work."""
    if any(before or after for before, after in padding):
        return value.pad(padding).contiguous()
    if _unaligned(value) or _rebindable(value):
        return value.clone()
    return value.contiguous()


def choose_matrix(
    device: object,
    renderer: DX12Renderer,
    a_dtype: DType,
    b_dtype: DType,
    accumulator: DType,
    m: int,
    n: int,
    k: int,
    tile: tuple[int, int, int] | None = None,
    scope: str = "auto",
) -> tuple[str, tuple[int, int, int], int]:
    """Select a supported LinAlg operation using only DX12's shape/type queries.

    Prefer a queried wave shape that divides the operation, then the first queried
    shape for callers that can pad. Try threadgroup scope when wave scope does not
    accept the requested tile. Group sizes come from the API. ``tiled`` requests
    ordinary compute when neither scope is reported as supported; there is no
    startup benchmarking, scratch execution, or adapter-specific ranking.
    """
    key = (a_dtype, b_dtype, accumulator, m, n, k, tile, scope)
    if key in renderer.matrix_choices:
        return renderer.matrix_choices[key]
    chosen_tile = tile or (16, 16, 16)
    wave_info = _support(device, renderer, a_dtype, b_dtype, accumulator, "Wave", chosen_tile)
    if tile is None and wave_info and wave_info.get("shapes"):
        shapes = wave_info["shapes"]
        sm, sk, sn = next((shape for shape in shapes if all(size % width == 0 for size, width in zip((m, k, n), shape))), shapes[0])
        chosen_tile = (sm, sn, sk)
    for candidate in ("Wave", "ThreadGroup") if scope == "auto" else (scope,):
        support = wave_info if candidate == "Wave" else _support(device, renderer, a_dtype, b_dtype, accumulator, candidate, chosen_tile)
        if not support or not support["flags"] & 1:
            continue
        if candidate == "Wave" and (chosen_tile[0], chosen_tile[2], chosen_tile[1]) not in support.get("shapes", ()):
            continue
        threads = (
            renderer.capabilities.get("wave_min", 32)
            if candidate == "Wave"
            else (support.get("preferred_threads") or support.get("min_threads") or 32)
        )
        renderer.matrix_choices[key] = (candidate, chosen_tile, threads)
        return renderer.matrix_choices[key]
    renderer.matrix_choices[key] = ("tiled", chosen_tile, 0)
    return renderer.matrix_choices[key]


def matmul(
    a: Tensor | PackedTensor, b: Tensor | PackedTensor, scope: str = "auto", tile: tuple[int, int, int] | None = None, dtype: DType | None = None
) -> Tensor:
    """Multiply two matrices, testing LinAlg acceptance on their owning DX12 device.

    ``scope`` chooses ``auto``, ``Wave``, ``ThreadGroup``, or ``tiled``. Auto selects
    a queried native operation, falling back to ordinary compute when unavailable.
    Explicit scopes raise the recorded compiler/PSO error when unsupported. The
    default tile follows the exact operation query, including its K dimension and
    supported group thread counts. A supplied tile is checked against that query. Input
    decode, tail padding, and output slicing are normal graph-visible Tensor
    operations, so a Program never hides an extra dispatch. FP8 payloads remain
    FP8 storage until the custom matrix shader. Scaled or higher-precision inputs
    use a common wider intermediate without quantizing activations. Captured
    inputs are staged so later native buffer views cannot invalidate alignment.

    ``dtype`` declares accumulator and output precision, otherwise using tinygrad's
    default reduction dtype. FP16 is passed to DX12 when requested;
    internal arithmetic and rounding belong to LinAlg. Acceptance is cached per
    exact shader and device renderer; no numerical or performance policy is inferred.
    This custom-kernel API does not define gradients; use ordinary Tensor matrix
    operations for training, whose scheduled DX12 rewrite preserves autograd.
    """
    if scope not in ("auto", "Wave", "ThreadGroup", "tiled"):
        raise ValueError(f"unknown matrix scope {scope!r}")
    automatic = scope == "auto"
    device = a.data.device if isinstance(a, PackedTensor) else a.device
    other_device = b.data.device if isinstance(b, PackedTensor) else b.device
    if device != other_device:
        raise ValueError("matrix operands must belong to one device")
    if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matrix operands must have compatible rank-two shapes")
    is_dx12 = isinstance(device, str) and device.split(":")[0] == "DX12"
    if not is_dx12 or 0 in (*a.shape, *b.shape):
        intermediate = matrix_intermediate(a, b)
        intermediate = _accumulator(intermediate, intermediate)
        lhs, rhs = [x.unpack(intermediate).contiguous() if isinstance(x, PackedTensor) else x for x in (a, b)]
        out_dtype = dtype or _accumulator(lhs.dtype, rhs.dtype)
        return lhs.matmul(rhs, dtype=out_dtype)
    renderer = Device[device].renderer
    if not isinstance(renderer, DX12Renderer):
        raise RuntimeError("DX12 matrix lowering requires its device-local renderer")
    intermediate = matrix_intermediate(a, b)
    lhs, rhs = [x.unpack(intermediate) if isinstance(x, PackedTensor) else x for x in (a, b)]
    if isinstance(a, PackedTensor) or isinstance(b, PackedTensor):
        lhs, rhs = lhs.cast(intermediate), rhs.cast(intermediate)
    m, k, n = lhs.shape[0], lhs.shape[1], rhs.shape[1]
    out_dtype = dtype or _accumulator(lhs.dtype, rhs.dtype)
    if not isinstance(m, int) or not isinstance(k, int) or not isinstance(n, int):
        return lhs.matmul(rhs, dtype=out_dtype)
    from math import lcm

    if scope == "tiled":
        return lhs.cast(out_dtype).matmul(rhs.cast(out_dtype), dtype=out_dtype)
    if tile is not None and any(x < 1 for x in tile):
        raise ValueError("matrix tile dimensions must be positive")
    chosen_scope, chosen_tile, group_threads = choose_matrix(Device[device], renderer, lhs.dtype, rhs.dtype, out_dtype, m, n, k, tile, scope)
    if chosen_scope == "tiled":
        if automatic:
            return lhs.matmul(rhs, dtype=out_dtype)
        raise RuntimeError(f"DX12 {scope} {chosen_tile} rejected by the device operation query")
    scope = chosen_scope
    tm, tn, tk = chosen_tile
    # Staging matrices start at their own allocation; row strides are multiples of 16 bytes.
    kp, np = lcm(tk, 16 // lhs.dtype.itemsize), lcm(tn, 16 // rhs.dtype.itemsize)
    mp, np, kp = (m + tm - 1) // tm * tm, (n + np - 1) // np * np, (k + kp - 1) // kp * kp
    padded_a, padded_b = _stage(lhs, ((0, mp - m), (0, kp - k))), _stage(rhs, ((0, kp - k), (0, np - n)))
    out = Tensor.empty((mp, np), device=device, dtype=out_dtype)
    source = ""

    def kernel(output: UOp, left: UOp, right: UOp) -> UOp:
        """Compile and check the exact custom matrix candidate before graph extraction."""
        program = _matrix_program(output, left, right, renderer, source, (np // tn, mp // tm, 1))
        get_runtime(device, program)
        return program

    try:
        source = linalg_source(
            lhs.dtype, rhs.dtype, mp, np, kp, scope, renderer.capabilities.get("wave_min", 32), chosen_tile, group_threads, out_dtype
        )
        if source in renderer.matrix_failures:
            raise RuntimeError(renderer.matrix_failures[source])
        return out.custom_kernel(padded_a, padded_b, fxn=kernel)[0][:m, :n]
    except (RuntimeError, CompileError, OSError, ValueError) as error:
        renderer.matrix_failures[source] = str(error)
        if automatic:
            return lhs.matmul(rhs, dtype=out_dtype)
        raise RuntimeError(f"DX12 {scope} candidate rejected: {error}") from error


def matvec(matrix: Tensor | PackedTensor, vector: Tensor, output_dtype: DType | None = None, thread: bool | None = None) -> Tensor:
    """Multiply a rank-two matrix by a vector through the shared matrix path.

    Output precision follows tinygrad's shared reduction default unless explicitly
    supplied. A requested dtype is passed unchanged to the device query. Passing
    ``output_dtype=dtypes.float16`` requests an accepted FP16 thread operation.
    Packed matrix decode and any tail padding stay in the normal
    computation graph. By default this is a one-column matmul, sharing native
    selection, padding and staging. ``thread=True`` requests the separate
    thread-scope API explicitly for callers that need its execution model;
    ``thread=False`` selects the ordinary implementation. Custom native operations
    do not define gradients; use ordinary Tensor operations for training.
    """
    device = matrix.data.device if isinstance(matrix, PackedTensor) else matrix.device
    if len(matrix.shape) != 2 or len(vector.shape) != 1 or matrix.shape[1] != vector.shape[0] or device != vector.device:
        raise ValueError("matvec expects a matrix and matching vector on one device")
    intermediate = matrix_intermediate(matrix, vector)
    output_dtype = output_dtype or _accumulator(intermediate, intermediate)
    if thread is None:
        return matmul(matrix, vector.reshape(-1, 1), dtype=output_dtype).reshape(matrix.shape[0])
    raw = matrix.unpack(intermediate) if isinstance(matrix, PackedTensor) else matrix
    fallback = raw.dot(vector, dtype=output_dtype)
    n, k = raw.shape
    if (
        thread is False
        or not isinstance(device, str)
        or not device.startswith("DX12")
        or not isinstance(n, int)
        or not isinstance(k, int)
        or not n
        or not k
    ):
        return fallback
    renderer = Device[device].renderer
    if not isinstance(renderer, DX12Renderer):
        raise RuntimeError("thread matrices require a DX12 renderer")
    if output_dtype not in SCALARS:
        return fallback
    if any(dt not in DATATYPES for dt in (raw.dtype, vector.dtype, output_dtype)):
        return fallback
    from tinygrad_dx12.runtime.support.dx12 import query_thread_matrix_support

    tv, tm, to = (DATATYPES[dt] for dt in (vector.dtype, raw.dtype, output_dtype))
    key = ("Thread", tv, tm, to)
    if key not in renderer.matrix_support:
        try:
            renderer.matrix_support[key] = {"flags": query_thread_matrix_support(getattr(Device[device], "native"), tv, tm, to, to)}
        except RuntimeError, OSError:
            renderer.matrix_support[key] = None
    if (support := renderer.matrix_support[key]) is not None and not support["flags"] & 1:
        return fallback
    kp = (k + (16 // raw.dtype.itemsize) - 1) // (16 // raw.dtype.itemsize) * (16 // raw.dtype.itemsize)
    matrix_arg = _stage(raw, ((0, 0), (0, kp - k)))
    vector_arg = vector.pad((0, kp - k)).contiguous()
    if _unaligned(vector_arg) or _rebindable(vector_arg):
        vector_arg = vector_arg.clone()
    source = ""
    output = Tensor.empty(n, device=device, dtype=output_dtype)

    def kernel(out: UOp, a: UOp, b: UOp) -> UOp:
        """Compile a complete one-dispatch thread-matrix program with explicit arguments."""
        program = _matrix_program(out, a, b, renderer, source, (1, 1, 1), name="dx12_matvec")
        get_runtime(device, program)
        return program

    try:
        source = thread_source(raw.dtype, vector.dtype, output_dtype, n, kp)
        if source in renderer.matrix_failures:
            return fallback
        return output.custom_kernel(matrix_arg, vector_arg, fxn=kernel)[0]
    except (RuntimeError, CompileError, OSError, ValueError) as error:
        renderer.matrix_failures[source] = str(error)
        return fallback
