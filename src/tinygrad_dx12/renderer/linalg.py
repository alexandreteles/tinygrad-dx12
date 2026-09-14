"""Shape-specific HLSL LinAlg candidates using only the packaged header API."""

from tinygrad.dtype import DType, dtypes, least_upper_dtype, strong_dtype, sum_acc_dtype
from tinygrad.uop.ops import UOp
from tinygrad_dx12.renderer.cstyle import WINDOW, buffer_access, make_layout
from tinygrad_dx12.runtime.autogen import d3d12 as d

# HLSL components and D3D12 query enums describe the same storage, but use different values.
MATRIX_TYPES = {
    dtypes.float32: ("F32", d.D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT32),
    dtypes.float16: ("F16", d.D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT16),
    dtypes.fp8e4m3: ("F8_E4M3FN", d.D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E4M3FN),
    dtypes.fp8e5m2: ("F8_E5M2", d.D3D12_LINEAR_ALGEBRA_DATATYPE_FLOAT8_E5M2),
    dtypes.int8: ("I8", d.D3D12_LINEAR_ALGEBRA_DATATYPE_SINT8),
    dtypes.uint8: ("U8", d.D3D12_LINEAR_ALGEBRA_DATATYPE_UINT8),
    dtypes.int16: ("I16", d.D3D12_LINEAR_ALGEBRA_DATATYPE_SINT16),
    dtypes.uint16: ("U16", d.D3D12_LINEAR_ALGEBRA_DATATYPE_UINT16),
    dtypes.int32: ("I32", d.D3D12_LINEAR_ALGEBRA_DATATYPE_SINT32),
    dtypes.uint32: ("U32", d.D3D12_LINEAR_ALGEBRA_DATATYPE_UINT32),
}
COMPONENTS = {dtype: value[0] for dtype, value in MATRIX_TYPES.items()}
DATATYPES = {dtype: value[1] for dtype, value in MATRIX_TYPES.items()}
SCALARS = {
    dtypes.float16: "float16_t",
    dtypes.float32: "float",
    dtypes.int16: "int16_t",
    dtypes.uint16: "uint16_t",
    dtypes.int32: "int",
    dtypes.uint32: "uint",
}


def _accumulator(a: DType, b: DType) -> DType:
    """Reuse tinygrad's default reduction dtype instead of a DX12 precision policy."""
    return strong_dtype(sum_acc_dtype(strong_dtype(least_upper_dtype(a, b))))


def linalg_source(
    dtype_a: DType,
    dtype_b: DType,
    m: int = 16,
    n: int = 16,
    k: int = 16,
    scope: str = "ThreadGroup",
    wave_size: int = 32,
    tile: tuple[int, int, int] = (16, 16, 16),
    group_threads: int = 32,
    accumulator: DType | None = None,
    output_dtype: DType | None = None,
    transposed: tuple[bool, bool] = (False, False),
) -> str:
    """Render a collective matrix candidate with byte strides and disjoint output tiles.

    Dimensions must already be padded to the tile and 16-byte row strides by
    graph-visible operations. The caller checks this exact source with DXC and
    the selected device's PSO creation. Wave candidates request the queried lane
    count and put one complete wave in each group; group threads follow the query.
    BF16 is intentionally absent because the pinned header declares no component.
    ``accumulator`` declares arithmetic precision; ``output_dtype`` optionally
    requests LinAlg's matrix cast before storing, without a temporary buffer or
    conversion dispatch. The caller checks the complete program with DXC/PSO.
    """
    if dtype_a not in COMPONENTS or dtype_b not in COMPONENTS:
        raise ValueError("matrix component has no packaged LinAlg component/query mapping")
    if scope not in ("Wave", "ThreadGroup"):
        raise ValueError("collective matrix scope must be Wave or ThreadGroup")
    tm, tn, tk = tile
    if any(d < 1 or d % t for d, t in zip((m, n, k), tile)):
        raise ValueError("LinAlg dimensions require complete tiles")
    stride_a, stride_b = (m if transposed[0] else k) * dtype_a.itemsize, (k if transposed[1] else n) * dtype_b.itemsize
    if stride_a % 16 or stride_b % 16:
        raise ValueError("LinAlg raw-buffer strides require 16-byte alignment")
    acc_dtype = accumulator or _accumulator(dtype_a, dtype_b)
    out_dtype = output_dtype or acc_dtype
    if acc_dtype not in COMPONENTS or out_dtype not in COMPONENTS:
        raise ValueError("matrix accumulator/output has no packaged LinAlg component/query mapping")
    component = COMPONENTS[acc_dtype]
    result = "acc" if out_dtype is acc_dtype else f"acc.Cast<ComponentType::{COMPONENTS[out_dtype]}>()"
    threads = wave_size if scope == "Wave" else group_threads
    address_a = f"kk*{m}u+gid.y*{tm}u" if transposed[0] else f"gid.y*{tm * k}u+kk"
    address_b = f"gid.x*{tn * k}u+kk" if transposed[1] else f"kk*{n}u+gid.x*{tn}u"
    layout_a, layout_b = ("ColMajor" if transpose else "RowMajor" for transpose in transposed)
    params = [UOp.placeholder((size,), dt, i) for i, (size, dt) in enumerate(((m * n, out_dtype), (m * k, dtype_a), (k * n, dtype_b)))]
    layout = make_layout([(name, (p, i == 0)) for i, (name, p) in enumerate(zip(("C", "A", "B"), params))], (threads, 1, 1))
    addresses = []
    for binding, index, dt, span in zip(
        layout.buffers,
        (f"gid.y*{tm * n}u+gid.x*{tn}u", address_a, address_b),
        (out_dtype, dtype_a, dtype_b),
        (
            (tm - 1) * n * out_dtype.itemsize + tn * out_dtype.itemsize,
            ((tk - 1) * stride_a + tm * dtype_a.itemsize) if transposed[0] else ((tm - 1) * stride_a + tk * dtype_a.itemsize),
            ((tn - 1) * stride_b + tk * dtype_b.itemsize) if transposed[1] else ((tk - 1) * stride_b + tn * dtype_b.itemsize),
        ),
    ):
        if binding.views > 1:
            if span > WINDOW:
                raise ValueError("LinAlg tile exceeds the raw descriptor overlap; use tinygrad lowering")
            index = index.replace("gid.y", "uint64_t(gid.y)").replace("gid.x", "uint64_t(gid.x)").replace("kk", "uint64_t(kk)")
        addresses.append(",".join(buffer_access(binding.name, f"_offset_{binding.slot}+({index})*{dt.itemsize}u", binding.views > 1, uniform=True)))
    address_c, address_a, address_b = addresses
    return "\n".join([
        "// DX12:" + layout.to_json(),
        f"#define DX12_LOCAL_X {threads}",
        "#define DX12_LOCAL_Y 1",
        "#define DX12_LOCAL_Z 1",
        "#include <dx/linalg.h>",
        "using namespace dx::linalg;",
        *layout.declarations(),
        f"using MA = Matrix<ComponentType::{COMPONENTS[dtype_a]},{tm},{tk},MatrixUse::A,MatrixScope::{scope}>;",
        f"using MB = Matrix<ComponentType::{COMPONENTS[dtype_b]},{tk},{tn},MatrixUse::B,MatrixScope::{scope}>;",
        f"using MC = Matrix<ComponentType::{component},{tm},{tn},MatrixUse::Accumulator,MatrixScope::{scope}>;",
        *([f"[WaveSize({wave_size})]"] if scope == "Wave" else []),
        "[numthreads(DX12_LOCAL_X,DX12_LOCAL_Y,DX12_LOCAL_Z)]",
        "void main(uint3 gid : SV_GroupID) {",
        "  MC acc = MC::Splat(0);",
        f"  for (uint kk=0; kk<{k}u; kk+={tk}u) {{",
        f"    MA a = MA::Load({address_a},{stride_a}u,MatrixLayout::{layout_a},16);",
        f"    MB b = MB::Load({address_b},{stride_b}u,MatrixLayout::{layout_b},16);",
        "    acc.MultiplyAccumulate(a,b);",
        "  }",
        f"  {result}.Store({address_c},{n * out_dtype.itemsize}u,MatrixLayout::RowMajor,16);",
        "}",
    ])


def thread_source(matrix_dtype: DType, vector_dtype: DType, output_dtype: DType, n: int, k: int) -> str:
    """Render a thread-scope matrix-vector operation with explicit scalar precision.

    Matrix rows and the vector are already padded by visible Tensor operations.
    The selected device must accept the exact vector/matrix/bias/result dtype
    query. A requested FP32 result is never narrowed to make an FP16 candidate fit.
    Raw offsets and strides are bytes; the header's row-major Load overload is used.
    FP8 vectors keep their packed bytes through MakeInterpretedVector; no scalar
    FP8 type or host-side conversion is required.
    """
    scalar = SCALARS
    packed = vector_dtype.itemsize == 1
    if matrix_dtype not in COMPONENTS or vector_dtype not in COMPONENTS or output_dtype not in scalar:
        raise ValueError("unsupported native thread-matrix scalar component")
    if n < 1 or k < 1 or k * matrix_dtype.itemsize % 16:
        raise ValueError("thread-matrix dimensions require a 16-byte matrix stride")
    params = [UOp.placeholder((size,), dt, i) for i, (size, dt) in enumerate(((n, output_dtype), (n * k, matrix_dtype), (k, vector_dtype)))]
    layout = make_layout([(name, (p, i == 0)) for i, (name, p) in enumerate(zip(("C", "A", "B"), params))], (1, 1, 1))
    if any(binding.views > 1 for binding in layout.buffers):
        raise ValueError("thread matrix exceeds one raw descriptor; use collective or tinygrad lowering")
    load = "B.Load(_offset_2+i*4u)" if packed else f"B.Load<{scalar[vector_dtype]}>(_offset_2+i*{vector_dtype.itemsize}u)"
    length = k // 4 if packed else k
    vector = f"MakeInterpretedVector<ComponentType::{COMPONENTS[vector_dtype]}>(x)" if packed else "x"
    return "\n".join([
        "// DX12:" + layout.to_json(),
        "#define DX12_LOCAL_X 1",
        "#define DX12_LOCAL_Y 1",
        "#define DX12_LOCAL_Z 1",
        "#include <dx/linalg.h>",
        "using namespace dx::linalg;",
        *layout.declarations(),
        f"using MA = Matrix<ComponentType::{COMPONENTS[matrix_dtype]},{n},{k},MatrixUse::A,MatrixScope::Thread>;",
        "[numthreads(DX12_LOCAL_X,DX12_LOCAL_Y,DX12_LOCAL_Z)]",
        "void main() {",
        f"  MA a = MA::Load<MatrixLayout::RowMajor>(A,_offset_1,{k * matrix_dtype.itemsize}u,128);",
        f"  vector<{'uint8_t4_packed' if packed else scalar[vector_dtype]},{length}> x;",
        f"  for (uint i=0;i<{length}u;i++) x[i]={load};",
        f"  vector<{scalar[output_dtype]},{n}> y = MultiplyAdd<{scalar[output_dtype]}>(a,{vector},(vector<{scalar[output_dtype]},{n}>)0);",
        f"  for (uint i=0;i<{n}u;i++) C.Store<{scalar[output_dtype]}>(_offset_0+i*{output_dtype.itemsize}u,y[i]);",
        "}",
    ])
