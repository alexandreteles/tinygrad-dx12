"""Device-query-selected LinAlg kernels checked against independent matrix products."""

import os
import pytest
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad_dx12.codegen.linalg import matmul, matvec

pytestmark = pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires the DX12 preview toolchain and adapter")


@pytest.mark.parametrize("dtype", [dtypes.float32, dtypes.float16, dtypes.fp8e4m3, dtypes.fp8e5m2, dtypes.int8, dtypes.uint8])
def test_matrix_tails(dtype: object) -> None:
    """Exercise non-tile dimensions with exact reference arithmetic across storage formats."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    a = [[(i + k) % 5 for k in range(19)] for i in range(17)]
    b = [[(k + j) % 3 for j in range(21)] for k in range(19)]
    expected = [[sum(a[i][k] * b[k][j] for k in range(19)) for j in range(21)] for i in range(17)]
    assert matmul(Tensor(a, device="DX12", dtype=dtype), Tensor(b, device="DX12", dtype=dtype)).tolist() == expected


@pytest.mark.parametrize("scope", ["Wave", "ThreadGroup"])
def test_matrix_scope_replay(scope: str) -> None:
    """Accepted explicit scopes must replay changed input data and preserve output tails."""
    import tinygrad_dx12

    tinygrad_dx12.register()

    def product(a: Tensor, b: Tensor) -> Tensor:
        """Capture the complete padded matrix pipeline."""
        return matmul(a, b, scope=scope).contiguous().realize()

    jit = TinyJit(product)
    for step in range(4):
        a = [[(i + k + step) % 5 - 2 for k in range(19)] for i in range(17)]
        b = [[(k - j) % 3 - 1 for j in range(21)] for k in range(19)]
        expected = [[sum(a[i][k] * b[k][j] for k in range(19)) for j in range(21)] for i in range(17)]
        try:
            actual = jit(Tensor(a, device="DX12", dtype=dtypes.float16), Tensor(b, device="DX12", dtype=dtypes.float16)).tolist()
        except RuntimeError as error:
            if "candidate rejected" in str(error):
                pytest.skip(str(error))
            raise
        assert actual == expected


@pytest.mark.parametrize("scope", ["Wave", "ThreadGroup", "Thread"])
def test_matrix_replay_stages_rebound_inputs(scope: str) -> None:
    """A captured matrix must remain aligned when both input buffers acquire hidden byte offsets."""
    import tinygrad_dx12
    from tinygrad.uop.ops import Ops, UOp

    tinygrad_dx12.register()

    def argument(values: list[int], shape: tuple[int, ...], offset: int) -> Tensor:
        """Wrap a concrete native view so JIT's shape metadata stays equal across offsets."""
        backing = Tensor([99] * (offset // 2) + values, device="DX12", dtype=dtypes.float16).realize()
        return Tensor(UOp.from_buffer(backing.uop.buffer.view(len(values), dtypes.float16, offset).ensure_allocated())).reshape(shape)

    def product(a: Tensor, b: Tensor) -> Tensor:
        """Capture the explicit native matrix plus a consumer for graph replay."""
        result = matvec(a, b, dtypes.float16, thread=True) if scope == "Thread" else matmul(a, b, scope=scope)
        return result.realize().relu().realize()

    jit, captured = TinyJit(product), None
    for step, offset in enumerate((0, 0, 2, 4, 126, 0)):
        values = [[(i + j + step) % 5 - 2 for j in range(16)] for i in range(16)]
        right = [j + step for j in range(16)] if scope == "Thread" else [int(i == j) for i in range(16) for j in range(16)]
        a = argument([x for row in values for x in row], (16, 16), offset)
        b = argument(right, (16,) if scope == "Thread" else (16, 16), offset)
        try:
            actual = jit(a, b).tolist()
        except RuntimeError as error:
            if "candidate rejected" in str(error):
                pytest.skip(str(error))
            raise
        expected = (
            [max(sum(x * y for x, y in zip(row, right)), 0) for row in values] if scope == "Thread" else [[max(x, 0) for x in row] for row in values]
        )
        assert actual == expected
        if step == 1:
            captured = jit.captured
            assert captured is not None
            if not any(u.op is Ops.SOURCE and f"MatrixScope::{scope}" in u.arg for u in captured._linear.toposort()):
                pytest.skip("the device rejected the exact native matrix candidate")
        if step > 1:
            assert jit.captured is captured


def test_misaligned_matrix_view() -> None:
    """A one-half-element view offset must be staged before aligned matrix loads."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    a = Tensor([99] + [i % 5 for i in range(256)], device="DX12", dtype=dtypes.float16)[1:].reshape(16, 16)
    b = Tensor([[int(i == j) for j in range(16)] for i in range(16)], device="DX12", dtype=dtypes.float16)
    assert matmul(a, b).tolist() == [[(16 * i + j) % 5 for j in range(16)] for i in range(16)]


@pytest.mark.parametrize("shape", [(16, 16), (17, 19)])
def test_thread_matrix_vector(shape: tuple[int, int]) -> None:
    """Execute an accepted native thread matrix, including padded reduction tails."""
    import tinygrad_dx12
    from tinygrad.uop.ops import Ops

    tinygrad_dx12.register()
    n, k = shape
    values = [[(row + col) % 5 - 2 for col in range(k)] for row in range(n)]
    vector = [col % 3 - 1 for col in range(k)]
    result = matvec(
        Tensor(values, device="DX12", dtype=dtypes.float16), Tensor(vector, device="DX12", dtype=dtypes.float16), dtypes.float16, thread=True
    )
    if not any(u.op is Ops.SOURCE and "MatrixScope::Thread" in u.arg for u in result.uop.toposort()):
        pytest.skip("the device rejected the exact thread-matrix candidate")
    assert result.tolist() == [sum(a * b for a, b in zip(row, vector)) for row in values]


def test_thread_matrix_preserves_requested_precision() -> None:
    """An FP16-only thread operation must not narrow a requested FP32 accumulation."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    matrix = Tensor([[2048.0, 1.0, -2048.0]], device="DX12", dtype=dtypes.float16)
    vector = Tensor([1.0, 1.0, 1.0], device="DX12", dtype=dtypes.float16)
    result = matvec(matrix, vector)
    assert result.dtype is dtypes.float32 and result.tolist() == [1.0]


@pytest.mark.parametrize("device", ["PYTHON", "DX12"])
@pytest.mark.parametrize("dtype,output_dtype,values", [(dtypes.int8, dtypes.int32, [100, 120]), (dtypes.float16, dtypes.float32, [300, 400])])
def test_matvec_uses_device_arithmetic(device: str, dtype: object, output_dtype: object, values: list[int]) -> None:
    """Native arithmetic belongs to LinAlg; the fallback retains tinygrad's narrow-product behavior."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    matrix, vector = Tensor([values], device=device, dtype=dtype), Tensor(values, device=device, dtype=dtype)
    expected = matrix.dot(vector, dtype=output_dtype).tolist() if device == "PYTHON" else [sum(value * value for value in values)]
    result = matvec(matrix, vector, output_dtype)
    assert result.dtype is output_dtype and result.tolist() == expected


def test_measured_matvec_replay() -> None:
    """Scratch tuning must stay out of a captured graph that receives changing vectors."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    identity = Tensor([[int(i == j) for j in range(16)] for i in range(16)], device="DX12", dtype=dtypes.float16).realize()

    def product(vector: Tensor) -> Tensor:
        """Capture the measured choice and any visible staging work."""
        return matvec(identity, vector, dtypes.float16).contiguous().realize()

    jit = TinyJit(product)
    for step in range(4):
        values = [i + step for i in range(16)]
        assert jit(Tensor(values, device="DX12", dtype=dtypes.float16)).tolist() == values


def test_explicit_tiled_uses_tinygrad_schedule() -> None:
    """Explicit tiled construction delegates to tinygrad's deferred matrix graph."""
    import tinygrad_dx12
    from tinygrad.uop.ops import Ops

    tinygrad_dx12.register()
    values = [[(i + j) % 3 for j in range(16)] for i in range(16)]
    identity = [[int(i == j) for j in range(16)] for i in range(16)]
    result = matmul(Tensor(values, device="DX12", dtype=dtypes.float16), Tensor(identity, device="DX12", dtype=dtypes.float16), scope="tiled")
    assert all(u.op is not Ops.SOURCE for u in result.uop.toposort())
    assert result.tolist() == values


@pytest.mark.parametrize(
    "dtype,values,expected",
    [(dtypes.float64, [1099511627776.25, 1.0, -1099511627776.0], 1.25), (dtypes.int64, [1099511627776, 7, -1099511627776], 7)],
)
def test_wide_matrix_fallback(dtype: object, values: list[int | float], expected: int | float) -> None:
    """A missing native matrix candidate must retain existing FP64 or integer64 precision."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    result = matmul(Tensor([values], device="DX12", dtype=dtype), Tensor([[1], [1], [1]], device="DX12", dtype=dtype))
    assert result.dtype is dtype and result.tolist() == [[expected]]


def test_backing_buffer_view_alignment() -> None:
    """Explicit matrix and vector staging must include offsets hidden in Buffer.view metadata."""
    import tinygrad_dx12
    from tinygrad.uop.ops import UOp

    tinygrad_dx12.register()
    values = [[(i + j) % 5 - 2 for j in range(16)] for i in range(16)]
    base = Tensor([99] + [x for row in values for x in row], device="DX12", dtype=dtypes.float16).realize()
    matrix = Tensor(UOp.from_buffer(base.uop.buffer.view(256, dtypes.float16, 2).ensure_allocated())).reshape(16, 16)
    vector_base = Tensor([99] + list(range(16)), device="DX12", dtype=dtypes.float16).realize()
    vector = Tensor(UOp.from_buffer(vector_base.uop.buffer.view(16, dtypes.float16, 2).ensure_allocated()))
    identity = Tensor([[int(i == j) for j in range(16)] for i in range(16)], device="DX12", dtype=dtypes.float16)
    assert matmul(matrix, identity).tolist() == values
    assert matvec(matrix, vector, dtypes.float16, thread=True).tolist() == [sum(x * j for j, x in enumerate(row)) for row in values]


@pytest.mark.parametrize(
    "fmt,dtype,values,expected",
    [
        ("int4", dtypes.int32, [16777217, 16777219], -2),
        ("int4", dtypes.int64, [1099511627777, 1099511627779], -2),
        ("uint4", dtypes.uint64, [1099511627777, 1099511627779], 4398046511114),
    ],
)
def test_packed_integer_matmul_precision(fmt: str, dtype: object, values: list[int], expected: int) -> None:
    """Packed integers must not route large integer activations through floating point."""
    import tinygrad_dx12
    from tinygrad_dx12.codegen.decomp.dx12 import PackedTensor

    tinygrad_dx12.register()
    weights = PackedTensor(Tensor([0xF1 if fmt == "int4" else 0x31], device="DX12", dtype=dtypes.uint8), fmt, (1, 2))
    output = weights.matmul(Tensor([[x] for x in values], device="DX12", dtype=dtype))
    assert output.dtype is dtype and output.tolist() == [[expected]]


@pytest.mark.parametrize("scope", ["auto", "tiled"])
@pytest.mark.parametrize("dtype,values", [(dtypes.int8, [100, 120]), (dtypes.uint8, [200, 250])])
def test_matrix_product_uses_accumulator_width(scope: str, dtype: object, values: list[int]) -> None:
    """Wider matrix accumulation must not overflow a narrow product before summation."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    result = matmul(Tensor([values], device="DX12", dtype=dtype), Tensor([[x] for x in values], device="DX12", dtype=dtype), scope=scope)
    assert result.dtype is (dtypes.uint32 if dtype is dtypes.uint8 else dtypes.int32)
    assert result.tolist() == [[sum(x * x for x in values)]]
