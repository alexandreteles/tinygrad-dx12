"""Conservative scheduled-matmul recognition and buffer alignment guards."""

import os
import pytest
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Buffer
from tinygrad.dtype import DType
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from tinygrad_dx12.codegen.matmul import _aligned, match_matmul


def matrix_sink(m: int = 16, n: int = 48, k: int = 32, dtype: DType = dtypes.float16, transposed: bool = False, activated: bool = False) -> UOp:
    """Build a mathematical matrix reduction with independent parameter indexing."""
    output = UOp.param(0, dtype, m * n, "DX12")
    left, right = UOp.param(1, dtype, m * k, "DX12"), UOp.param(2, dtype, k * n, "DX12")
    row, column, inner = UOp.range(m, 1), UOp.range(n, 2), UOp.range(k, 0, AxisType.REDUCE)
    a = left.index(inner * m + row if transposed else row * k + inner)
    product = (a * right.index(inner * n + column)).cast(dtypes.float32).reduce(inner, arg=Ops.ADD).cast(dtype)
    if activated:
        product = product.maximum(0)
    return output.index(row * n + column).store(product).end(row, column).sink(arg=KernelInfo(name="test"))


def test_recognizes_exact_row_major_matrix() -> None:
    """The accepted operation exposes all three independent matrix dimensions and buffer slots."""
    spec = match_matmul(matrix_sink())
    assert spec is not None
    assert (spec.m, spec.n, spec.k) == (16, 48, 32)
    assert tuple(p.arg.slot for p in (spec.output, spec.left, spec.right)) == (0, 1, 2)


@pytest.mark.parametrize("kwargs", [{"transposed": True}, {"activated": True}, {"dtype": dtypes.float32}])
def test_recognizes_native_matrix_semantics(kwargs: dict) -> None:
    """Matching preserves transpose, epilogue, and dtype for the exact DX12 operation query."""
    spec = match_matmul(matrix_sink(**kwargs))
    assert spec is not None
    assert spec.transposed == (kwargs.get("transposed", False), False)
    assert spec.epilogue == kwargs.get("activated", False)
    assert spec.left.dtype is kwargs.get("dtype", dtypes.float16)


def test_rejects_broadcast_and_second_output() -> None:
    """A shared reduction index or extra side effect must retain its complete original kernel."""
    ast = matrix_sink()
    row = ast.src[0].src[1]
    assert match_matmul(ast.substitute({row: row.const_like(0)})) is None
    assert match_matmul(ast.replace(src=ast.src * 2)) is None


def test_rejects_explicit_codegen_options() -> None:
    """A user's supplied optimizer decisions must not be replaced by an automatic matrix candidate."""
    assert match_matmul(matrix_sink().replace(arg=KernelInfo(opts_to_apply=()))) is None


def test_jit_base_and_view_alignment() -> None:
    """Base JIT inputs are aligned, while a one-half offset remains visible through the view chain."""
    parameter = UOp.param(0, dtypes.float16, 512, "DX12")
    assert _aligned(parameter)
    assert not _aligned(parameter[1:257])
    assert _aligned(parameter[64:320])


def test_hidden_buffer_view_alignment() -> None:
    """Native Buffer offset metadata must be checked even when the UOp itself is a BUFFER."""
    base = Buffer("DX12", 512, dtypes.float16)
    assert not _aligned(UOp.from_buffer(base.view(256, dtypes.float16, 2)))
    assert _aligned(UOp.from_buffer(base.view(256, dtypes.float16, 128)))


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires DX12 adapters and an exclusive validation window")
@pytest.mark.parametrize("device", ["DX12", "DX12:1"])
def test_ordinary_matmul_eager_gradient_and_jit(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary Tensor matmul preserves storage dtype, autograd, and changed replay inputs."""
    import numpy as np
    import tinygrad_dx12
    from tinygrad.engine.realize import run_linear
    from tinygrad_dx12.codegen.linalg import _support

    tinygrad_dx12.register()
    if device == "DX12:1" and Device["DX12"].count() < 2:
        pytest.skip("requires two DX12 adapters")
    renderer = Device[device].renderer
    support = _support(Device[device], renderer, dtypes.float16, dtypes.float16, dtypes.float32, "Wave", (16, 16, 16))
    if not support or (16, 16, 16) not in support.get("shapes", ()):
        pytest.skip("requires a queried FP16 Wave matrix shape")
    # Fix a supported candidate so this tests lowering and lifetimes independently of timing noise.
    monkeypatch.setitem(renderer.matrix_choices, (dtypes.float16, dtypes.float16, dtypes.float16, 16, 48, 32, None), ("Wave", (16, 16, 16), 32))
    data_a = np.arange(16 * 32, dtype=np.float32).reshape(16, 32) % 5 - 2
    data_b = np.arange(32 * 48, dtype=np.float32).reshape(32, 48) % 3 - 1
    a = Tensor(data_a, device=device, dtype=dtypes.float16).realize()
    b = Tensor(data_b, device=device, dtype=dtypes.float16).realize()
    eager = a @ b
    linear = eager.schedule_linear()
    assert len(linear.src) == 1
    matrix = linear.src[0]
    assert matrix.src[0].op is Ops.PROGRAM and matrix.src[0].arg.name == "dx12_linalg"
    assert matrix.src[1].dtype is dtypes.float16
    assert any(u.op is Ops.SOURCE and "acc.Cast<ComponentType::F16>()" in u.arg for u in matrix.toposort())
    run_linear(linear)
    np.testing.assert_allclose(eager.numpy(), data_a @ data_b, rtol=0.003, atol=0.003)

    def product(a: Tensor, b: Tensor) -> Tensor:
        """Capture the ordinary matrix expression through the backend's scheduled rewrite."""
        return (a @ b).realize()

    jit = TinyJit(product)
    for step in range(4):
        a = Tensor(data_a + step, device=device, dtype=dtypes.float16)
        b = Tensor(data_b, device=device, dtype=dtypes.float16)
        result = jit(a, b)
        assert result.dtype is dtypes.float16
        np.testing.assert_allclose(result.numpy(), (data_a + step) @ data_b, rtol=0.003, atol=0.003)
        if step == 0:
            grad_a, grad_b = (a @ b).sum().gradient(a, b)
            np.testing.assert_allclose(grad_a.numpy(), np.ones((16, 48)) @ data_b.T, rtol=0.003, atol=0.003)
            np.testing.assert_allclose(grad_b.numpy(), (data_a + step).T @ np.ones((16, 48)), rtol=0.003, atol=0.003)


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires DX12 adapters and an exclusive validation window")
@pytest.mark.parametrize("hidden_offset", [False, True])
def test_misaligned_ordinary_matmul_jit(hidden_offset: bool) -> None:
    """Changing data in a two-byte-offset view survives staging and JIT input parameterization."""
    import numpy as np
    import tinygrad_dx12

    tinygrad_dx12.register()

    def product(a: Tensor, b: Tensor) -> Tensor:
        """Capture ordinary matmul with the backing buffer's nonzero byte offset."""
        if not hidden_offset:
            a = a[1:].reshape(16, 32)
        return (a @ b).realize()

    jit = TinyJit(product)
    data_a = np.arange(16 * 32 + 1, dtype=np.float32) % 5 - 2
    data_b = np.arange(32 * 48, dtype=np.float32).reshape(32, 48) % 3 - 1
    for step in range(4):
        base = Tensor(data_a + step, device="DX12", dtype=dtypes.float16).realize()
        a = Tensor(UOp.from_buffer(base.uop.buffer.view(16 * 32, dtypes.float16, 2).ensure_allocated())).reshape(16, 32) if hidden_offset else base
        b = Tensor(data_b, device="DX12", dtype=dtypes.float16).realize()
        np.testing.assert_allclose(
            jit(a, b).numpy(), (data_a[1:] + step).reshape(16, 32) @ data_b, rtol=0.003, atol=0.003, err_msg=f"replay step {step}"
        )


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires a DX12 adapter")
def test_ordinary_matmul_jit_stages_rebound_alignment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aligned capture may replay with hidden byte offsets while native matrix operands stay aligned."""
    import numpy as np
    import tinygrad_dx12
    from tinygrad_dx12.codegen.linalg import _support

    tinygrad_dx12.register()
    renderer = Device["DX12"].renderer
    support = _support(Device["DX12"], renderer, dtypes.float16, dtypes.float16, dtypes.float32, "Wave", (16, 16, 16))
    if not support or (16, 16, 16) not in support.get("shapes", ()):
        pytest.skip("requires a queried FP16 Wave matrix shape")
    key = (dtypes.float16, dtypes.float16, dtypes.float16, 16, 48, 32, None)
    for suffix in ((), ("staging", (True, True))):
        monkeypatch.setitem(renderer.matrix_choices, key + suffix, ("Wave", (16, 16, 16), 32))

    def product(a: Tensor, b: Tensor) -> Tensor:
        """Capture native ordinary matmul with replaceable input buffers."""
        return (a @ b).realize()

    jit = TinyJit(product)
    for step, (left_offset, right_offset) in enumerate(((0, 0), (0, 0), (2, 6), (128, 2), (6, 128))):
        values = [((np.arange(size + 64, dtype=np.float32) + step) % 7 - 3).astype(np.float16) for size in (16 * 32, 32 * 48)]
        arguments = []
        for data, size, offset, shape in zip(values, (16 * 32, 32 * 48), (left_offset, right_offset), ((16, 32), (32, 48))):
            base = Tensor(data, device="DX12", dtype=dtypes.float16).realize()
            view = base.uop.buffer.view(size, dtypes.float16, offset).ensure_allocated()
            arguments.append(Tensor(UOp.from_buffer(view)).reshape(shape))
        expected = values[0][left_offset // 2 : left_offset // 2 + 16 * 32].reshape(16, 32) @ values[1][
            right_offset // 2 : right_offset // 2 + 32 * 48
        ].reshape(32, 48)
        np.testing.assert_allclose(jit(*arguments).numpy(), expected, rtol=0.003, atol=0.003, err_msg=f"replay step {step}")
    assert jit.captured is not None
    programs = [u for u in jit.captured.linear.toposort() if u.op is Ops.PROGRAM]
    assert sum(u.arg.name == "dx12_matrix_input" for u in programs) == 2
    assert any(u.arg.name == "dx12_linalg" for u in programs)
