"""Inspect real DXIL feature requirements after late dtype lowering."""

import os, struct
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops
from tinygrad_dx12.renderer.cstyle import DX12Renderer
from tinygrad_dx12.runtime.support.compiler_dx12 import DX12Compiler, unpack_shader

pytestmark = pytest.mark.skipif(
    os.name != "nt" or os.getenv("DX12_TEST_GPU") != "1", reason="set DX12_TEST_GPU=1 to use the installed Windows toolchain"
)


def feature_flags(binary: bytes) -> int:
    """Read the DXIL container's SFI0 required-feature mask, independent of HLSL spelling."""
    assert binary[:4] == b"DXBC"
    offsets = struct.unpack_from(f"<{struct.unpack_from('<I', binary, 28)[0]}I", binary, 32)
    return next(struct.unpack_from("<Q", binary, offset + 8)[0] for offset in offsets if binary[offset : offset + 4] == b"SFI0")


def test_integer64_feature_positive_control() -> None:
    """A real wide integer multiply must set the published INT64_OPS requirement bit."""
    binary = DX12Compiler().compile_dxil(
        "RWByteAddressBuffer data:register(u0); [numthreads(1,1,1)] void main() { "
        "uint64_t x=uint64_t(data.Load(0))|(uint64_t(data.Load(4))<<32); data.Store(8,uint((x*x)>>32)); }"
    )
    # D3D_SHADER_REQUIRES_INT64_OPS in the pinned d3d12shader.h.
    assert feature_flags(binary) & 0x8000


@pytest.mark.parametrize("operation", ["copy", "arithmetic", "trunc", "exp2", "log2", "sqrt", "sin", "sin32", "sin_bool"])
def test_double_and_transcendentals_do_not_require_integer64(operation: str) -> None:
    """Independent double support cannot silently add an unavailable integer64 device requirement."""
    renderer = DX12Renderer(Target("DX12", arch='{"doubles":true,"int64":false,"shader_model":104}'))
    value = Tensor.empty(32, device="NULL", dtype=dtypes.float32 if operation == "sin32" else dtypes.float64)
    match operation:
        case "copy":
            result = value.clone()
        case "arithmetic":
            result = value * 1.25 + 0.25
        case "sin32":
            result = value.sin()
        case "sin_bool":
            result = value.sin() > 0
        case _:
            result = getattr(value, operation)()
    programs = [to_program(call.src[0], renderer) for call in result.schedule_linear().src if call.src[0].op is Ops.SINK]
    assert programs
    for program in programs:
        _, binary = unpack_shader(program.src[3].arg)
        flags = feature_flags(binary)
        assert flags & 0x8000 == 0
        if operation != "sin32":
            assert flags & 1


def execute_with_renderer(result: Tensor, renderer: DX12Renderer) -> Tensor:
    """Compile explicit program bodies for a capability subset without changing the live device renderer."""
    linear = result.schedule_linear()
    calls = []
    for call in linear.src:
        if call.src[0].op is Ops.SINK:
            program = to_program(call.src[0], renderer)
            _, binary = unpack_shader(program.src[3].arg)
            assert feature_flags(binary) & (0x8000 | 0x40000) == 0
            call = call.replace(src=(program, *call.src[1:]))
        calls.append(call)
    run_linear(linear.replace(src=tuple(calls)))
    return result


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires a DX12 adapter")
@pytest.mark.parametrize(
    "operation", ["arithmetic", "trunc", "exp2", "log2", "sqrt", "sin", "sin32", "half_reduce", "long_arithmetic", "ulong_arithmetic"]
)
def test_independent_capabilities_numerical(operation: str) -> None:
    """The decomposed program preserves arithmetic against independent NumPy values on real hardware."""
    import numpy as np

    renderer = DX12Renderer(Target("DX12", arch='{"doubles":true,"int64":false,"native16":false,"shader_model":104}'))
    dtype = {"half_reduce": dtypes.float16, "long_arithmetic": dtypes.int64, "ulong_arithmetic": dtypes.uint64, "sin32": dtypes.float32}.get(
        operation, dtypes.float64
    )
    match operation:
        case "long_arithmetic":
            data = np.asarray([-(2**60), -(2**40) - 7, -1, 0, 1, 2**40 + 13, 2**60], dtype=np.int64)
        case "ulong_arithmetic":
            data = np.asarray([0, 1, 2**32 - 1, 2**40 + 13, 2**63, 2**64 - 1], dtype=np.uint64)
        case "half_reduce":
            data = (np.arange(16 * 33, dtype=np.float32).reshape(16, 33) % 17 - 8).astype(np.float16)
        case "trunc":
            data = np.asarray([-1e12 - 0.75, -1.75, -0.5, 0, 0.5, 1.75, 1e12 + 0.75])
        case "sin" | "sin32":
            data = np.asarray(
                [-1e6, -39800.25, -100, -1, -0.0, 0, 0.125, 0.75, 30.5, 39800.25, 1e6], dtype=np.float32 if operation == "sin32" else np.float64
            )
        case _:
            data = np.asarray([0.0001, 0.125, 0.75, 1, 1.25, 5.5, 20.25])
    value = Tensor(data, device="DX12", dtype=dtype)
    match operation:
        case "long_arithmetic" | "ulong_arithmetic":
            result, expected = value * 3 + 7, data * 3 + 7
        case "half_reduce":
            result, expected = (value + 1).sum(1), (data + 1).sum(1)
        case "arithmetic":
            result, expected = value * 1.25 + 0.25, data * 1.25 + 0.25
        case "sin32":
            result, expected = value.sin(), np.sin(data)
        case _:
            result, expected = getattr(value, operation)(), getattr(np, operation)(data)
    actual = execute_with_renderer(result, renderer).numpy()
    if operation in ("long_arithmetic", "ulong_arithmetic", "half_reduce"):
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-6 if operation == "sin32" else 1e-10, atol=2e-7 if operation == "sin32" else 1e-10)
