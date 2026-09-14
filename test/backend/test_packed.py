"""OCP payload decoding checked against tabulated values, including cross-byte FP6."""

import math, os
import pytest
from tinygrad import Tensor, dtypes
from tinygrad_dx12.codegen.decomp.dx12 import PackedTensor, matrix_intermediate, matmul

GOLDEN = {
    "fp4_e2m1": (0, 0.5, 1, 1.5, 2, 3, 4, 6),
    "fp6_e2m3": (
        0,
        0.125,
        0.25,
        0.375,
        0.5,
        0.625,
        0.75,
        0.875,
        1,
        1.125,
        1.25,
        1.375,
        1.5,
        1.625,
        1.75,
        1.875,
        2,
        2.25,
        2.5,
        2.75,
        3,
        3.25,
        3.5,
        3.75,
        4,
        4.5,
        5,
        5.5,
        6,
        6.5,
        7,
        7.5,
    ),
    "fp6_e3m2": (
        0,
        0.0625,
        0.125,
        0.1875,
        0.25,
        0.3125,
        0.375,
        0.4375,
        0.5,
        0.625,
        0.75,
        0.875,
        1,
        1.25,
        1.5,
        1.75,
        2,
        2.5,
        3,
        3.5,
        4,
        5,
        6,
        7,
        8,
        10,
        12,
        14,
        16,
        20,
        24,
        28,
    ),
}


def payload_bytes(codes: list[int], bits: int) -> list[int]:
    """Encode test payloads least-significant first independently of GPU decoding."""
    number = sum(code << (i * bits) for i, code in enumerate(codes))
    return list(number.to_bytes((len(codes) * bits + 7) // 8, "little"))


@pytest.mark.parametrize("fmt", list(GOLDEN))
@pytest.mark.parametrize("device", ["PYTHON"] + (["DX12"] if os.getenv("DX12_TEST_GPU") == "1" else []))
def test_complete_ocp_payloads(fmt: str, device: str) -> None:
    """Exhaustively check every finite payload value and both signed-zero encodings."""
    if device == "DX12":
        import tinygrad_dx12

        tinygrad_dx12.register()
    positive = GOLDEN[fmt]
    bits = 4 if fmt.startswith("fp4") else 6
    expected = [*positive, *[-x for x in positive]]
    packed = PackedTensor(Tensor(payload_bytes(list(range(1 << bits)), bits), device=device, dtype=dtypes.uint8), fmt, (1 << bits,))
    actual = packed.unpack().tolist()
    assert actual == expected
    assert math.copysign(1, actual[len(positive)]) == -1


@pytest.mark.parametrize("fmt,expected", [("int4", list(range(8)) + list(range(-8, 0))), ("uint4", list(range(16)))])
def test_integer_nibbles(fmt: str, expected: list[int]) -> None:
    """Validate signed extension and zero extension independently for the same stored bytes."""
    packed = PackedTensor(Tensor(payload_bytes(list(range(16)), 4), device="PYTHON", dtype=dtypes.uint8), fmt, (16,))
    assert packed.unpack().tolist() == expected


def test_scales_stay_with_k_blocks() -> None:
    """Different K-block scales must affect separate summands inside a matrix reduction."""
    data = Tensor([0x22, 0x22], device="PYTHON", dtype=dtypes.uint8)
    packed = PackedTensor(data, "fp4_e2m1", (1, 4), Tensor([[2, 7]], device="PYTHON"), block_size=2)
    assert matmul(packed, Tensor([[1], [2], [3], [4]], device="PYTHON")).tolist() == [[55.0]]


def test_matrix_intermediate_preserves_activations() -> None:
    """FP32 activations cannot be quantized merely because packed weights fit FP8."""
    weights = PackedTensor(Tensor([0], device="PYTHON", dtype=dtypes.uint8), "fp6_e2m3", (1, 1))
    assert matrix_intermediate(Tensor([[1.125]], device="PYTHON", dtype=dtypes.float32), weights, True) is dtypes.float32
    assert matrix_intermediate(weights, weights, True) is dtypes.fp8e4m3
    converted = weights.unpack(dtypes.fp8e5m2)
    assert converted.dtype is dtypes.fp8e5m2 and converted.tolist() == [[0.0]]


@pytest.mark.parametrize("device", ["PYTHON"] + (["DX12"] if os.getenv("DX12_TEST_GPU") == "1" else []))
def test_double_block_scales(device: str) -> None:
    """Scaling exact packed values must retain fractional FP64 scale bits."""
    if device == "DX12":
        import tinygrad_dx12

        tinygrad_dx12.register()
    packed = PackedTensor(
        Tensor([0x22], device=device, dtype=dtypes.uint8),
        "fp4_e2m1",
        (1, 2),
        Tensor([[1099511627776.25, -1099511627776.0]], device=device, dtype=dtypes.float64),
        block_size=1,
    )
    decoded = packed.unpack()
    assert decoded.dtype is dtypes.float64 and decoded.tolist() == [[1099511627776.25, -1099511627776.0]]
    assert matrix_intermediate(packed, packed) is dtypes.float64
    assert matmul(packed, Tensor([[1.0], [1.0]], device=device)).tolist() == [[0.25]]


@pytest.mark.parametrize(
    "fmt,boundary_kind,scaled",
    [(fmt, kind, False) for fmt in ("uint4", "fp6_e2m3", "fp6_e3m2") for kind in ("bit", "element")] + [("uint4", "element", True)],
)
@pytest.mark.parametrize("device", ["PYTHON"] + (["DX12"] if os.getenv("DX12_TEST_GPU") == "1" else []))
def test_large_packed_bit_positions(fmt: str, boundary_kind: str, scaled: bool, device: str) -> None:
    """Decode across signed 32-bit bit and element offsets using a 17-byte backing pattern."""
    from tinygrad.engine.realize import run_linear

    bits = 4 if fmt == "uint4" else 6
    boundary = 2**31 // bits if boundary_kind == "bit" else 2**31
    count = boundary + 32
    byte_count = (count * bits + 7) // 8
    pattern = [(i * 37 + 13) % 256 for i in range(17)]
    storage = Tensor(pattern, device=device, dtype=dtypes.uint8).repeat((byte_count + 16) // 17)[:byte_count]
    scale_count = (count + 31) // 32
    scales = Tensor([2, 7, 11], device=device, dtype=dtypes.float32).repeat((scale_count + 2) // 3)[:scale_count] if scaled else None
    result = PackedTensor(storage, fmt, (count,), scales).unpack()[boundary - 3 : boundary + 7].contiguous()
    linear = result.schedule_linear()
    assert all(arg.numel() <= 17 for call in linear.src for arg in call.src[1:])
    run_linear(linear)
    expected = []
    for index in range(boundary - 3, boundary + 7):
        byte, shift = divmod(index * bits, 8)
        payload = ((pattern[byte % 17] | pattern[(byte + 1) % 17] << 8) >> shift) & ((1 << bits) - 1)
        value = payload if fmt == "uint4" else GOLDEN[fmt][payload & 31] * (-1 if payload & 32 else 1)
        expected.append(value * ([2, 7, 11][index // 32 % 3] if scaled else 1))
    assert result.tolist() == expected
