"""Packed low-bit inputs represented by bytes, with ordinary GPU conversion nodes."""

from dataclasses import dataclass
import math
from tinygrad import Tensor, dtypes
from tinygrad.dtype import DType, least_upper_dtype, strong_dtype
from tinygrad.uop.ops import UOp

# (total bits, exponent bits, mantissa bits, exponent bias); integers have no exponent.
FORMATS = {"int4": (4, 0, 0, 0), "uint4": (4, 0, 0, 0), "fp4_e2m1": (4, 2, 1, 1), "fp6_e2m3": (6, 2, 3, 1), "fp6_e3m2": (6, 3, 2, 3)}
FP8_CANDIDATES = {"fp4_e2m1": (dtypes.fp8e4m3, dtypes.fp8e5m2), "fp6_e2m3": (dtypes.fp8e4m3,), "fp6_e3m2": (dtypes.fp8e4m3, dtypes.fp8e5m2)}


@dataclass(frozen=True)
class PackedTensor:
    """A contiguous low-bit tensor stored least-significant payload first in bytes.

    ``shape`` specifies logical elements, while ``data`` is a flat uint8 Tensor.
    Six-bit elements may straddle bytes. FP4 and FP6 use finite OCP payloads.
    Optional numeric ``scales`` have shape ``shape[:-1] + (ceil(K/block_size),)``;
    each scale applies to one block along the final dimension. Scale tensors may
    contain a decoded E8M0 scale or any other numeric scaling convention.
    Conversion remains visible to tinygrad's scheduler and JIT memory planner.
    """

    data: Tensor
    format: str
    shape: tuple[int, ...]
    scales: Tensor | None = None
    block_size: int = 32

    def __post_init__(self) -> None:
        """Reject missing format metadata, insufficient storage, and incompatible scales."""
        if self.format not in FORMATS:
            raise ValueError(f"unknown packed format {self.format!r}")
        if not self.shape or any(x < 0 for x in self.shape):
            raise ValueError("packed shape must contain nonnegative dimensions")
        if self.data.dtype is not dtypes.uint8 or len(self.data.shape) != 1:
            raise ValueError("packed data must be a flat uint8 Tensor")
        if self.data.numel() < (math.prod(self.shape) * FORMATS[self.format][0] + 7) // 8:
            raise ValueError("packed storage is smaller than the declared shape")
        if self.block_size < 1:
            raise ValueError("packed block size must be positive")
        if self.scales is not None:
            expected = self.shape[:-1] + ((self.shape[-1] + self.block_size - 1) // self.block_size,)
            if self.scales.shape != expected or self.scales.device != self.data.device:
                raise ValueError("packed scales have incompatible shape or device")

    def unpack(self, dtype: DType | None = None) -> Tensor:
        """Decode to numeric elements and apply scales at their original K boundaries.

        Floating payloads and scaled integers default to FP32, retaining FP64 scales;
        unscaled integers default to INT32. An explicit dtype uses ordinary tinygrad
        conversion. Unscaled, exactly representable FP8 payloads take a direct
        encoding shortcut; scaled or inexact conversions use the shared cast path.
        """
        integer = self.format in ("int4", "uint4")
        dtype = dtype or (
            dtypes.int32
            if integer and self.scales is None
            else least_upper_dtype(dtypes.float32, self.scales.dtype)
            if self.scales is not None
            else dtypes.float32
        )
        bits, _, mantissa, bias = FORMATS[self.format]
        count = math.prod(self.shape)
        if not count:
            return Tensor.empty(self.shape, device=self.data.device, dtype=dtype)
        # Four elements occupy two bytes for INT4/FP4 or three bytes for FP6.
        # Strided views keep decoding elementwise instead of materializing gathers.
        byte_count, elements = bits // 2, 4
        raw = self.data.pad((0, (-self.data.numel()) % byte_count)).reshape(-1, byte_count)
        word = raw[:, 0].cast(dtypes.uint32)
        for i in range(1, byte_count):
            word = word | (raw[:, i].cast(dtypes.uint32) << (8 * i))
        shifts = Tensor(UOp.arange(elements).cast(dtypes.uint32), device=self.data.device) * bits
        payload = ((word.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).flatten()[:count]
        if self.scales is None and dtype in FP8_CANDIDATES.get(self.format, ()):
            # Every admitted FP4/FP6 value is exact in this FP8 format. Encode its
            # exponent/significand directly, including signed zero and subnormals.
            target_mantissa, target_bias = (3, 7) if dtype is dtypes.fp8e4m3 else (2, 15)
            fraction = payload & ((1 << mantissa) - 1)
            exponent = (payload >> mantissa) & ((1 << (bits - mantissa - 1)) - 1)
            subnormal = fraction.const_like(0)
            for high in range(mantissa):
                encoded = ((target_bias + 1 - bias - mantissa + high) << target_mantissa) | (
                    (fraction << (target_mantissa - high)) & ((1 << target_mantissa) - 1)
                )
                subnormal = (fraction >= 1 << high).where(encoded, subnormal)
            normal = ((exponent + target_bias - bias) << target_mantissa) | (fraction << (target_mantissa - mantissa))
            encoded = ((payload >> (bits - 1)) << 7) | (exponent == 0).where(subnormal, normal)
            return encoded.cast(dtypes.uint8).bitcast(dtype).reshape(self.shape)
        if integer:
            value = ((payload.cast(dtypes.int32) ^ 8) - 8) if self.format == "int4" else payload.cast(dtypes.int32)
        else:
            fraction = (payload & ((1 << mantissa) - 1)).cast(dtypes.float32)
            exponent = ((payload >> mantissa) & ((1 << (bits - mantissa - 1)) - 1)).cast(dtypes.int32)
            significand = (exponent == 0).where(fraction, fraction + (1 << mantissa))
            value = significand * ((exponent == 0).where(1, exponent) - bias - mantissa).cast(dtypes.float32).exp2()
            value = (payload & (1 << (bits - 1)) != 0).where(-value, value)
        value = value.reshape(self.shape)
        if self.scales is not None:
            scale_dtype = least_upper_dtype(dtypes.float32, self.scales.dtype, dtype)
            scales = self.scales.unsqueeze(-1).expand(*self.scales.shape, self.block_size).reshape(*self.shape[:-1], -1)[..., : self.shape[-1]]
            value = value.cast(scale_dtype) * scales.cast(scale_dtype)
        return value.cast(dtype)

    def matmul(self, other: Tensor | PackedTensor) -> Tensor:
        """Select a DX12 matrix candidate with visible decoding; custom kernels have no gradient."""
        from tinygrad_dx12.codegen.linalg import matmul

        return matmul(self, other)


def matrix_intermediate(a: Tensor | PackedTensor, b: Tensor | PackedTensor, native16: bool = False) -> DType:
    """Use shared dtype promotion after selecting an exact packed storage representation.

    Prefer a common FP8 encoding when both operands fit; otherwise promote their
    declared types with tinygrad. ``native16`` is retained for existing callers but
    no longer changes dtype selection: device support belongs to lowering.
    """
    operands = (a, b)
    candidates = set.intersection(*[
        {operand.dtype} if isinstance(operand, Tensor) else set(FP8_CANDIDATES.get(operand.format, ())) if operand.scales is None else set()
        for operand in operands
    ])
    if common := next((dt for dt in (dtypes.fp8e4m3, dtypes.fp8e5m2) if dt in candidates), None):
        return common
    return strong_dtype(
        least_upper_dtype(*[
            operand.dtype
            if isinstance(operand, Tensor)
            else least_upper_dtype(dtypes.float32, operand.scales.dtype)
            if operand.scales is not None
            else {"int4": dtypes.int8, "uint4": dtypes.uint8}.get(operand.format, dtypes.float16)
            for operand in operands
        ])
    )


def matmul(a: Tensor | PackedTensor, b: Tensor | PackedTensor) -> Tensor:
    """Multiply tensors or packed operands through a normal, graph-visible reduction.

    Packed inputs widen to FP32 by default, ensuring existing higher-precision
    activations and block-scaled weights are preserved. The backend's matrix
    candidate API can select another exact common intermediate before this fallback.
    """
    lhs, rhs = [operand.unpack() if isinstance(operand, PackedTensor) else operand for operand in (a, b)]
    return lhs.matmul(rhs)
