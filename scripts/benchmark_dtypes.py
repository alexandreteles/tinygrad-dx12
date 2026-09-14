"""Compare every supported numeric storage type across DX12, OpenCL, CUDA, and WebGPU.

This benchmark is deliberately separate from ``benchmark_dx12.py``. It reuses
that script's capture, timing, and adapter-identification machinery without
changing the established FP32/FP16 benchmark or its reports. Every storage format
runs vector, row-reduction, and matrix workloads in eager and JIT modes; arithmetic
precision and matrix APIs differ between native and packed rows.

FP4, both FP6 formats, and INT4 are byte-packed formats rather than tinygrad
``DType`` objects. Their rows therefore include graph-visible unpacking (and,
for matmul, the backend's packed matrix path) in the measured operation.
Packed floating vector/reduction rows decode to FP32. Packed floating matmul uses
wide accumulation/output, while ordinary FP8 matmul returns FP8. Multiplication and
accumulation follow the selected backend kernel; FP8 storage does not imply FP8 ALU
rounding. Those rows do not isolate unpacking cost or native dtype throughput.
"""

import argparse, ctypes, json, math, os, platform, time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tinygrad import Device, Tensor, dtypes
from tinygrad.dtype import DType
from tinygrad_dx12 import register
from tinygrad_dx12.codegen.decomp.dx12 import PackedTensor

from benchmark_dx12 import capture, device_info, measure


@dataclass(frozen=True)
class DTypeSpec:
    """Describe one native or packed dtype included in the comparison.

    ``key`` is the CLI and JSON identifier and ``label`` is the report label.
    Native rows set ``dtype``. Packed rows instead set ``packed_format``,
    ``bits``, and exact payload codes for the benchmark's {-1, 0, 1} inputs.
    """

    key: str
    label: str
    dtype: DType | None = None
    packed_format: str | None = None
    bits: int | None = None
    codes: dict[int, int] | None = None

    @property
    def packed(self) -> bool:
        """Return whether this spec uses packed-byte storage."""
        return self.packed_format is not None


@dataclass(frozen=True)
class OperationCase:
    """Hold deterministic inputs and an independently constructed result."""

    name: str
    arrays: tuple[np.ndarray, ...]
    expected: np.ndarray


DTYPE_SPECS = (
    DTypeSpec("fp64", "FP64", dtype=dtypes.float64),
    DTypeSpec("fp32", "FP32", dtype=dtypes.float32),
    DTypeSpec("fp16", "FP16", dtype=dtypes.float16),
    DTypeSpec("bf16", "BF16", dtype=dtypes.bfloat16),
    DTypeSpec("fp8e4m3", "FP8 E4M3", dtype=dtypes.fp8e4m3),
    DTypeSpec("fp8e5m2", "FP8 E5M2", dtype=dtypes.fp8e5m2),
    DTypeSpec("fp6e2m3", "FP6 E2M3", packed_format="fp6_e2m3", bits=6, codes={-1: 40, 0: 0, 1: 8}),
    DTypeSpec("fp6e3m2", "FP6 E3M2", packed_format="fp6_e3m2", bits=6, codes={-1: 44, 0: 0, 1: 12}),
    DTypeSpec("fp4e2m1", "FP4 E2M1", packed_format="fp4_e2m1", bits=4, codes={-1: 10, 0: 0, 1: 2}),
    DTypeSpec("int8", "INT8", dtype=dtypes.int8),
    DTypeSpec("int4", "INT4", packed_format="int4", bits=4, codes={-1: 15, 0: 0, 1: 1}),
)
OPERATIONS = ("vector", "reduction", "matmul")
MODES = ("eager", "jit")


def vector(x: Tensor) -> Tensor:
    """Run fused integer-safe multiply/add followed by a materialized square.

    Inputs are restricted to -1 and 0, so the intermediate and result are exactly
    representable in every dtype under test without making the multiply an identity.
    """
    return (x * 2 + 1).realize().square().realize()


def reduction(x: Tensor) -> Tensor:
    """Run a materialized square followed by independent row reductions."""
    return x.square().realize().sum(axis=1).realize()


def matrix(a: Tensor, b: Tensor) -> Tensor:
    """Run matrix multiplication followed by a separately materialized ReLU."""
    return (a @ b).realize().relu().realize()


def make_cases(vector_size: int, reduction_size: int, matrix_size: int) -> tuple[OperationCase, ...]:
    """Build data-independent-cost workloads with exactly representable results.

    The vector cycles through {-1, 0}. Each reduction row contains one signed
    unit, keeping its sum equal to one. The left matrix contains one unit per row,
    so the expected product is selected directly from the dense right matrix.
    This avoids using a second tensor implementation as the result oracle.
    """
    vector_input = (np.arange(vector_size, dtype=np.int64) % 2 - 1).astype(np.int8)
    vector_expected = np.square(vector_input.astype(np.int64) * 2 + 1)

    reduction_input = np.zeros((reduction_size, reduction_size), dtype=np.int8)
    reduction_rows = np.arange(reduction_size)
    reduction_input[reduction_rows, reduction_rows] = np.where(reduction_rows % 2, 1, -1)
    reduction_expected = np.ones(reduction_size, dtype=np.int64)

    matrix_rows = np.arange(matrix_size)
    selected_rows = (matrix_rows * 5 + 1) % matrix_size
    matrix_left = np.zeros((matrix_size, matrix_size), dtype=np.int8)
    matrix_left[matrix_rows, selected_rows] = 1
    row, column = np.indices((matrix_size, matrix_size), dtype=np.int64)
    matrix_right = ((row + 2 * column) % 3 - 1).astype(np.int8)
    matrix_expected = np.maximum(matrix_right[selected_rows].astype(np.int64), 0)
    return (
        OperationCase("vector", (vector_input,), vector_expected),
        OperationCase("reduction", (reduction_input,), reduction_expected),
        OperationCase("matmul", (matrix_left, matrix_right), matrix_expected),
    )


def pack_payload(values: np.ndarray, spec: DTypeSpec) -> np.ndarray:
    """Pack {-1, 0, 1} values least-significant bit first for one low-bit spec.

    Args:
      values: Logical values to encode. Any value outside {-1, 0, 1} is rejected.
      spec: A packed dtype specification containing its exact payload mapping.

    Returns:
      A flat uint8 array whose padding bits, when present, are zero.

    Raises:
      ValueError: If ``spec`` is native or ``values`` contains an unmapped value.
    """
    if spec.bits is None or spec.codes is None:
        raise ValueError(f"{spec.key} is not a packed dtype")
    flat = values.reshape(-1)
    unknown = set(int(value) for value in np.unique(flat)) - spec.codes.keys()
    if unknown:
        raise ValueError(f"{spec.key} cannot encode benchmark values {sorted(unknown)}")
    payloads = np.zeros(flat.size, dtype=np.uint8)
    for value, code in spec.codes.items():
        payloads[flat == value] = code
    shifts = np.arange(spec.bits, dtype=np.uint8)
    return np.packbits(((payloads[:, None] >> shifts) & 1).reshape(-1), bitorder="little")


def operation_for(spec: DTypeSpec, case: OperationCase) -> Callable[..., Tensor]:
    """Select ordinary Tensor operations or the packed API's wide-arithmetic path."""
    if not spec.packed:
        return {"vector": vector, "reduction": reduction, "matmul": matrix}[case.name]
    if spec.packed_format is None:
        raise ValueError(f"{spec.key} is missing its packed format")
    shapes = tuple(array.shape for array in case.arrays)

    if case.name == "vector":

        def packed_vector(data: Tensor) -> Tensor:
            """Decode one packed vector inside the captured operation."""
            return vector(PackedTensor(data, spec.packed_format, shapes[0]).unpack())

        return packed_vector
    if case.name == "reduction":

        def packed_reduction(data: Tensor) -> Tensor:
            """Decode one packed matrix inside the captured row reduction."""
            return reduction(PackedTensor(data, spec.packed_format, shapes[0]).unpack())

        return packed_reduction
    if case.name == "matmul":

        def packed_matmul(left: Tensor, right: Tensor) -> Tensor:
            """Run the backend-aware packed matrix path and a separate ReLU."""
            lhs = PackedTensor(left, spec.packed_format, shapes[0])
            rhs = PackedTensor(right, spec.packed_format, shapes[1])
            return lhs.matmul(rhs).realize().relu().realize()

        return packed_matmul
    raise ValueError(f"unknown operation {case.name!r}")


def inputs_for(spec: DTypeSpec, case: OperationCase, device: str) -> tuple[Tensor, ...]:
    """Create and realize native or packed buffers before capture and timing."""
    if spec.packed:
        return tuple(Tensor(pack_payload(array, spec), device=device, dtype=dtypes.uint8).realize() for array in case.arrays)
    if spec.dtype is None:
        raise ValueError(f"{spec.key} is missing its native dtype")
    return tuple(Tensor(array, device=device, dtype=spec.dtype).realize() for array in case.arrays)


def comparison_device_info(device: str) -> dict[str, object]:
    """Return shared device details plus Dawn's adapter identity for WebGPU."""
    info = device_info(device)
    if not device.startswith("WEBGPU"):
        return info
    from tinygrad.runtime.autogen import webgpu
    from tinygrad.runtime.ops_webgpu import from_wgpu_str

    adapter = webgpu.WGPUAdapterInfo()
    status = webgpu.wgpuDeviceGetAdapterInfo(Device[device].device_res, ctypes.byref(adapter))
    if status != webgpu.WGPUStatus_Success:
        return info
    try:
        info.update(
            name=from_wgpu_str(adapter.device),
            driver=from_wgpu_str(adapter.description),
            backend=webgpu.enum_WGPUBackendType.get(adapter.backendType, str(adapter.backendType)).removeprefix("WGPUBackendType_"),
        )
    finally:
        webgpu.wgpuAdapterInfoFreeMembers(adapter)
    return info


def add_relative_speeds(rows: list[dict[str, object]]) -> None:
    """Add speed normalized to the fastest successful runtime for every row group.

    Relative speed is the fastest latency divided by the row latency. Consequently,
    the fastest runtime is 1.00x and 0.50x means half its speed. Failed runtimes do
    not set the baseline or receive a relative value.
    """
    groups: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    for row in rows:
        if "error" not in row:
            groups.setdefault((row["dtype"], row["operation"], row["mode"]), []).append(row)
    for group in groups.values():
        for metric, relative in (("gpu_us", "gpu_relative"), ("host_us", "host_relative")):
            positive = [value for row in group if (value := float(row[metric])) > 0]
            fastest = min(positive) if positive else None
            for row in group:
                value = float(row[metric])
                row[relative] = fastest / value if fastest is not None and value > 0 else None


def metric_cell(row: dict[str, object] | None, metric: str) -> str:
    """Format one report cell as latency plus relative speed, failure, or unavailable."""
    if row is None:
        return "—"
    if "error" in row:
        return "Failed"
    relative = row[{"gpu_us": "gpu_relative", "host_us": "host_relative"}[metric]]
    return f"{float(row[metric]):.2f} ({float(relative):.2f}×)" if relative is not None else f"{float(row[metric]):.2f} (n/a)"


def aggregate_speeds(rows: list[dict[str, object]], devices: list[str], mode: str, relative: str) -> tuple[dict[str, float], int]:
    """Return geometric-mean relative speeds over rows valid for every runtime.

    Args:
      rows: Measured benchmark rows containing per-row relative speeds.
      devices: Runtime names that must all have a valid result for inclusion.
      mode: Either ``eager`` or ``jit``.
      relative: ``gpu_relative`` or ``host_relative``.

    Returns:
      A mapping from runtime to geometric-mean speed and the shared row count.
      If there are no common rows, the mapping is empty and the count is zero.
    """
    groups: dict[tuple[object, object], dict[str, float]] = {}
    for row in rows:
        if row["mode"] != mode or "error" in row or row.get(relative) is None:
            continue
        groups.setdefault((row["dtype"], row["operation"]), {})[str(row["device"])] = float(row[relative])
    common = [group for group in groups.values() if all(device in group for device in devices)]
    if not common:
        return {}, 0
    return ({device: math.exp(sum(math.log(group[device]) for group in common) / len(common)) for device in devices}, len(common))


def report(results: dict[str, object]) -> str:
    """Render side-by-side device metrics and explicit unsupported-case details."""
    devices = list(results["devices"])
    rows = results["rows"]
    index = {(row["dtype"], row["operation"], row["mode"], row["device"]): row for row in rows}
    lines = [
        "# Cross-backend dtype performance measurements",
        "",
        f"Measured {results['timestamp']} on {results['platform']}.",
        "",
        "Each storage-format/operation row is compared across the selected backends. Compiled kernels may differ between backends.",
        "Failed rows are retained rather than replaced with a different dtype or operation.",
        "",
        "**Input storage is not arithmetic precision.** Ordinary FP8 matmul returns FP8; internal arithmetic follows the selected backend kernel.",
        "Packed FP4/FP6 matmul uses wide accumulation/output; on supported DX12 devices it decodes to FP8 and invokes native LinAlg.",
        "Ordinary FP8 can also use native DX12 LinAlg. Its output cast means these rows do not isolate unpacking overhead.",
        "Packed floating vector/reduction rows decode to FP32. Compare backends within each row; cross-format rankings need matched arithmetic.",
        "",
        "| Device | Adapter | Renderer |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {device} | {info.get('name', 'unavailable')} | {info.get('renderer', info.get('error', 'unavailable'))} |"
        for device, info in results["devices"].items()
    ]
    dx_luid = results["devices"].get("DX12", {}).get("luid")
    cuda_luid = results["devices"].get("CUDA", {}).get("luid")
    if dx_luid is not None and dx_luid == cuda_luid:
        lines += ["", "DX12 and CUDA report the same physical adapter LUID."]
    cl_luid = results["devices"].get("CL", {}).get("luid")
    if dx_luid is not None and cl_luid == dx_luid:
        lines += ["OpenCL also reports the same adapter LUID through `cl_khr_device_uuid`."]
    webgpu = results["devices"].get("WEBGPU", {})
    if webgpu.get("name"):
        lines += ([""] if dx_luid is None else []) + [f"WebGPU uses Dawn's {webgpu.get('backend', 'default')} backend on {webgpu['name']}."]
    lines += [
        "",
        "Each timing is `latency (relative speed)`. The fastest successful runtime in that dtype/operation/mode row is `1.00×`;",
        "for example, `0.50×` is half the fastest runtime's speed. Failed, unavailable, or zero-duration timestamp results are excluded",
        "from normalization; a zero-duration GPU result is shown as `n/a` because it is below that runtime's timestamp resolution.",
    ]
    aggregate_rows = []
    for mode in MODES:
        for relative, metric in (("gpu_relative", "GPU"), ("host_relative", "host")):
            speeds, count = aggregate_speeds(rows, devices, mode, relative)
            aggregate_rows.append(
                f"| {mode} | {metric} | {count} | " + " | ".join(f"{speeds[device]:.3f}×" if speeds else "—" for device in devices) + " |"
            )
    lines += [
        "",
        "Aggregate relative speed is the geometric mean over only the rows with valid measurements from every selected runtime.",
        "| Mode | Metric | Shared rows | " + " | ".join(devices) + " |",
        "| --- | --- | ---: | " + " | ".join("---:" for _ in devices) + " |",
        *aggregate_rows,
    ]
    metric_headers = [f"{device} GPU µs (speed)" for device in devices] + [f"{device} host µs/op (speed)" for device in devices]
    lines += [
        "",
        "| Input storage | Operation | Mode | " + " | ".join(metric_headers) + " |",
        "| --- | --- | --- | " + " | ".join("---:" for _ in metric_headers) + " |",
    ]
    labels = results["dtype_labels"]
    for dtype in results["dtypes"]:
        for operation in results["operations"]:
            for mode in MODES:
                keyed = [index.get((dtype, operation, mode, device)) for device in devices]
                cells = [metric_cell(row, "gpu_us") for row in keyed] + [metric_cell(row, "host_us") for row in keyed]
                lines.append(f"| {labels[dtype]} | {operation} | {mode} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Native storage rows: FP64, FP32, FP16, BF16, both FP8 variants, and INT8.",
        "Packed storage rows: FP6 E2M3, FP6 E3M2, FP4 E2M1, and INT4. Their timings include graph-visible decoding; packed matmul",
        "uses the package's backend-aware packed path. No backend is silently substituted when a case is unsupported.",
        "All inputs use only -1, 0, and 1. The vector, reduction, and matrix references are constructed analytically and are exactly",
        "representable in every tested output format, so every operation is checked exactly before timing. These inputs do not establish",
        "equivalence between native and packed arithmetic for general values.",
        f"Vector length: {int(results['vector_size']):,}; reduction shape: {int(results['reduction_size']):,} × "
        f"{int(results['reduction_size']):,}; square matrix size: {results['matrix_size']}.",
        f"GPU values are medians of {results['samples']} timestamp samples after 10 warmups. Host values are medians of three batches",
        f"of {results['iterations']} replays, including final synchronization. Compilation, initial transfers, and validation are excluded.",
        "Eager is a compiled ungraphed sequence; JIT uses normal TinyJit graph extraction. GPU time sums every kernel in the operation.",
        "",
        "Reproduce from the repository development environment with `DEBUG=0`, `DEV=DX12;CPU:X86`, and",
        f"`python scripts/benchmark_dtypes.py --devices {' '.join(devices)} --samples {results['samples']} "
        f"--iterations {results['iterations']} --vector-size {results['vector_size']} --reduction-size {results['reduction_size']} "
        f"--matrix-size {results['matrix_size']}`.",
        "Raw results and compiled program names are saved in `.dx12/benchmark-dtypes.json`.",
    ]
    errors = [row for row in rows if "error" in row]
    if errors:
        lines += ["", "Unmeasured/rejected cases:"]
        lines += [f"- {row['device']} {labels[row['dtype']]} {row['operation']} {row['mode']}: {row['error']}" for row in errors]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the full dtype/operation matrix and write independent JSON and Markdown reports."""
    spec_by_key = {spec.key: spec for spec in DTYPE_SPECS}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["DX12", "CL", "CUDA", "WEBGPU"])
    parser.add_argument("--dtypes", nargs="+", choices=spec_by_key, default=list(spec_by_key))
    parser.add_argument("--operations", nargs="+", choices=OPERATIONS, default=list(OPERATIONS))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--vector-size", type=int, default=1 << 20)
    parser.add_argument("--reduction-size", type=int, default=1024)
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--json", type=Path, default=Path(".dx12/benchmark-dtypes.json"))
    parser.add_argument("--report", type=Path, default=Path("PERFORMANCE-DTYPES.md"))
    args = parser.parse_args()
    if min(args.samples, args.iterations, args.vector_size, args.reduction_size, args.matrix_size) < 1:
        parser.error("sizes and iteration counts must be positive")

    register()
    selected_specs = tuple(spec_by_key[key] for key in args.dtypes)
    selected_cases = tuple(case for case in make_cases(args.vector_size, args.reduction_size, args.matrix_size) if case.name in args.operations)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "platform": platform.platform(),
        "devices": {},
        "rows": [],
        "dtypes": list(args.dtypes),
        "dtype_labels": {spec.key: spec.label for spec in selected_specs},
        "operations": list(args.operations),
        "vector_size": args.vector_size,
        "reduction_size": args.reduction_size,
        "matrix_size": args.matrix_size,
        "samples": args.samples,
        "iterations": args.iterations,
        "environment": {key: os.getenv(key) for key in ("DEV", "BEAM", "ALLOW_TF32", "OMP_NUM_THREADS", "WEBGPU_BACKEND", "WEBGPU_PATH")},
    }
    for device in args.devices:
        try:
            results["devices"][device] = comparison_device_info(device)
        except Exception as error:
            results["devices"][device] = {"error": str(error)}
            print(f"{device}: unavailable: {error}", flush=True)
            continue
        for spec in selected_specs:
            for case in selected_cases:
                try:
                    inputs = inputs_for(spec, case, device)
                    operation = operation_for(spec, case)
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"[:1200].replace("\n", " ")
                    for mode in MODES:
                        results["rows"].append({"dtype": spec.key, "operation": case.name, "device": device, "mode": mode, "error": message})
                    print(f"{device:5} {spec.label:10} {case.name:9} setup {message}", flush=True)
                    continue
                for mode in MODES:
                    row = {"dtype": spec.key, "operation": case.name, "device": device, "mode": mode}
                    try:
                        plan = capture(operation, inputs, graph=mode == "jit")
                        np.testing.assert_equal(plan.output.numpy(), case.expected)
                        row.update(graphs=plan.graphs, programs=plan.programs, **measure(plan, device, args.samples, args.iterations))
                        print(
                            f"{device:5} {spec.label:10} {case.name:9} {mode:5} GPU {row['gpu_us']:9.2f} us host {row['host_us']:9.2f} us", flush=True
                        )
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"[:1200].replace("\n", " ")
                        print(f"{device:5} {spec.label:10} {case.name:9} {mode:5} {row['error']}", flush=True)
                    results["rows"].append(row)
    add_relative_speeds(results["rows"])
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.report.write_text(report(results), encoding="utf-8")


if __name__ == "__main__":
    main()
