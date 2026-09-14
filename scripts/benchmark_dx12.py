"""Compare complete compiled operations on DX12, OpenCL, and WebGPU.

Run with the repository's development environment (NumPy is used for independent
references). Inputs and output checks are identical across runtimes. Eager rows
replay tinygrad's compiled, ungraphed sequence; JIT rows use normal TinyJit graph
extraction. Both timings exclude initial compilation, initial transfers, and verification.
GPU time is the sum of backend timestamp spans for every kernel or graph in the
operation. Host time includes Python replay, submission, and batch synchronization.
"""

import argparse, ctypes, hashlib, json, os, platform, statistics, sys, time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from importlib.metadata import distribution
import numpy as np
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.engine.realize import run_linear
from tinygrad.helpers import Context, GlobalCounters
from tinygrad.uop.ops import Ops, UOp
from tinygrad_dx12 import register


@dataclass
class Plan:
    """One compiled operation, including graph-visible conversions and activation."""

    linear: UOp
    inputs: tuple[UOp, ...]
    output: Tensor
    graphs: int
    programs: tuple[str, ...]

    def run(self, wait: bool = False) -> float:
        """Replay without frontend construction and return summed GPU timestamp seconds."""
        before = GlobalCounters.time_sum_s
        run_linear(self.linear, input_uops=self.inputs, jit=True, wait=wait, update_stats=wait)
        return GlobalCounters.time_sum_s - before


def capture(operation: Callable[..., Tensor], inputs: tuple[Tensor, ...], graph: bool) -> Plan:
    """Use tinygrad's ordinary capture path with graph extraction enabled or disabled."""
    runner = TinyJit(operation)
    with Context(JIT=1 if graph else 2):
        runner(*inputs)
        output = runner(*inputs)
    assert runner.captured is not None
    linear = runner.captured.linear
    calls = [
        nested
        for call in linear.src
        for nested in (call.src[0].src[0].src if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "graph" else (call,))
    ]
    return Plan(
        linear,
        tuple(t.uop.base for t in inputs),
        output,
        sum(call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "graph" for call in linear.src),
        tuple(call.src[0].arg.name for call in calls if call.src[0].op is Ops.PROGRAM),
    )


def vector(x: Tensor) -> Tensor:
    """Execute two elementwise stages with one materialized intermediate."""
    return (x * 1.25 + 0.5).realize().square().realize()


def reduction(x: Tensor) -> Tensor:
    """Execute a materialized square followed by row reductions."""
    return x.square().realize().sum(axis=1).realize()


def matrix(a: Tensor, b: Tensor) -> Tensor:
    """Execute matrix multiplication and a separate ReLU activation."""
    return (a @ b).realize().relu().realize()


def measure(plan: Plan, device: str, samples: int, iterations: int) -> dict[str, float | None]:
    """Measure warmed GPU spans and median synchronized throughput separately."""
    for _ in range(10):
        plan.run()
    Device[device].synchronize()
    gpu = [plan.run(wait=True) for _ in range(samples)]
    for _ in range(10):
        plan.run()
    Device[device].synchronize()
    host = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(iterations):
            plan.run()
        Device[device].synchronize()
        host.append((time.perf_counter() - start) / iterations)
    return {
        "gpu_us": statistics.median(gpu) * 1e6 if all(gpu) else None,
        "host_us": statistics.median(host) * 1e6,
        "gpu_min_us": min(gpu) * 1e6 if all(gpu) else None,
        "gpu_max_us": max(gpu) * 1e6 if all(gpu) else None,
        "host_min_us": min(host) * 1e6,
        "host_max_us": max(host) * 1e6,
    }


def device_info(device: str) -> dict:
    """Report renderer and adapter identity without assuming ordinal equality across APIs."""
    dev = Device[device]
    info = {"renderer": type(dev.renderer).__name__, "arch": dev.arch}
    if device.startswith("DX12"):
        from tinygrad_dx12.setup import PACKAGES

        info.update(name=dev.adapter.name, luid=dev.adapter.luid, toolchain=[p.__dict__ for p in PACKAGES])
    elif device.startswith("CL"):
        from tinygrad.runtime.autogen import opencl as cl
        from tinygrad.runtime.ops_cl import check

        info.update(name=dev.device_name, driver=dev.driver_version)
        if "cl_khr_device_uuid" in dev.device_exts:
            # Khronos CL/cl_ext.h: LUID_VALID_KHR=0x106C, LUID_KHR=0x106D; omitted from tinygrad's core-only bindings.
            valid, luid = ctypes.c_uint32(), ctypes.create_string_buffer(8)
            check(cl.clGetDeviceInfo(dev.cl_dev, 0x106C, 4, ctypes.byref(valid), None))
            if valid.value:
                check(cl.clGetDeviceInfo(dev.cl_dev, 0x106D, 8, luid, None))
                info["luid"] = int.from_bytes(luid.raw, "little")
    elif device.startswith("WEBGPU"):
        from tinygrad.runtime.autogen import webgpu
        from tinygrad.runtime.ops_webgpu import from_wgpu_str

        adapter = webgpu.WGPUAdapterInfo()
        if webgpu.wgpuDeviceGetAdapterInfo(dev.device_res, adapter) != webgpu.WGPUStatus_Success:
            raise RuntimeError("WebGPU adapter information is unavailable")
        try:
            info.update(
                name=from_wgpu_str(adapter.device), driver=from_wgpu_str(adapter.description),
                backend=webgpu.enum_WGPUBackendType[adapter.backendType],
                timestamp_queries=webgpu.WGPUFeatureName_TimestampQuery in dev.features,
            )
        finally:
            webgpu.wgpuAdapterInfoFreeMembers(adapter)
        library = Path(webgpu.dll._name)
        with library.open("rb") as stream:
            info.update(library=str(library), library_sha256=hashlib.file_digest(stream, "sha256").hexdigest())
    elif device.startswith("CUDA"):
        from tinygrad.runtime.autogen import cuda
        from tinygrad.runtime.ops_cuda import check

        name, luid, mask = ctypes.create_string_buffer(256), ctypes.create_string_buffer(8), ctypes.c_uint32()
        check(cuda.cuDeviceGetName(name, 256, dev.cu_device))
        check(cuda.cuDeviceGetLuid(luid, ctypes.byref(mask), dev.cu_device))
        info.update(name=name.value.decode(), luid=int.from_bytes(luid.raw, "little"))
    return info


def report(results: dict) -> str:
    """Write measured times, comparison limits, and commands to repeat the run."""
    revision = results["tinygrad_source"].get("vcs_info", {}).get("commit_id", "unknown")
    lines = [
        "# Performance", "",
        f"Measured {results['timestamp']} on {results['platform']}.",
        f"Python: {results['python'].split()[0]}. Tinygrad source: `{revision}`.", "",
        "These results apply to this computer and these operations. Lower times are better.", "",
        "## Hardware", "",
        "| Runtime | Adapter | Driver or backend |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {device} | {info.get('name', 'unavailable')} | {info.get('driver', info.get('error', info['renderer']))} |"
        if "error" not in info else f"| {device} | unavailable | {info['error']} |"
        for device, info in results["devices"].items()
    ]
    dx_luid = results["devices"].get("DX12", {}).get("luid")
    cl_luid = results["devices"].get("CL", {}).get("luid")
    lines += ["", "DX12 and OpenCL report the same adapter LUID."
              if dx_luid is not None and cl_luid == dx_luid else "DX12 and OpenCL physical adapter identity is not confirmed."]
    if "WEBGPU" in results["devices"]:
        info = results["devices"]["WEBGPU"]
        lines += [f"WebGPU backend: `{info.get('backend', 'unavailable')}`.",
                  "WebGPU reports its adapter name and driver. This report does not check its LUID."]
    lines += ["", "## Comparison", ""]
    for mode in ("eager", "jit"):
        rows = {(row["device"], row["operation"]): row for row in results["rows"] if row["mode"] == mode and "error" not in row}
        for other in ("CL", "WEBGPU"):
            common = [name for device, name in rows if device == "DX12" and (other, name) in rows]
            if common:
                wins = sum(rows["DX12", name]["host_us"] < rows[other, name]["host_us"] for name in common)
                lines += [f"DX12 {mode} has lower host time than {other} in {wins} of {len(common)} measured operations."]
    lines += ["", "| Operation | Runtime | Mode | Graphs | GPU us | Host us/op |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in results["rows"]:
        if "error" in row:
            lines.append(f"| {row['operation']} | {row['device']} | {row['mode']} | - | Failed | - |")
        else:
            gpu = f"{row['gpu_us']:.2f}" if row["gpu_us"] is not None else "Unavailable"
            lines.append(f"| {row['operation']} | {row['device']} | {row['mode']} | {row['graphs']} | {gpu} | {row['host_us']:.2f} |")
    jit = {(row["device"], row["operation"]): row for row in results["rows"] if row["mode"] == "jit" and "error" not in row}
    common = [name for device, name in jit if device == "DX12" and all((other, name) in jit for other in ("CL", "WEBGPU"))]
    if common:
        lines += [
            "", "## Diagnosis", "",
            "The following ratios use JIT host time. A ratio above 1 means DX12 is faster.",
            "A ratio below 1 means the other runtime is faster.", "",
            "| Operation | OpenCL time / DX12 time | WebGPU time / DX12 time |",
            "| --- | ---: | ---: |",
        ]
        lines += [
            f"| {name} | {jit['CL', name]['host_us'] / jit['DX12', name]['host_us']:.2f} | "
            f"{jit['WEBGPU', name]['host_us'] / jit['DX12', name]['host_us']:.2f} |"
            for name in common
        ]
        lines += [
            "", "The relative speed depends on the operation. Check each row before selecting a runtime.",
            "DX12 uses native graphs for these JIT operations. This can reduce submission work, but does not guarantee a lower host time.",
            "Matrix work can also include staging and output conversion. The raw results list the compiled programs.", "",
            "WebGPU creates bind groups, pipeline layouts, and compute pipelines on each kernel call in this tinygrad revision.",
            "It also checks error scopes through synchronous callbacks. This work is included in host time.",
            "This is a possible cause of its high host time; this measurement does not isolate each cost.",
            f"See the [measured WebGPU implementation](https://github.com/tinygrad/tinygrad/blob/{revision}/tinygrad/runtime/ops_webgpu.py).",
            "These results describe tinygrad's runtime implementations, not a general speed limit of the graphics APIs.",
        ]
        if "shader-f16" not in results["devices"]["WEBGPU"]["arch"]:
            lines += [
                "", "WebGPU did not expose the shader-f16 feature in this run.",
                "Tinygrad therefore lowered FP16 operations to supported types. The FP16 result includes this extra work.",
                f"See the [WGSL dtype support](https://github.com/tinygrad/tinygrad/blob/{revision}/tinygrad/renderer/wgsl.py).",
                "This result does not measure native WebGPU FP16 arithmetic.",
            ]
    lines += [
        "", "## Method and limits", "",
        "Each result is checked against NumPy before timing. The random seed is 20260911.",
        f"Vector length: {results['vector_size']:,}. Reduction shape: 1024 by 1024.",
        f"Matrix shape: {results['matrix_size']} by {results['matrix_size']}.",
        "The vector operation has two stages: multiply and add, then square.",
        "The reduction squares its input, then sums each row. Matrix operations multiply two matrices, then apply ReLU.",
        "Each stage stores its result. These operations therefore measure more than one kernel.", "",
        f"GPU time is the median of {results['samples']} samples after 10 warmup calls.",
        f"Host time is the median of three batches of {results['iterations']} calls, with a device wait at each batch end.",
        "Ten more warmup calls separate GPU timing from host timing.",
        "Host time includes Python replay, command submission, and the final wait.",
        "It excludes tensor construction, initial input copies, initial compilation, and result checks.",
        "Runtime work done on every replay remains in the host measurement.", "",
        "Eager mode replays a compiled sequence without graphs. It does not measure the Tensor front end.",
        "JIT mode uses TinyJit. OpenCL and WebGPU do not have native graph support in this tinygrad revision.",
        "Their JIT rows still use separate kernel calls. Both modes use the same replay function.", "",
        "GPU times sum the timestamp spans for all kernels or graphs in an operation.",
        "Separate kernel spans exclude queue gaps. DX12 graph spans include gaps inside the graph.",
        "OpenCL measures kernel events. WebGPU measures compute passes when timestamp queries are available.",
        "If any sampled operation returns zero GPU time, its GPU result is shown as unavailable.",
        "Zero can mean missing timing support or insufficient timer precision. Host time is the main comparison metric.",
        "Required DX12 matrix staging copies are included in host time. Copy time is included in GPU time only when the timed scope covers it.", "",
        "FP16 matrix inputs and ordinary outputs use FP16. The reference uses FP32 multiplication of the FP16 inputs.",
        "FP16 tolerances are 0.003 absolute and relative. FP32 tolerances are 0.0001 absolute and 0.001 relative.",
        "Optional explicit LinAlg rows use a named DX12 scope and can return FP32. They are not shared comparison cases.",
        "One run does not measure power use, cold start, training, or all tensor sizes. Small differences can be measurement noise.",
        "The JSON file records host batch ranges, GPU sample ranges, programs, library hashes, and environment settings.",
        "", "## Repeat the measurement", "",
        "Run from a checkout with the package and NumPy installed.",
        "For WebGPU, use the Dawn binary linked by the [tinygrad runtime guide](https://docs.tinygrad.org/runtime/).",
        "Set `WEBGPU_PATH` to the full path of `libwebgpu_dawn.dll` if the library is not in the default location.",
        "This library is needed only for the WebGPU comparison.", "",
        "Environment used for this run:", "", "```json",
        json.dumps(results["environment"], indent=2), "```", "",
        "A null value means that the variable was not set. No DEV override is required.", "",
        "```powershell", " ".join(results["command"]), "```", "",
        "The command writes the Markdown report and the raw JSON file. Keep the JSON file to compare later runs.",
    ]
    errors = [row for row in results["rows"] if "error" in row]
    if errors:
        lines += ["", "## Failed operations", ""]
        lines += [f"- {row['device']} {row['operation']} {row['mode']}: {row['error']}" for row in errors]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run repeatable comparisons and write raw JSON plus a readable report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", nargs="+", default=["DX12", "CL", "WEBGPU"])
    parser.add_argument("--operations", nargs="+", help="restrict to named operations shown in the report")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--vector-size", type=int, default=1 << 20)
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--json", type=Path, default=Path("PERFORMANCE.json"))
    parser.add_argument("--report", type=Path, default=Path("PERFORMANCE.md"))
    parser.add_argument("--skip-linalg", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.iterations, args.vector_size, args.matrix_size) < 1:
        parser.error("sizes and iteration counts must be positive")
    register()
    rng = np.random.default_rng(20260911)
    x = rng.uniform(-0.5, 0.5, args.vector_size).astype(np.float32)
    r = rng.uniform(-0.5, 0.5, (1024, 1024)).astype(np.float32)
    a, b = (rng.uniform(-0.5, 0.5, (args.matrix_size, args.matrix_size)).astype(np.float32) for _ in range(2))
    cases = [
        ("vector_f32", vector, (x,), np.square(x * 1.25 + 0.5)),
        ("reduction_f32", reduction, (r,), np.square(r).sum(axis=1)),
        ("matmul_f32", matrix, (a, b), np.maximum(a @ b, 0)),
        (
            "matmul_f16",
            matrix,
            (a.astype(np.float16), b.astype(np.float16)),
            np.maximum(a.astype(np.float16).astype(np.float32) @ b.astype(np.float16).astype(np.float32), 0),
        ),
    ]
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "tinygrad_source": json.loads(distribution("tinygrad").read_text("direct_url.json") or "{}"),
        "devices": {},
        "rows": [],
        "vector_size": args.vector_size,
        "matrix_size": args.matrix_size,
        "samples": args.samples,
        "iterations": args.iterations,
        "environment": {key: os.getenv(key) for key in (
            "DEV", "DEBUG", "BEAM", "ALLOW_TF32", "OMP_NUM_THREADS", "WEBGPU_BACKEND", "WEBGPU_PATH",
            *(key for key in os.environ if key.startswith(("DX12_", "TINYGRAD_DX12_"))),
        )},
        "command": ["python", *sys.argv],
    }
    for device in args.devices:
        try:
            results["devices"][device] = device_info(device)
        except Exception as error:
            results["devices"][device] = {"error": str(error)}
            print(f"{device}: unavailable: {error}", flush=True)
            continue
        device_cases = list(cases)
        if device.split(":")[0] == "DX12" and not args.skip_linalg:
            from tinygrad_dx12.codegen.linalg import matmul

            for scope in ("Wave", "ThreadGroup"):

                def linalg(left: Tensor, right: Tensor, scope: str = scope) -> Tensor:
                    """Time an explicit accepted LinAlg scope with its surrounding operations."""
                    return matmul(left, right, scope=scope).realize().relu().realize()

                device_cases.append((f"linalg_f16_{scope}", linalg, cases[3][2], cases[3][3]))
        for name, operation, arrays, expected in device_cases:
            if args.operations is not None and name not in args.operations:
                continue
            for mode in ("eager", "jit"):
                row = {"operation": name, "device": device, "mode": mode}
                try:
                    inputs = tuple(
                        Tensor(data, device=device, dtype=dtypes.float16 if data.dtype == np.float16 else dtypes.float32).realize() for data in arrays
                    )
                    plan = capture(operation, inputs, graph=mode == "jit")
                    half = "f16" in name
                    np.testing.assert_allclose(plan.output.numpy(), expected, rtol=0.003 if half else 0.001, atol=0.003 if half else 0.0001)
                    row.update(graphs=plan.graphs, programs=plan.programs, **measure(plan, device, args.samples, args.iterations))
                    gpu = f"{row['gpu_us']:.2f} us" if row["gpu_us"] is not None else "unavailable"
                    print(f"{device:6} {name:24} {mode:5} GPU {gpu:>14} host {row['host_us']:9.2f} us", flush=True)
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"[:1200].replace("\n", " ")
                    print(f"{device:5} {name:24} {mode:5} {row['error']}", flush=True)
                results["rows"].append(row)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    args.report.write_text(report(results), encoding="utf-8")


if __name__ == "__main__":
    main()
