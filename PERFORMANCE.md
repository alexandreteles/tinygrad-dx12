# Performance

Measured 2026-09-13 22:23:44 -0300 on Windows-11-10.0.26200-SP0.
Python: 3.14.4. Tinygrad source: `e4570fcb46b3725d3eb9243507a507b4ae19131b`.

Raw data: [PERFORMANCE.json](PERFORMANCE.json).

These results apply to this computer and these operations. Lower times are better.

## Hardware

| Runtime | Adapter | Driver or backend |
| --- | --- | --- |
| DX12 | NVIDIA GeForce RTX 4070 Ti SUPER | DX12Renderer |
| CL | NVIDIA GeForce RTX 4070 Ti SUPER | 616.92 |
| WEBGPU | NVIDIA GeForce RTX 4070 Ti SUPER | D3D12 driver version 32.0.16.1692 |

DX12 and OpenCL report the same adapter LUID.
WebGPU backend: `WGPUBackendType_D3D12`.
WebGPU reports its adapter name and driver. This report does not check its LUID.

## Comparison

DX12 eager has lower host time than CL in 3 of 4 measured operations.
DX12 eager has lower host time than WEBGPU in 4 of 4 measured operations.
DX12 jit has lower host time than CL in 2 of 4 measured operations.
DX12 jit has lower host time than WEBGPU in 4 of 4 measured operations.

| Operation | Runtime | Mode | Graphs | GPU us | Host us/op |
| --- | --- | --- | ---: | ---: | ---: |
| vector_f32 | DX12 | eager | 0 | 11.49 | 29.85 |
| vector_f32 | DX12 | jit | 1 | 11.70 | 50.09 |
| reduction_f32 | DX12 | eager | 0 | 14.06 | 30.30 |
| reduction_f32 | DX12 | jit | 1 | 13.65 | 52.77 |
| matmul_f32 | DX12 | eager | 0 | 66.43 | 74.59 |
| matmul_f32 | DX12 | jit | 1 | 66.98 | 74.79 |
| matmul_f16 | DX12 | eager | 0 | 27.12 | 55.87 |
| matmul_f16 | DX12 | jit | 1 | 24.45 | 40.04 |
| vector_f32 | CL | eager | 0 | 14.74 | 37.18 |
| vector_f32 | CL | jit | 0 | 14.99 | 38.32 |
| reduction_f32 | CL | eager | 0 | 17.41 | 65.82 |
| reduction_f32 | CL | jit | 0 | 17.41 | 68.41 |
| matmul_f32 | CL | eager | 0 | 69.60 | 69.16 |
| matmul_f32 | CL | jit | 0 | 69.94 | 69.56 |
| matmul_f16 | CL | eager | 0 | 92.21 | 88.89 |
| matmul_f16 | CL | jit | 0 | 92.16 | 93.56 |
| vector_f32 | WEBGPU | eager | 0 | Unavailable | 7866.57 |
| vector_f32 | WEBGPU | jit | 0 | Unavailable | 7723.00 |
| reduction_f32 | WEBGPU | eager | 0 | Unavailable | 9863.01 |
| reduction_f32 | WEBGPU | jit | 0 | 131.07 | 9899.38 |
| matmul_f32 | WEBGPU | eager | 0 | 4325.38 | 42264.77 |
| matmul_f32 | WEBGPU | jit | 0 | 884.74 | 42566.07 |
| matmul_f16 | WEBGPU | eager | 0 | 15990.78 | 221458.55 |
| matmul_f16 | WEBGPU | jit | 0 | 17137.66 | 212427.35 |

## Diagnosis

The following ratios use JIT host time. A ratio above 1 means DX12 is faster.
A ratio below 1 means the other runtime is faster.

| Operation | OpenCL time / DX12 time | WebGPU time / DX12 time |
| --- | ---: | ---: |
| vector_f32 | 0.77 | 154.20 |
| reduction_f32 | 1.30 | 187.61 |
| matmul_f32 | 0.93 | 569.15 |
| matmul_f16 | 2.34 | 5305.25 |

The relative speed depends on the operation. Check each row before selecting a runtime.
DX12 uses native graphs for these JIT operations. This can reduce submission work, but does not guarantee a lower host time.
Matrix work can also include staging and output conversion. The raw results list the compiled programs.

WebGPU creates bind groups, pipeline layouts, and compute pipelines on each kernel call in this tinygrad revision.
It also checks error scopes through synchronous callbacks. This work is included in host time.
This is a possible cause of its high host time; this measurement does not isolate each cost.
See the [measured WebGPU implementation](https://github.com/tinygrad/tinygrad/blob/e4570fcb46b3725d3eb9243507a507b4ae19131b/tinygrad/runtime/ops_webgpu.py).
These results describe tinygrad's runtime implementations, not a general speed limit of the graphics APIs.

WebGPU did not expose the shader-f16 feature in this run.
Tinygrad therefore lowered FP16 operations to supported types. The FP16 result includes this extra work.
See the [WGSL dtype support](https://github.com/tinygrad/tinygrad/blob/e4570fcb46b3725d3eb9243507a507b4ae19131b/tinygrad/renderer/wgsl.py).
This result does not measure native WebGPU FP16 arithmetic.

## Method and limits

Each result is checked against NumPy before timing. The random seed is 20260911.
Vector length: 1,048,576. Reduction shape: 1024 by 1024.
Matrix shape: 512 by 512.
The vector operation has two stages: multiply and add, then square.
The reduction squares its input, then sums each row. Matrix operations multiply two matrices, then apply ReLU.
Each stage stores its result. These operations therefore measure more than one kernel.

GPU time is the median of 20 samples after 10 warmup calls.
Host time is the median of three batches of 100 calls, with a device wait at each batch end.
Ten more warmup calls separate GPU timing from host timing.
Host time includes Python replay, command submission, and the final wait.
It excludes tensor construction, initial input copies, initial compilation, and result checks.
Runtime work done on every replay remains in the host measurement.

Eager mode replays a compiled sequence without graphs. It does not measure the Tensor front end.
JIT mode uses TinyJit. OpenCL and WebGPU do not have native graph support in this tinygrad revision.
Their JIT rows still use separate kernel calls. Both modes use the same replay function.

GPU times sum the timestamp spans for all kernels or graphs in an operation.
Separate kernel spans exclude queue gaps. DX12 graph spans include gaps inside the graph.
OpenCL measures kernel events. WebGPU measures compute passes when timestamp queries are available.
If any sampled operation returns zero GPU time, its GPU result is shown as unavailable.
Zero can mean missing timing support or insufficient timer precision. Host time is the main comparison metric.
Required DX12 matrix staging copies are included in host time. Copy time is included in GPU time only when the timed scope covers it.

FP16 matrix inputs and ordinary outputs use FP16. The reference uses FP32 multiplication of the FP16 inputs.
FP16 tolerances are 0.003 absolute and relative. FP32 tolerances are 0.0001 absolute and 0.001 relative.
Optional explicit LinAlg rows use a named DX12 scope and can return FP32. They are not shared comparison cases.
One run does not measure power use, cold start, training, or all tensor sizes. Small differences can be measurement noise.
The JSON file records host batch ranges, GPU sample ranges, programs, library hashes, and environment settings.

## Repeat the measurement

Run from a checkout with the package and NumPy installed.
For WebGPU, use the Dawn binary linked by the [tinygrad runtime guide](https://docs.tinygrad.org/runtime/).
Set `WEBGPU_PATH` to the full path of `libwebgpu_dawn.dll` if the library is not in the default location.
This library is needed only for the WebGPU comparison.

Environment used for this run:

```json
{
  "DEV": null,
  "DEBUG": "0",
  "BEAM": null,
  "ALLOW_TF32": null,
  "OMP_NUM_THREADS": null,
  "WEBGPU_BACKEND": null,
  "WEBGPU_PATH": "C:\\Users\\alexs\\OneDrive\\Documentos\\Projetos\\tinygrad-dx12\\.dx12\\webgpu\\libwebgpu_dawn.dll"
}
```

A null value means that the variable was not set. No DEV override is required.

```powershell
python scripts/benchmark_dx12.py --devices DX12 CL WEBGPU --samples 20 --iterations 100 --skip-linalg --json .dx12/runtime-comparison.json --report PERFORMANCE.md
```

The command writes the Markdown report and the raw JSON file. Keep the JSON file to compare later runs.
