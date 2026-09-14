# tinygrad-dx12

Use Direct3D 12 to run [tinygrad](https://github.com/tinygrad/tinygrad) on a Windows GPU.

## Install

You need Windows, Git, 64-bit Python 3.14 or later, and a GPU driver with
Enhanced Barriers support. Enable Windows Developer Mode before first use.
Hardware operation has been checked on x64 only.

Install from Git HEAD. This package is not published on PyPI.
Replace the placeholder URL with the repository URL:

With `pip`:

```powershell
python -m pip install --upgrade --force-reinstall "git+https://<HOST>/<OWNER>/tinygrad-dx12.git"
```

With [uv](https://docs.astral.sh/uv/pip/packages/), create and activate a virtual
environment, then install the package:

```powershell
uv venv --python 3.14
.\.venv\Scripts\Activate.ps1
uv pip install --upgrade --reinstall "git+https://<HOST>/<OWNER>/tinygrad-dx12.git"
```

Both methods also install tinygrad from Git HEAD. Repeat the `pip install` or
`uv pip install` command to update both packages. A virtual environment keeps
these packages separate from other projects.

On first DX12 use, the runtime downloads and checks the required Agility SDK and
DXC compiler files. It stores them in `%LOCALAPPDATA%/tinygrad_dx12` for later use.
You do not need Visual Studio or a separate Windows SDK installation.

## Support matrix

The Windows versions below include the required D3D12 Independent Devices API.
Install current Windows updates and a compatible Windows graphics driver.

| Component | Minimum requirement or documented starting point |
| --- | --- |
| Direct3D | Direct3D 12, feature level `11_0`, Shader Model 6.0, root signatures 1.1, and Enhanced Barriers. |
| Agility SDK | This package uses `1.721.3-preview` and installs it automatically. Enhanced Barriers entered the stable SDK in [1.608.0][enhanced-release]. |
| Windows 10 | 22H2, build **19045.3393**, with [KB5029331][win10-dx12] or a later update. |
| Windows 11 | 21H2, build **22000.2360**, with [KB5029332][win11-dx12] or a later update. Later releases also need the August 2023 or newer cumulative updates. |
| Windows Server | Server 2022, build **20348.1970**, with [KB5030216][server-dx12] or a later update; or a later server release. |
| Developer Mode | Required. The package enables experimental shader models when it opens DX12. |
| Python | 3.14 or later, 64-bit. Package execution has been checked on x64 only. |

Microsoft lists the Enhanced Barriers driver interface under WDDM 3.0, but also
allows its use on earlier Windows releases through the Agility SDK. WDDM 3.0
alone does not guarantee support. The runtime checks the driver's
`EnhancedBarriersSupported` flag. See [Microsoft's Enhanced Barriers guide][enhanced-barriers].

The GPU support matrix below lists *some* of the known architectures and driver versions that *might* work.

### Dedicated GPUs

| Manufacturer | GPU families | Examples | Driver reference |
| --- | --- | --- | --- |
| AMD | RDNA 1, 2, 3, and 4 | Radeon RX 5700 XT, RX 6800, RX 7900 XTX, RX 9070 XT; Radeon PRO W7900; [Radeon PRO V620][amd-v620] for servers | [Adrenalin 23.9.2][amd-barriers] added Enhanced Barriers for RDNA (`31.0.22011.1008`). Use a compatible graphics driver for Radeon PRO. |
| Intel | Xe-HPG (Alchemist), Xe2-HPG (Battlemage) | Arc A380, A770, B580; [Data Center GPU Flex 140 and 170][intel-flex] | [31.0.101.3959][intel-barriers] added Enhanced Barriers for Arc A-series. Flex driver example: [32.0.101.8331][intel-flex]. |
| NVIDIA | Maxwell 1 and 2, Pascal, Turing, Ampere, Ada, and Blackwell | GeForce GTX 750 Ti, GTX 900/10/16 series, RTX 20/30/40/50 series; [Tesla M60/P40 and NVIDIA T4, A10, L4, L40S][nvidia-server] for servers | [526.98][enhanced-release] added Enhanced Barriers. Server graphics driver example: [536.25][nvidia-server]. |
| Matrox | Intel Xe-HPG (Alchemist) | [LUMA A310, A310F, and A380][matrox-luma]; [LUMA Pro A380 Octal][matrox-luma-pro] | Compatible Intel Arc driver, or a Matrox driver that exposes the required features. [LUMA 8.03.00][matrox-luma] is a driver reference for A310/A380 cards. |

Important: NVIDIA datacenter cards must use [WDDM graphics driver mode][nvidia-wddm] to expose D3D12, meaning that vGPU is
required.

### Integrated GPUs

| Manufacturer | GPU families | Examples | Driver reference |
| --- | --- | --- | --- |
| AMD | GCN 5 (Vega); RDNA 2, 3, and 3.5 | [Radeon Vega 8 in Ryzen 5 2500U][amd-vega-mobile]; [Radeon RX Vega 11 in Ryzen 5 2400G][amd-vega-desktop]; Radeon 680M, 780M, and 890M | Vega: special [22.6.1 Agility driver][amd-preview] (`30.0.21023.5005`); see the note below. RDNA: [680M with `32.0.21007.1034`][report-680m]. |
| Intel | Xe-LPG (Meteor Lake), Xe-LPG+ (Arrow Lake), Xe2 (Lunar Lake), Xe3-LPG (Panther Lake) | Arc graphics in Core Ultra 100H; Arc 140V; Arc B390 | [Meteor Lake, `32.0.101.6736`][report-meteor]; [140V, `32.0.101.5736`][report-lunar]; [B390, `32.0.101.8826`][report-panther]. |
| Qualcomm | Adreno X1 and X2 | Snapdragon X Plus and X Elite; [Snapdragon X2 Elite with Adreno X2-85][qualcomm-x2] | [X Plus, `31.0.53.1`][report-x1]; [X2-85, `32.0.147.8`][report-x2]. Use Windows on Arm. |

**Vega support:** [AMD announced Enhanced Barriers support][amd-preview-announcement]
for GPUs supported at that time with the special 22.6.1 driver and Agility SDK
`1.706`.

Native FP16, FP64, and matrix operations depend on additional GPU features.
The driver versions above do not guarantee the newer [preview LinAlg features][linalg-support].
The runtime queries each adapter and uses normal tensor operations where a
supported fallback exists.

To check your own adapter and make sure it's compatible, run:

```powershell
python -m tinygrad_dx12.diagnostics --probe
```

A false Enhanced Barriers result prevents DX12 from opening. The runtime has
no fallback to legacy resource barriers (yet?).

## Use

Import the package before you create tensors:

```python
import tinygrad_dx12
from tinygrad import Tensor, TinyJit

tinygrad_dx12.register()  # Optional. The import already registers DX12.

x = Tensor([1.0, 2.0, 3.0])

print(x.device)           # DX12 when available

@TinyJit
def compute(x: Tensor) -> Tensor:
    """Multiply the input by two, then square the result."""
    return (x * 2).realize().square().realize()

for _ in range(5):
    print(compute(x).tolist())  # [4.0, 16.0, 36.0]
```

Tinygrad tries DX12 first. If DX12 cannot open, it tries its other runtimes.
Registration does not open a GPU. A default selected before registration stays
unchanged. You do not need to set `DEV=DX12`.

To require DX12 for a tensor, set its device:

```python
x = Tensor([1.0, 2.0, 3.0], device="DX12")
print((x * 2).tolist())
```

An explicit DX12 request reports an error if DX12 cannot open.
`DX12` and `DX12:0` select the first adapter. Use `DX12:1` for the second adapter.
Each number selects a different physical adapter.

Use `DEV` to override automatic selection. For example, in PowerShell:

```powershell
$env:DEV = "CL"
python example.py
```

This sets the default to OpenCL. An explicit tensor device still takes precedence.
Use `Remove-Item Env:DEV` to restore automatic selection for later runs.

### Packed tensors and matrix operations

For most matrix operations, use `a @ b`. DX12 selects native matrix operations
when the data types and shapes are supported. Other cases use normal tinygrad
operations.

Use `PackedTensor` for values stored in fewer than eight bits. It accepts a flat
`uint8` tensor, a format name, and the shape of the decoded tensor:

```python
import tinygrad_dx12
from tinygrad import Tensor, dtypes
from tinygrad_dx12.codegen.decomp.dx12 import PackedTensor
from tinygrad_dx12.codegen.linalg import matmul, matvec

# Each byte holds two UINT4 values. The lower four bits hold the first value.
data = Tensor([0x21, 0x43], device="DX12", dtype=dtypes.uint8)
packed = PackedTensor(data, "uint4", (2, 2))
print(packed.unpack().tolist())  # [[1, 2], [3, 4]]

column = Tensor([[1.0], [2.0]], device="DX12")
print(matmul(packed, column).tolist())  # [[5.0], [11.0]]

vector = Tensor([1.0, 2.0], device="DX12")
print(matvec(packed, vector).tolist())  # [5.0, 11.0]
```

Supported formats are `int4`, `uint4`, `fp4_e2m1`, `fp6_e2m3`, and `fp6_e3m2`.
Store values from the lowest bits to the highest bits. Six-bit values can cross
byte boundaries. Optional `scales=` multiply blocks along the final dimension.
Use `block_size=` to set the number of values per block; the default is 32.
Decoding and scaling run as normal tensor operations on the selected device.

The matrix functions accept packed and normal tensors on the same device:

- `matmul(a, b)` multiplies two matrices. Its default `scope="auto"` selects a
  supported native operation, with normal tensor operations as a fallback.
- `scope="Wave"` or `scope="ThreadGroup"` requests a specific native method.
  Unsupported requests report an error. `scope="tiled"` uses normal tensor operations.
- `matmul(..., dtype=...)` sets the accumulator and output data type.
- `matvec(matrix, vector)` multiplies a matrix by a vector. Use `output_dtype=`
  to set the result type. `thread=True` requests the native thread method when supported.

The custom native matrix operations do not define gradients. For training, use
normal tensor operations such as `a @ b`.

See [PERFORMANCE.md](PERFORMANCE.md) for measurements against OpenCL and WebGPU.
See [ARCHITECTURE.md](ARCHITECTURE.md) for runtime details and offline setup.

[enhanced-release]: https://devblogs.microsoft.com/directx/agility-sdk-1-608-0/
[enhanced-barriers]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/enhanced-barriers
[win10-dx12]: https://support.microsoft.com/en-us/servicing/os/windows-10/2023/08/august-22-2023-kb5029331-os-build-19045-3393-preview
[win11-dx12]: https://support.microsoft.com/en-us/servicing/os/windows-11/2023/08/august-22-2023-kb5029332-os-build-22000-2360-preview
[server-dx12]: https://support.microsoft.com/en-US/servicing/os/windows-server/2023/09/september-12-2023-kb5030216-os-build-20348-1970
[report-680m]: https://d3d12infodb.boolka.dev/index.html?ID=917
[report-meteor]: https://d3d12infodb.boolka.dev/index.html?ID=645
[report-lunar]: https://d3d12infodb.boolka.dev/index.html?ID=1271
[report-panther]: https://d3d12infodb.boolka.dev/index.html?ID=1448
[report-x1]: https://d3d12infodb.boolka.dev/index.html?ID=260
[report-x2]: https://d3d12infodb.boolka.dev/index.html?ID=1366
[amd-barriers]: https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-23-9-2.html
[amd-preview-announcement]: https://gpuopen.com/news/shader-model-6-7-agility-sdk/
[amd-preview]: https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-2022-1-706.html
[amd-v620]: https://www.amd.com/en/products/accelerators/radeon-pro/amd-radeon-pro-v620.html
[amd-vega-mobile]: https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen/ryzen-2000-series/amd-ryzen-5-2500u.html
[amd-vega-desktop]: https://www.amd.com/en/support/downloads/drivers.html/processors/ryzen/ryzen-2000-series/amd-ryzen-5-2400g.html
[intel-barriers]: https://dl.comss.org/download/ReleaseNotes_101.3959_WHQL.pdf
[intel-flex]: https://www.intel.com/content/www/us/en/download/780185/intel-data-center-gpu-flex-series-windows.html
[nvidia-server]: https://docs.nvidia.com/vgpu/16.0/grid-vgpu-release-notes-microsoft-windows-server/index.html
[nvidia-wddm]: https://docs.nvidia.com/datacenter/tesla/pdf/NVIDIA_Data_Center_GPU_Driver_Release_Notes_550_v4.0.pdf
[matrox-luma]: https://ftp.matrox.com/pub/mga/archive/Luma/2024/2024-Jul/Matrox%20LUMA%20Series%20Release%20Notes_v.8.03.00.pdf
[matrox-luma-pro]: https://video.matrox.com/en/products/video-walls/luma-pro-series/a380-octal-tech-specs
[qualcomm-x2]: https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-X2-Elite-Product-Brief.pdf
[linalg-support]: https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/
