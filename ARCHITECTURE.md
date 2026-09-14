# tinygrad DX12 architecture

This document describes `src/tinygrad_dx12` as reviewed on 2026-09-13.
It specifies the backend's structure, data formats, algorithms, and limits.
It does not claim that every path works on every adapter.

The implementation uses Python 3.14+, `ctypes`, tinygrad, and Direct3D 12 compute.
LinAlg is the matrix API in the packaged HLSL header `dx/linalg.h`.
DXIL is the compiled shader format. A PSO is a device-specific pipeline state object.
A fence is a counter that marks completed queue work.

| Term | Meaning in this document |
| --- | --- |
| UOp | A node in tinygrad's program representation |
| View | A byte range that shares an allocation with another buffer |
| SRV / UAV | A read-only / writable shader resource binding |
| CBV | A constant buffer binding for shader parameters |
| Root signature | The layout of the shader's resource and parameter bindings |
| Staging | A visible copy or conversion before the operation that needs it |
| Tail | Matrix elements that do not fill a complete tile |
| Epilogue | Ordinary tensor work after a matrix product |

## 1. System boundary and source map

The backend converts tinygrad programs to HLSL, compiles HLSL with DXC, and runs
DXIL on a compute queue. tinygrad owns tensor operations, gradients, scheduling,
memory planning, ordinary dtype decomposition, kernel optimization, and JIT capture.
A replacement backend must use those tinygrad contracts. It need not reimplement
tinygrad or the Microsoft APIs.
The Python interfaces are `DX12Device(Compiled)`, `DX12Allocator(Allocator)`,
`DX12Renderer(CStyleLanguage)`, `DX12Compiler(Compiler)`,
`DX12Program(Program)`, and `DX12Graph(GraphRunner)`.

The declared tinygrad dependency follows Git source without a fixed revision.
The current `uv.lock` resolves commit `e4570fcb46b3725d3eb9243507a507b4ae19131b`
(version `0.14.0`). Use that revision to reproduce the integration described here.
A later dependency update can require changes to the integration hooks.

All paths in this table are relative to `src/tinygrad_dx12/`.

| Module | Responsibility |
| --- | --- |
| `__init__.py` | Public registration and diagnostic command |
| `compat.py` | External device loader, device-lane splitting, scheduled matrix rewrite hooks, cached eager argument resolution |
| `setup.py` | Download, verify, and install the pinned native packages |
| `diagnostics.py` | DLL information, adapter features, and optional compile/PSO probes |
| `renderer/cstyle.py` | `DX12Renderer`, raw memory operations, shader metadata, and binding layout |
| `renderer/linalg.py` | LinAlg component map and collective/thread shader generation |
| `codegen/matmul.py` | Recognize and replace suitable scheduled matrix reductions |
| `codegen/linalg.py` | Matrix selection and explicit `matmul`/`matvec` APIs |
| `codegen/decomp/dx12.py` | `PackedTensor`, packed decoding, and intermediate dtype selection |
| `codegen/decomp/late.py` | Split late 64-bit integer operations into 32-bit words |
| `runtime/ops_dx12.py` | Device, buffers, allocator, commands, PSOs, eager calls, timing, and staging transfers |
| `runtime/graph/dx12.py` | Graph dependency barriers and three replay slots |
| `runtime/support/dx12.py` | COM ownership, DLL loading, adapter enumeration, and feature queries |
| `runtime/support/compiler_dx12.py` | DXC COM calls, compiler cache identity, and shader container |
| `runtime/support/transfer.py` | Shared-heap transfers between adapters |
| `runtime/autogen/{win32,dxgi,d3d12,dxc}.py` | The native declarations used by the backend |

The main execution path is:

```text
Tensor operations
  -> tinygrad schedule
  -> split DX12 device lanes
  -> replace eligible matrix reductions with LinAlg programs
  -> tinygrad memory plan and ordinary code generation
  -> HLSL + binding metadata
  -> DXC -> source/DXIL container
  -> DX12Program -> root signature + PSO
  -> eager command list, or DX12Graph replay slot
  -> compute queue -> fence -> release completed resources
```

## 2. Installation and native initialization

### Toolchain

The process must be 64-bit Windows Python. Windows Developer Mode must be enabled.
The setup code recognizes x64 and ARM64 PE machine types. It checks the available
DLLs against the Python executable; architecture recognition is not an execution test.

| Package | Version | SHA-256 of the NuGet archive |
| --- | --- | --- |
| `microsoft.direct3d.d3d12` | `1.721.3-preview` | `0131bce1e4bace3fc08c03018c29a09ede2570b263721c6449b0ec75762ab22d` |
| `microsoft.direct3d.dxc` | `1.10.2605.37-preview` | `8cf5ec1a8abb832de43251bae1ffcec89b22240677c599c38d30a18dee2f4de6` |

The Agility SDK number is `721`. The NuGet URL pattern is
`https://api.nuget.org/v3-flatcontainer/{name}/{version}/{name}.{version}.nupkg`.
A package is a ZIP archive. Setup checks its hash before extraction.

```powershell
uv sync
uv run python -m tinygrad_dx12.setup install
# Offline alternative; the directory must contain both original archive names:
uv run python -m tinygrad_dx12.setup install --archives C:\packages
```

The default cache is `%LOCALAPPDATA%/tinygrad_dx12`.
`TINYGRAD_DX12_CACHE` overrides it. If `LOCALAPPDATA` is absent, use
`~/.cache/tinygrad_dx12`. The toolchain subdirectory is
`agility-{agility_version}_dxc-{dxc_version}_{architecture}`.

Extract native DLLs/EXEs for the selected architecture, headers, and license files.
Reject absolute paths, parent traversal, colon-containing path parts, and symlinks.
Check each extracted DLL's PE signature and machine type. Required assets are:

```text
agility/build/native/bin/{arch}/D3D12Core.dll
dxc/build/native/bin/{arch}/dxcompiler.dll
dxc/build/native/bin/{arch}/dxil.dll
dxc/build/native/include/hlsl/dx/linalg.h
```

Write a manifest and rename the temporary installation into place. An existing
manifest makes installation return the existing directory. First use installs
missing files with the same checks. Later calls reuse them and do not replace
loaded DLLs. No Visual Studio, Windows SDK installation,
.NET, DirectML, or vendor compute library is part of the runtime.

### Native ABI and ownership

Use Windows calling conventions and explicit argument/result types for each COM
vtable call. Include inherited methods in slot numbers. `HRESULT`, `LONG`, and
`BOOL` are signed 32-bit values. DWORDs are unsigned 32-bit values. GPU addresses
and fences are unsigned 64-bit values. Pointers and `SIZE_T` are pointer-sized.
Use normal structure alignment and GUID bytes in Windows little-endian field order.

`Com` owns one returned COM reference. `query()` obtains another owned reference.
`close()` releases once and clears the pointer. Reject null or released objects.
Cache typed function pointers by address and signature, not by COM object.
Copy blob bytes while the blob is alive. Check HRESULTs as signed 32-bit values;
errors include the operation and hexadecimal result.

Structure-return methods use an explicit result pointer. This applies to resource
`GetDesc` (slot 10) and descriptor heap handle getters (slots 9 and 10).
Use the pinned package headers for structure and interface declarations.

### Factory and adapters

`NativeRuntime` is a process singleton. It owns loaded DLLs, DLL-directory handles,
the local device factory, and a stable list of hardware adapters. Its locks protect
initialization and device creation. Each device has a separate submission lock.

1. Check the installed manifest and DLL architectures.
2. Retain `os.add_dll_directory` handles for the Agility and DXC directories.
3. Load system `d3d12.dll`, `dxgi.dll`, and `kernel32.dll` with `WinDLL`, flag `0x800`.
4. Call `D3D12GetInterface` for `ID3D12SDKConfiguration1`.
5. Call configuration slot 4, `CreateDeviceFactory(721, sdk_directory, ...)`.
   Pass an absolute directory with a trailing separator, encoded as `mbcs`.
6. Call factory slot 8 to enable `D3D12ExperimentalShaderModels`.
7. Create `IDXGIFactory1`. Enumerate adapters with `EnumAdapters1`, slot 12.
   Read `GetDesc1`, slot 10, and omit software adapters. Keep enumeration order.
8. Create the selected device through factory slot 9 at feature level `0xB000`.

The bootstrap identifiers are:

```text
CLSID_D3D12SDKConfiguration  7cda6aca-a03e-49c8-9458-0334d20e07ce
IID_ID3D12SDKConfiguration1  8aaf9303-ad25-48b9-9a57-d9c37e009d9f
IID_ID3D12DeviceFactory      61f307d3-d34e-4e7c-8374-3ba4de23cccb
ExperimentalShaderModels    76f5573e-f13a-40f5-b297-81ce9e18933f
```

`DX12` and `DX12:0` identify adapter zero. Each nonzero ordinal selects its own
adapter. `count()` returns the hardware list length. An invalid ordinal raises.
The adapter record contains name, vendor ID, device ID, LUID, and dedicated memory.

`DX12_DEBUG=1` enables the factory's debug interface before the first device.
Enabling it after a device exists raises. Device creation requires Enhanced Barriers,
root signature 1.1, and a shader model from 6.0 through 6.10. Query shader models
from `0x6A` down to `0x60` until a query succeeds. Command list creation also
requires `ID3D12GraphicsCommandList7`.

Store these capabilities as sorted, compact JSON in `Target(device="DX12", arch=...)`:
`shader_model`, `native16`, `doubles`, `int64`, `wave_ops`, `wave_min`, `wave_max`,
and `enhanced_barriers`. Query OPTIONS, OPTIONS1, OPTIONS4, and OPTIONS12.
The capability JSON separates tinygrad program and compiler caches.
LinAlg queries are optional and occur when a matrix candidate needs them.

## 3. tinygrad integration

Import `tinygrad_dx12` before device lookup. Import calls `register()` automatically.
An explicit call to `tinygrad_dx12.register()` is also valid. Repeated calls have no effect.
It inserts `tinygrad_dx12.runtime.ops_dx12` as `tinygrad.runtime.ops_dx12` in
`sys.modules` and adds `DX12` to `Device._devices`. It also puts DX12 first in
tinygrad's `ALL_DEVICES` selection list. It does not open a GPU or download files.
Tinygrad tries DX12 when it next selects a default. If DX12 cannot open, tinygrad
tries the other runtimes. An explicit `DEV` setting takes precedence. A cached
default stays unchanged. Explicit tensor devices also stay unchanged and report
initialization errors without a fallback. No installed tinygrad files are edited.

```python
import tinygrad_dx12
tinygrad_dx12.register()

from tinygrad import Tensor
x = Tensor([1.0, 2.0], device="DX12")
result = (x * 2).tolist()
```

`compat.py` installs these in-memory hooks:

| Hook | DX12 action before calling or replacing the original path |
| --- | --- |
| Memory planning | Split device lanes, then lower eligible matrix calls so copies and temporaries enter the normal memory plan |
| `lower_and_compile` | Apply the same lane split and matrix lowering |
| `jit_lower` | Lower with the original input UOps, then call tinygrad with `Context(DX12_LINALG=0)` to prevent a second matrix rewrite |
| `full_rewrite_to_sink` | Widen BF16/FP8 MAX reduction inputs to FP32 and cast the result back |
| `exec_kernel` | Cache DX12 runtime and resolved buffer/view templates for eligible cached single-device calls |

For a tuple made only of DX12 devices, expand each kernel into one call per lane.
Select each tuple buffer with `MSELECT`. Replace device ranges and `_device_num`
with the original lane index, which can differ from the adapter ordinal. Replace
tuple parameter devices with the selected full device name. Rebuild prior PROGRAM
metadata from its retained sink so another adapter's binary cannot be reused.
Preserve copy order and leave other calls unchanged.

Eager resolution uses weak call keys and weak input references. Resolve all views
again when input identities change. On every launch, call `ensure_allocated()` so
storage replacement under an unchanged input is detected. Use the same full-view
resolver for graphs, including `MSELECT(PARAM)`. Defer other execution cases to tinygrad.

## 4. Supported data types and scalar arithmetic

A logical dtype describes tensor storage. Native support describes shader arithmetic.
LinAlg support describes one matrix operation. These are separate decisions.
The allocator stores bytes and has no dtype conversion logic.

| tinygrad dtype | Bytes per element | Ordinary HLSL path | Direct LinAlg component |
| --- | ---: | --- | --- |
| `bool` | 1 | Boolean expressions; load byte != 0; store 0 or 1 | None |
| `int8` | 1 | `int`, with sign extension and truncation after narrow operations | `I8` |
| `uint8` | 1 | `uint`, masked to 8 bits | `U8` |
| `int16` | 2 | `int16_t` with native16; otherwise truncated `int` | `I16` |
| `uint16` | 2 | `uint16_t` with native16; otherwise masked `uint` | `U16` |
| `int32` | 4 | `int` | `I32` |
| `uint32` | 4 | `uint` | `U32` |
| `int64` | 8 | `int64_t` with int64 support; otherwise tinygrad word decomposition | None |
| `uint64` | 8 | `uint64_t` with int64 support; otherwise tinygrad word decomposition | None |
| `float16` (`half`) | 2 | `float16_t` with native16; otherwise tinygrad FP32 decomposition | `F16` |
| `bfloat16` | 2 | tinygrad BF16 storage conversion and FP32 arithmetic | None |
| `float32` (`float`) | 4 | `float` | `F32` |
| `float64` (`double`) | 8 | `double`; requires device double support | None |
| `fp8e4m3` (E4M3FN) | 1 | tinygrad FP8 conversion; FP16 arithmetic with native16, otherwise FP32 | `F8_E4M3FN` |
| `fp8e5m2` (E5M2) | 1 | Same scalar conversion policy as E4M3FN | `F8_E5M2` |
| `fp8e4m3fnuz` (E4M3FNUZ) | 1 | tinygrad FP8 conversion; FP16 or FP32 arithmetic | None |
| `fp8e5m2fnuz` (E5M2FNUZ) | 1 | tinygrad FP8 conversion; FP16 or FP32 arithmetic | None |

These are 17 scalar storage types, with capability limits as shown.
A LinAlg component in the table is a candidate, not a support guarantee.
BF16 is absent from the implementation's pinned LinAlg mapping.
There is no backend FP64 arithmetic emulation when double support is absent.
There are no native tinygrad INT4, UINT4, FP4, or FP6 dtypes; use `PackedTensor`
from section 7. Complex and image/texture tensor paths are not implemented here.

`DX12Renderer.supported_dtypes()` removes all FP8 types and BF16 from native
arithmetic support. It also removes FP16, INT64/UINT64, and FP64 when their required
features are absent. tinygrad performs the applicable shared decompositions.
Narrow signed integers use left/right shifts to retain their sign and width;
unsigned integers use a width mask. Apply that conversion after narrow ALU operations,
not only at stores.

The renderer inherits tinygrad's C-style operations and rewrites. It uses scalar
memory lowering (`supports_float4=False`) but can render explicit HLSL vectors.
Shared arrays use `groupshared`; register arrays stay inside `main`.
A workgroup barrier is `GroupMemoryBarrierWithGroupSync()`.

Use typed raw loads/stores at natural element alignment for 32/64-bit values and
native 16-bit values. Other accesses use DWORD extraction or masked writes:

- Load the containing DWORD. Load the next DWORD only if the element crosses it.
- Reassemble low/high words for 64-bit values. Use HLSL bitcast intrinsics for floats.
- For a partial DWORD store, use an `InterlockedCompareExchange` loop with a byte mask.
  Repeat until the old value matches. Split a crossing store across two DWORDs.
- Use a direct DWORD write when the operation owns all four bytes.
- Combine four consecutive byte stores only within a straight sequence of stores
  and indexes to the same nonvolatile global parameter. Do not cross a load or control flow.
- Preserve tinygrad's load predicates and their default values.

For floating-point math, set `strict_math` when the rendered UOps have no RANGE,
or when they contain an ALU BITCAST whose result is floating-point. Strict shaders
use DXC `-Gis` and render floating multiply-add as `mad`, or `fma` for FP64.
Other shaders retain ordinary multiply/add expressions. All shaders use
`-fno-finite-math-only`. This rule is stored in source metadata and survives specialization.

Render floating self-comparisons with `isnan`. Use tinygrad expansions for sine
and FP64 EXP2, LOG2, and SQRT; FP64 SQRT uses the shared power expansion.
FP64 truncation clears mantissa bits through two DWORDs. If transcendental lowering
introduces integer64 values on a device without int64 operations, run the late
word decomposition. Reuse tinygrad `l2i` arithmetic and add double/word bitcast
boundaries. This adds no buffers or dispatches.

Reduction precision follows tinygrad. The matrix default is
`strong_dtype(sum_acc_dtype(strong_dtype(least_upper_dtype(a, b))))`.
Normally this gives FP32 for low-precision floats, INT32 for narrow signed integers
and BOOL, and UINT32 for narrow unsigned integers. It retains FP64 and 64-bit integers.
The shared `SUM_DTYPE` setting can change the floating reduction default.
LinAlg owns its internal matrix arithmetic and rounding.

## 5. Shader format, bindings, and compilation

### Binding metadata

Every shader starts with `// DX12:` followed by sorted, compact JSON.
The JSON represents `BindingLayout`:

| Record | Fields |
| --- | --- |
| Buffer | `slot`, `name`, `writable`, `register`, `root_index`, `offset_word`, `views` (default 1) |
| Scalar | `slot`, `name`, `dtype`, `word`, `words` |
| Layout | `buffers`, `scalars`, `scalar_words`, `scalar_root`, `scalar_cbv`, `table`, `local_size`, `alignable`, `has_fp64`, `strict_math` |

Sort parameters by tinygrad slot. Each buffer takes one offset DWORD at the start
of the scalar block. Then append scalar values in slot order. A scalar takes
`max(1, itemsize // 4)` words. Encode integers modulo their word width, little-endian.
Encode FP32/FP64 with IEEE `<f`/`<d`. Put low words first. This layout does not add
padding between scalar values.

Use `ByteAddressBuffer` for read-only arguments and `RWByteAddressBuffer` for
writable arguments. Assign `t` and `u` registers with separate counters. Advance a
counter by the buffer's view count. Declare the scalar block at `b0` as individual
`uint _pN` fields: register `c[N // 4]`, component `xyzw[N % 4]`, using `packoffset`.
Define each `_offset_slot` as its assigned `_pN`. Scalar macros reconstruct the
requested value from those words.

Let `B` be the buffer count and `W` the scalar word count, including offsets:

1. Set `scalar_cbv = (2*B + W > 64)`.
2. Set `table = (2*B + (2 if scalar_cbv else W) > 64)`, or true if any buffer has multiple views.
3. Root buffers each cost two DWORDs. A table costs one. A CBV costs two.
4. Root buffer indices follow buffer order. A table uses root index zero.
   The scalar root is `B` for root buffers or 1 for a table.
5. Round CBV storage up to 256 bytes. Bind scalar constants directly when a CBV is not needed.

Serialize root signature version 1.1, with no static samplers and no extra flags.
Root data uses `DATA_VOLATILE=2`. Table ranges use
`DESCRIPTORS_VOLATILE | DATA_VOLATILE=3` and append offsets.
Use a shader-visible CBV/SRV/UAV descriptor heap. Raw descriptors have
`R32_TYPELESS=39`, RAW flag 1, and DWORD element units; SRVs use component mapping 5768.

### Addressing and large buffers

Align a raw binding base down to 16 bytes. Pass the remaining offset, `offset & 15`,
in the buffer's offset DWORD. Add index times element size in the shader.
This supports byte views without changing the tensor's logical layout.

Set `WINDOW = 1 << 30` bytes. For a parameter with maximum byte span `S`, use:

```text
declared_span = S + 15
views = 1                              if declared_span <= 2*WINDOW - 4
views = ceil(declared_span / WINDOW)    otherwise
```

A large parameter uses an array of raw descriptors. Compute logical byte addresses
with `uint64_t`. Select descriptor `byte_address >> 30`; its byte offset is
`byte_address & 0x3fffffff`. Scalar code marks the descriptor index with
`NonUniformResourceIndex`. Collective matrix addresses are group-uniform.

Descriptor `j` starts at `(view.offset & ~15) + j*WINDOW` in the allocation.
Its requested length is
`max(0, min(2*WINDOW-4, view.size + (view.offset & 15) - j*WINDOW))`, rounded to four bytes.
An unused descriptor is a null resource with four DWORDs. Neighboring descriptors
overlap so one vector or matrix tile can cross a 1 GiB boundary.

Large allocations use one heap. Cache placed resource windows over the same heap
at `start = requested_offset & ~65535`. A placed window has width
`min(base_resource_width-start, 2*WINDOW+65536)` and retains the original state.
Return its GPU address and the residual resource offset. These resource windows
serve raw descriptors, root views, and copies. They do not copy tensor data.
There is no backend 4 GiB allocation cap or adapter-memory-size policy.
Native allocation and shader acceptance still apply; the wide shader path uses integer64 addresses.

### DXC and PSO specialization

Load the private `dxcompiler.dll` by absolute path, flag `0x1100`, with its matching
`dxil.dll` beside it. Create `IDxcUtils`, `IDxcCompiler3`, and the default include
handler. Each compiler has a lock and retains its interfaces.

Pass UTF-8 source through `DxcBuffer` and UTF-16 arguments:

```text
-T cs_{major}_{minor} -E main -HV 2021 -O3
-fno-finite-math-only -res_may_alias
-I <toolchain>/dxc/build/native/include/hlsl
```

Add `-enable-16bit-types` when `native16` is true. Add `-Gis` for strict metadata.
Check the COM call HRESULT and the result status. Retain compiler diagnostics;
copy output bytes before releasing the result. `DX12_DUMP_SHADERS=1` prints each
source passed to compilation.

The compiler cache name is `dx12_` plus SHA-256 of the `repr` of this tuple:
`(DXC package version, compiler settings, capability JSON, container magic)`.
`compile_cached()` also keys on the complete source, including its metadata.

`TinyELF.lib` carries this backend container; it is not an ELF file:

```text
<8sI header: magic b"TGDX12\x01\0", UTF-8 source byte count
source bytes
DXIL bytes
```

Reject a short header, wrong magic, or a source length that leaves no DXIL.
`DX12Program(dev, obj)` reads the container and creates its own root signature and PSOs.
Cache PSOs by `(local_size, aligned and layout.alignable)` on that program's device.
Reuse the original DXIL for its original local size without alignment specialization.
Otherwise update the `DX12_LOCAL_X/Y/Z` macros and compile through the cache.
If all view offsets are 16-byte aligned, replace only `_offset_slot` macros with `0u`.
DXC then decides which individual accesses are aligned.

`Dispatch` takes group counts, not total thread counts. HLSL uses `SV_GroupID` and
`SV_GroupThreadID`. `[numthreads]` is fixed by the local-size macros. Require three
positive local dimensions, at most 1,024 total threads, and Z <= 64. Group dimensions
must be 0 through 65,535. The renderer advertises 32 KiB of shared memory.
Changing a local size requires a PSO specialization; changing groups does not.

## 6. LinAlg selection and matrix execution

### Exact operation queries

Map the HLSL names in section 4 to these D3D12 datatype values:

| HLSL component | D3D12 query value |
| --- | ---: |
| `I16`, `U16` | 2, 3 |
| `I32`, `U32` | 4, 5 |
| `F16`, `F32` | 7, 8 |
| `I8`, `U8` | 18, 19 |
| `F8_E4M3FN`, `F8_E5M2` | 20, 21 |

These values are not HLSL enum values. Use the named HLSL components in shaders.
Call `CheckFeatureSupport` with feature 78 and an operation-tagged union:

- Operation 1, Wave: wave size, A type, B type, and accumulator type.
  The result gives support flags and shapes in `(M,K,N)` order. Query once for
  the shape count, allocate the shape array, then query again to fill it.
- Operation 2, ThreadGroup: the same inputs plus one `(M,K,N)` shape.
  The result gives flags and minimum, maximum, and preferred group thread counts.
- Operation 3, Thread: vector, matrix, bias, and result types; the result gives flags.

Bit 0 means supported. Cache collective queries per renderer by
`(A, B, accumulator, scope, tile, wave_min)`. Cache a failed optional query as `None`.
A missing datatype mapping produces zero support. Cache matrix choices by
`(A, B, accumulator, M, N, K, requested_tile, requested_scope)`.

Selection starts with the supplied tile, or `(16,16,16)`. The public tile order is
`(M,N,K)`. Query Wave support first. Without an explicit tile, choose the first
reported shape that divides all matrix dimensions; otherwise choose the first
reported shape. Convert its order to `(M,N,K)`.

For `auto`, try Wave, then ThreadGroup. Explicit collective scopes try only the
requested scope, although the initial Wave query still supplies default shapes.
Wave requires the chosen shape in the reported list and uses `wave_min` threads.
ThreadGroup uses preferred threads, then minimum threads, then 32 if neither is set.
If neither candidate is supported, return the ordinary-compute choice `tiled`.
There is no benchmark search, vendor ranking, startup dtype sweep, or native-instruction check.

After selection, compile the complete source and create its PSO before accepting it.
Cache rejection text by exact source in the renderer. A rejected candidate does not
remove a whole dtype. Output conversion acceptance is checked through the complete
shader and PSO, not inferred from the accumulator query.

### Automatic scheduled rewrite

`DX12_LINALG` is a tinygrad `ContextVar`, default 1. It controls the scheduled rewrite.
The matcher accepts a single output store over two output axes and one positive,
constant reduction axis. The reduction must be ADD over a product of two direct
parameter indexes, with optional casts to the accumulator. Require these layouts:

```text
C[m,n] -> m*N + n
A[m,k] -> m*K + k   or   k*M + m
B[k,n] -> k*N + n   or   n*K + k
```

The first choice is row-major; the second is column-major. Allow the same read
parameter on both sides. Require sufficient declared buffer extents and a distinct
output parameter. Reject explicit optimization options, additional outputs,
broadcast/data-dependent indexes, symbolic sizes, unsupported component types,
and patterns outside these rules.

Require 16-byte A, B, and output row strides. Require complete selected tiles and
group counts within renderer limits. Prove input start alignment from view metadata
at 128 bytes. Insert an ordinary copy for an input whose alignment is not proved,
or which can be rebound by JIT. With no epilogue, require output alignment too.
The automatic rewrite does not add tail padding; unmatched calls use normal tinygrad lowering.

Use the scheduled accumulator unchanged. If output storage differs, let LinAlg
cast before the matrix store. If an expression follows the reduction, create an
accumulator-typed temporary and replace the reduction with a load from it in the
original expression. This separate ordinary kernel is the epilogue. Copies and
temporaries enter tinygrad's memory plan. This rewrite occurs after autograd and
preserves the normal tensor gradient path.

### Explicit matrix APIs

```python
from tinygrad_dx12.codegen.linalg import matmul, matvec

# Rank-two A[M,K] and B[K,N] on the same device.
result = matmul(a, b, scope="auto", tile=None, dtype=None)
# Matrix[M,K] and vector[K] on the same device.
result = matvec(matrix, vector, output_dtype=None, thread=None)
```

`matmul` accepts Tensor or PackedTensor operands. Reject incompatible devices,
ranks, or inner dimensions. Decode packed operands with a common intermediate;
when either operand is packed, cast both to that intermediate. Ordinary tensor
operands retain their own storage types. `dtype` sets accumulator and output;
otherwise use the shared reduction default.

Reject unknown scopes. Supplied tile dimensions must be positive on the collective
DX12 path. The shader generator also requires complete, positive tile dimensions
and valid component mappings. Keep these errors separate from ordinary dtype support.

`auto` returns ordinary tinygrad matmul if selection, compilation, or PSO creation
fails. Explicit `Wave`/`ThreadGroup` raise on rejected queries or candidates.
Non-DX12, zero-size, and symbolic-size paths delegate to ordinary tensor operations.
`scope="tiled"` casts operands to the output dtype and builds ordinary tinygrad
matmul. It does not itself disable the later scheduled rewrite; use
`Context(DX12_LINALG=0)` when that rewrite must be disabled.

The explicit collective path can pad tails. For selected tile `(tm,tn,tk)`:

```text
Mpad = round_up(M, tm)
Kpad = round_up(K, lcm(tk, 16 / A.itemsize))
Npad = round_up(N, lcm(tn, 16 / B.itemsize))
```

Pad A to `(Mpad,Kpad)` and B to `(Kpad,Npad)`, then make them contiguous.
Without padding, clone known misaligned or rebindable inputs; otherwise use
`contiguous()`. Allocate output `(Mpad,Npad)`. Return its `[:M,:N]` slice.
All decoding, copies, padding, and slicing remain visible tensor operations.

Create one PROGRAM with output slot 0 and input slots 1 and 2, globals `(0,1,2)`,
outs `(0,)`, ins `(1,2)`, the device renderer target, and explicit launch sizes.
Its source is a normal SOURCE UOp. Call tinygrad `to_program()` and `get_runtime()`
to compile and check the PSO. The runtime has no hidden matrix dispatch.
These explicit custom matrix kernels do not define gradients.

### Collective shader

Include `<dx/linalg.h>` and use `dx::linalg`. Define matrices:

```text
MA = Matrix<A_component, tm, tk, MatrixUse::A, scope>
MB = Matrix<B_component, tk, tn, MatrixUse::B, scope>
MC = Matrix<acc_component, tm, tn, MatrixUse::Accumulator, scope>
```

Dispatch `(N/tn, M/tm, 1)` groups. Each group writes a distinct output tile.
Wave shaders set `[WaveSize(wave_min)]` and use one wave per group.
ThreadGroup shaders use the selected group thread count.

Initialize `acc = MC::Splat(0)`. For `kk` from 0 to K in steps of `tk`, load A and B
with byte strides and alignment argument 16, then call `acc.MultiplyAccumulate(a,b)`.
The element starts are:

| Buffer | Row-major start | Column-major start |
| --- | --- | --- |
| A | `gid.y*tm*K + kk` | `kk*M + gid.y*tm` |
| B | `kk*N + gid.x*tn` | `gid.x*tn*K + kk` |
| C | `gid.y*tm*N + gid.x*tn` | Not used |

Multiply these starts by item size and add the binding offset.
A strides are `K*itemsize` or `M*itemsize`; B strides are `N*itemsize` or
`K*itemsize`. Store C row-major with stride `N*C.itemsize`. If needed, call
`acc.Cast<ComponentType::output>()` before Store. For multi-view buffers, reject
a tile whose full byte span exceeds `WINDOW`; otherwise use the shared window addressing.

### Matrix-vector behavior

`thread=None` uses `matmul(matrix, vector.reshape(-1,1))`, then reshapes the result.
`thread=False` builds ordinary `dot`. `thread=True` tries the separate Thread API.
It still falls back to ordinary dot on unsupported types, explicit zero support,
source/PSO rejection, or unsuitable dimensions. If its optional query raises,
it tries source/PSO acceptance instead.

The Thread result type must be FP16, FP32, INT16, UINT16, INT32, or UINT32.
Query vector/matrix/bias/result types with bias and result equal to the requested
output. Pad K to a multiple of `16 / matrix.itemsize`. Stage matrix and vector
as needed. Reject multiple raw views in this shader.

Run one group with one thread. Load a Thread-scope `Matrix<A_component,M,K>` using
`Load<MatrixLayout::RowMajor>(A, offset, K*itemsize, 128)`. For one-byte vector
components, load `K/4` DWORDs into `vector<uint8_t4_packed,...>` and use
`MakeInterpretedVector<ComponentType::...>`. Other components use typed loads.
Call `MultiplyAdd<output_scalar>(matrix, vector, zero_bias)` and store the M outputs.

## 7. Packed low-bit formats

`PackedTensor(data, format, shape, scales=None, block_size=32)` is a frozen metadata
record. `data` must be a flat UINT8 Tensor. Payloads are consecutive, least-significant
first, across bytes. Storage must contain at least `ceil(product(shape)*bits/8)` bytes.
Require a nonempty shape with nonnegative dimensions and positive block size.

| Format name | Bits | Exponent bits | Fraction bits | Exponent bias |
| --- | ---: | ---: | ---: | ---: |
| `int4` | 4 | 0 | 0 | 0 |
| `uint4` | 4 | 0 | 0 | 0 |
| `fp4_e2m1` | 4 | 2 | 1 | 1 |
| `fp6_e2m3` | 6 | 2 | 3 | 1 |
| `fp6_e3m2` | 6 | 3 | 2 | 3 |

These FP4/FP6 encodings use finite OCP payloads. Scales, when supplied, are already
numeric tensors on the same device. Their shape must be
`shape[:-1] + (ceil(shape[-1]/block_size),)`. The backend does not decode raw E8M0
scale bytes. A caller can supply scales that were decoded from that format.

Decode four elements at a time from two bytes for 4-bit formats or three bytes for
6-bit formats. Pad the byte count to the group width. Use strided byte views to
form a UINT32 word. Extract element `i` with `(word >> (i*bits)) & ((1<<bits)-1)`.
Flatten, truncate to the logical count, and reshape.
The word is `byte0 | (byte1 << 8)`, with `byte2 << 16` added for FP6.

For INT4, decode `(payload ^ 8) - 8`; UINT4 is unsigned. For floating payloads,
extract sign `s`, exponent `e`, and fraction `f`. With fraction width `p` and bias `b`:

```text
significand = f              if e == 0 else f + 2**p
value = (-1)**s * significand * 2**((1 if e == 0 else e) - b - p)
```

All exponent encodings in these source formats are finite. Broadcast each scale
across its block along the last dimension and truncate the final partial block.
Multiply in `least_upper_dtype(float32, scale.dtype, requested_dtype)` before the
final cast. Empty shapes produce an empty output.

`unpack()` defaults to INT32 for unscaled integers, FP32 for unscaled floats, and
at least FP32 for scaled data, retaining FP64 scales. An explicit dtype requests
normal conversion. Unscaled exact FP8 destinations use a direct bit encoding:

| Source | Exact FP8 choices |
| --- | --- |
| FP4 E2M1 | E4M3FN, E5M2 |
| FP6 E2M3 | E4M3FN |
| FP6 E3M2 | E4M3FN, E5M2 |

For a normal value, adjust the exponent by `target_bias-source_bias` and shift the
fraction to the target width. For a nonzero subnormal fraction, let `h` be its
highest set bit: target exponent is `target_bias+1-source_bias-source_fraction_bits+h`;
shift and mask the fraction to remove its leading bit. Preserve the sign at bit 7
and zero when the fraction is zero. Cast the encoded UINT8 to FP8 by bitcast.
Do not reinterpret the original low-bit payload as FP8.
The target fraction width and bias are `(3,7)` for E4M3FN and `(2,15)` for E5M2.

For matrix intermediate selection, intersect the exact unscaled FP8 choices of
packed operands with ordinary operands' declared dtypes. Prefer E4M3FN, then E5M2.
If no common choice exists, apply tinygrad promotion to these types:
ordinary operand dtype; INT8/UINT8 for unscaled INT4/UINT4; FP16 for unscaled
FP4/FP6; at least FP32 for scaled data. This preserves higher-precision activations.
The `native16` argument retained by `matrix_intermediate` does not change this rule.
`PackedTensor.matmul()` uses the explicit matrix API. The separate decomp-module
`matmul()` helper unpacks with default dtypes and delegates to ordinary Tensor matmul.

## 8. Buffers, submission, and transfers

### Allocation and lifetime

`DX12Buffer` records resource, base GPU address, logical byte size, device, byte offset,
base owner, optional mapped memory, and last-use fences by device. Large owners also
hold a heap and cached resource windows. A view retains its original owner.
Reject negative or out-of-bounds view ranges.

Allocate physical bytes as `round_up(max(logical_size,4)+4,256)`. The extra bytes
permit raw DWORD access at a logical tail. DEFAULT buffers allow UAV access and
start in COMMON. UPLOAD buffers start in GENERIC_READ (`0xAC3`); READBACK buffers
start in COPY_DEST (`0x400`). Map staging buffers persistently. The upload read
range is empty; the readback range covers physical storage.

For sizes above `2*WINDOW-4`, create a heap rounded to 64 KiB and a full placed
buffer. Use buffer-only heap flags `0xC0`. Smaller allocations use committed resources.
Copies split into chunks of at most `WINDOW` through the same resource windows.

The allocator supports transfers, but not disk copies or arbitrary `external_ptr`
imports. `BufferSpec.zero` uploads zero-filled data. Freed allocations enter a deferred
list. Return them to tinygrad's allocation cache only when every recorded device
use has completed. `free_cache()` collects completed frees without waiting for active work.

### Device and eager calls

Each `DX12Device` owns one compute queue, a fence counter, its renderer/compiler,
command pools, pending lists, in-flight owners/callbacks, transfer records, and failure state.
A command owns one compute allocator and one command list. Reset both only after
its fence completes. No process-global submission lock is used.

The program call contract is:

```text
program(*buffers, global_size=(1,1,1), local_size=(1,1,1),
        vals=(), wait=False, timeout=None) -> float | None
```

Check group dimensions, buffer count/device, and scalar count. Zero groups skip
eager work and return `0.0` when timed, otherwise `None`. Local dimensions select
the specialized PSO. `timeout` is in milliseconds.

Cache eager recordings by local size, each resource/offset/size, scalar bytes,
group counts, and whether timing is required. Reuse only a completed, nonbusy
recording with the same key. Otherwise reuse an idle pair or allocate another.
There is no fixed eager recording count.

Record a global compute-to-compute barrier first, with UAV|SRV access on both sides.
Then set root signature, PSO, buffer bindings, scalar constants or CBV, and Dispatch.
Record timestamps only for `wait` or profiling. Close the list.

Submission records the command and all owners before queue execution. Mark each
buffer owner's last use. Untimed eager calls batch up to 16 lists under one fence
value. Each list's leading barrier orders dependencies inside the batch.
Timed calls, profiling, copies, graphs, waits, and synchronization flush pending work.
A flush executes the lists and signals the device fence. Keep owners alive through
that fence, including on submission failure.

A timed caller keeps its command busy until it has read timestamps, even if the GPU
has completed. Return GPU seconds when `wait=True`; otherwise return `None`.
Retirement runs completed profile callbacks and releases completed submission owners.

A wait flushes pending work for its fence point, then waits without the device lock.
Create an auto-reset Win32 event for the wait and close it afterward. Check completion
at intervals of at most 100 ms. Use the supplied timeout or tinygrad's default
30,000 ms. An all-ones completed fence means device removal. Store failures and
report them on subsequent completion/submission calls. `synchronize()` waits for
this device's fence and its transfer obligations, then retires work.

### Copies

Copyin first copies caller bytes into owned UPLOAD storage, then queues a GPU copy.
It can return before GPU completion. Copyout queues a READBACK copy, waits for its
fence, then copies bytes into the caller's memory. Each copy uses its own execution
boundary. Same-device copies skip identical views and use a temporary for overlapping
ranges of the same resource. Check transfer sizes and skip zero-size transfers.

For different adapters, acquire both device locks in increasing `id(device)` order.
Try a shared transfer unless `DX12_FORCE_STAGING=1` or that ordered pair rejected sharing:

1. Allocate a source DEFAULT heap rounded to 64 KiB with SHARED|SHARED_CROSS_ADAPTER (`0x21`).
2. Open it on the destination and create matching placed, cross-adapter buffers.
3. Create a shared cross-adapter fence (flags 3) and open it on the destination.
   Close temporary NT handles after opening the objects.
4. Queue source-to-shared copy and signal readiness 1 on the source queue.
5. Queue a destination wait for readiness 1, then shared-to-destination copy.
6. Retain both heap/resource views, fences, buffers, and transfer until both queue fences complete.

A resource-creation failure before submission caches rejection for that ordered
pair and uses staging. A failure after submission marks both devices failed;
it does not retry the copy through another path.

Staging allocates source READBACK and destination UPLOAD storage plus a destination
readiness fence. Queue both copy lists and the destination wait before returning.
Start an independent worker for that transfer. It waits for source completion,
copies mapped bytes on the CPU, then signals readiness 1. Later destination work
is already behind the wait. On failure, record the error on both devices and still
attempt to signal readiness so the queue is not left waiting forever.
Track both completion points in both owners' use maps. Do not recycle storage
until both sides complete. Transfers do not require caller synchronization before
an immediate destination consumer.

## 9. Graph capture and replay

`DX12Graph` subclasses tinygrad `GraphRunner` and calls its initializer. Require
only DX12 program calls on one actual device. Restore that full device identifier
after base initialization. Copies and unsupported operations remain outside the graph.
Use tinygrad's existing input restrictions, dependency analysis, memory plan,
`updated_vars()`, and `updated_launch_dims()`.

Construct graph-specific `DX12Program` objects from the existing shader containers.
Use a scalar CBV whenever scalar words exist. Use descriptor tables for non-FP64
buffer shaders. FP64 retains root buffer descriptors unless its original layout
already required a table. All layouts keep the shader's original registers and DXIL.

Build three slots. Each slot owns an allocator/list pair, mapped UPLOAD arguments,
per-kernel descriptor heaps, optional timestamp storage, resolved buffers, launch
sizes, alignment flags, and its last fence. Give each kernel a 256-byte-aligned
scalar region. Slots share the planned GPU temporaries; they do not duplicate them.

For dependencies inside the graph, use base `_access_resources` with each program's
output slots. Classify buffer access by binding: UAV for writable bindings, SRV for
read-only bindings. A writable buffer's reads still use UAV access.

Before a consumer, collect unresolved producer access masks and emit one global
compute-to-compute barrier. Track which destination access bits each prior producer
has already reached. A prior UAV-to-SRV barrier does not satisfy a later UAV consumer.
Because a global barrier covers earlier matching source accesses, mark those
producers covered too. Separate queue execution calls provide boundaries between
copies, eager batches, and graph submissions.

Record each graph kernel as barrier if needed, root signature, PSO, table/root
buffers, scalar CBV, and direct Dispatch. Close the list. Graph replay does not
use ExecuteIndirect or Work Graphs.

Hold the graph lock across replay. Select a completed slot in ring order. If all
slots are active, wait for the oldest fence outside the device lock. Then:

1. Resolve full input/view templates again if input identities changed. Keep weak
   input references. Check current storage even when input identity is unchanged.
2. Patch slot buffer references and residual offset words. Reject foreign-device buffers.
3. Patch changed scalar values with the precomputed little-endian formats.
4. Apply updated group/local sizes and update changed descriptor tables.
5. Record the idle list again if groups, local sizes, alignment, timing mode, or
   any root-buffer binding changed. Descriptor-only and scalar-only changes can reuse it.
6. Submit once, retaining the slot, programs, and current buffers. Record its fence.
7. Wait and return GPU seconds only when requested.

Never patch active arguments, reset an active allocator, or resubmit an active list.
Submission ownership keeps native work valid if the graph or tensor references disappear.

### Timing and optional indirect support

Use the compute queue timestamp frequency. Timestamp queries resolve to mapped
READBACK storage and are read only after completion. Untimed/unprofiled graphs use
zero queries. Timed graphs use two endpoint queries. Profiling uses `kernel_count+1`
queries and tinygrad `ProfileGraphEvent` records. Eager profiling uses `ProfileRangeEvent`.
Convert ticks to seconds for calls and to microseconds for profile events.
Changing the query count records an idle slot again; stable timing modes reuse lists.

`BindingLayout.pack_record()` and `DX12Program.indirect_signature()` remain available
for explicit callers and diagnostics. They are not the replay mechanism. Their ABI is:

```text
root buffer addresses: B little-endian uint64 values (omitted for a table)
scalar area: one uint64 CBV address, or W DWORD constants
Dispatch: three uint32 group counts
record stride: round total bytes up to 8
```

Thus scalar offset is `0` for a table or `8*B` otherwise. Dispatch offset adds
8 bytes for a CBV or `4*W` for constants. Validate 16-byte raw addresses and a
nonzero, 256-byte-aligned CBV address. The command signature uses matching root
SRV/UAV arguments, CBV/constants, then Dispatch. Tables must already be bound.

## 10. Diagnostics and implementation checks

```powershell
uv run python -m tinygrad_dx12.diagnostics
uv run python -m tinygrad_dx12.diagnostics --adapter 0 --probe
uv run python -m tinygrad_dx12.diagnostics --debug
```

Diagnostics emit JSON with package versions, DLL paths and file versions, adapter
identity, capabilities, and optional LinAlg tier (feature 77). `--probe` reports
Wave/ThreadGroup query, compiler, and PSO results separately for mapped dtypes plus
BF16/FP64 rejection cases. These probes do not dispatch work or measure speed.

| Setting | Effect |
| --- | --- |
| `TINYGRAD_DX12_CACHE` | Toolchain cache location |
| `DX12_DEBUG=1` | Debug layer before device creation |
| `DX12_DUMP_SHADERS=1` | Print compiled source |
| `DX12_FORCE_STAGING=1` | Bypass shared transfers |
| `DX12_LINALG=0` | Disable scheduled automatic matrix replacement |
| tinygrad `PROFILE` | GPU profile collection |
| `DX12_TEST_GPU=1` | Enable the repository's opt-in hardware tests |

The existing tests cover renderer/dtype paths, packed decoding, matrix matching,
late integer lowering, COM bindings, toolchain extraction, views, large addresses,
transfers, graphs, and concurrency. Hardware-dependent cases need a suitable adapter;
no static support table proves their execution. When a check run is requested, use
`uv run pytest -q -n12`; hardware cases also need `DX12_TEST_GPU=1`.
The configured source checks are `uv run ruff check .` and `uv run pyrefly check`.
Benchmark scripts are `scripts/benchmark_dx12.py` and `scripts/benchmark_dtypes.py`.
Include decode, padding, staging, and epilogue work when comparing matrix operation cost.

This revision records source behavior. It adds no new GPU validation or performance claim.
