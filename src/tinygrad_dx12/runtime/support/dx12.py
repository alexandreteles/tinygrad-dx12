"""Owned COM pointers, pinned DLL loading, adapter enumeration and device bootstrap."""

import ctypes as C
import functools, json, os, threading
from dataclasses import asdict, dataclass
from collections.abc import Callable
from tinygrad_dx12.runtime.autogen.win32 import GUID, HRESULT, UINT, WINFUNCTYPE
from tinygrad_dx12.runtime.autogen import d3d12 as d3d, dxgi


def check(result: int, operation: str) -> int:
    """Return a successful HRESULT, otherwise raise with the operation and hexadecimal code."""
    if C.c_int32(result).value < 0:
        raise RuntimeError(f"{operation} failed: HRESULT 0x{result & 0xFFFFFFFF:08X}")
    return result


type ReturnType = type[C.c_int32] | type[C.c_uint32] | type[C.c_uint64] | type[C.c_void_p] | type[C.c_size_t] | None


@functools.cache
def _method(address: int, restype: ReturnType, argtypes: tuple) -> Callable[..., int | None]:
    """Cache an explicitly typed COM function pointer without retaining a COM object."""
    return WINFUNCTYPE(restype, C.c_void_p, *argtypes)(address)


class Com:
    """One owned COM reference; callers retain resources through their final native use."""

    def __init__(self, pointer: C.c_void_p | int) -> None:
        """Take ownership of a non-null reference returned by a COM creation/query call."""
        self.ptr = pointer if isinstance(pointer, C.c_void_p) else C.c_void_p(pointer)
        if not self.ptr:
            raise ValueError("Null COM interface")
        self._vtable = C.cast(self.ptr, C.POINTER(C.POINTER(C.c_void_p))).contents

    def call(self, slot: int, restype: ReturnType, argtypes: tuple, *args: object) -> int:
        """Invoke an explicitly typed integer/pointer/void method, normalizing void or NULL to zero."""
        if not self.ptr:
            raise RuntimeError("COM object already released")
        address = self._vtable[slot]
        return _method(address, restype, argtypes)(self.ptr, *args) or 0

    def query(self, iid: GUID) -> Com:
        """Query another owned interface, raising on an unsupported interface."""
        result = C.c_void_p()
        check(self.call(0, HRESULT, (C.POINTER(GUID), C.POINTER(C.c_void_p)), C.byref(iid), C.byref(result)), "QueryInterface")
        return Com(result)

    def close(self) -> None:
        """Release this owned reference exactly once."""
        if self.ptr:
            self.call(2, UINT, ())
            self.ptr = C.c_void_p()

    def __del__(self) -> None:
        """Release a remaining COM reference during normal object destruction."""
        if getattr(self, "ptr", None):
            self.close()


def blob_bytes(blob: Com) -> bytes:
    """Copy an ID3DBlob/IDxcBlob while its storage remains alive."""
    return C.string_at(blob.call(3, C.c_void_p, ()), blob.call(4, C.c_size_t, ()))


def system_dll(name: str) -> C.WinDLL:
    """Load a Windows system DLL with system-directory dependency lookup only."""
    if os.name != "nt":
        raise RuntimeError("DX12 requires Windows and Windows Developer Mode")
    return C.WinDLL(name, winmode=0x800, use_last_error=True)


@dataclass(frozen=True)
class AdapterInfo:
    """Stable process-local DXGI adapter identity and dedicated memory capacity."""

    name: str
    vendor_id: int
    device_id: int
    luid: int
    memory: int


@dataclass(frozen=True)
class Capabilities:
    """Device features that affect lowering and compiled-program cache identity."""

    shader_model: int
    native16: bool
    doubles: bool
    int64: bool
    wave_ops: bool
    wave_min: int
    wave_max: int
    enhanced_barriers: bool

    @property
    def target(self) -> str:
        """Return a stable JSON target architecture for tinygrad compiler cache separation."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def arch(self) -> str:
        """Return the tinygrad Target.arch cache identity."""
        return self.target


class NativeRuntime:
    """Process-local immutable toolchain, device factory, and stable DXGI adapter mapping.

    Device submission remains independent; the initialization lock only protects this
    factory's lazy construction. Experimental shader models always use this local factory.
    """

    _instance: NativeRuntime | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> NativeRuntime:
        """Install missing toolchain files and load the runtime once per process."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        """Load the pinned runtime and enable experimental shader models before any device."""
        if os.name != "nt":
            raise RuntimeError("DX12 requires Windows")
        from tinygrad_dx12.setup import SDK_VERSION, architecture, check_pe, install

        self.path, self.arch = install(), architecture()
        self.sdk_dir = self.path / "agility/build/native/bin" / self.arch
        self.dxc_dir = self.path / "dxc/build/native/bin" / self.arch
        self.include_dir = self.path / "dxc/build/native/include/hlsl"
        for path in (self.sdk_dir / "D3D12Core.dll", self.dxc_dir / "dxcompiler.dll", self.dxc_dir / "dxil.dll"):
            check_pe(path, self.arch)
        self.dll_directories = [os.add_dll_directory(str(path.resolve())) for path in (self.sdk_dir, self.dxc_dir)]
        self.d3d12, self.dxgi, self.kernel32 = system_dll("d3d12.dll"), system_dll("dxgi.dll"), system_dll("kernel32.dll")
        self.d3d12.D3D12GetInterface.argtypes = (C.POINTER(GUID), C.POINTER(GUID), C.POINTER(C.c_void_p))
        self.d3d12.D3D12GetInterface.restype = HRESULT
        self.d3d12.D3D12SerializeVersionedRootSignature.argtypes = (
            C.POINTER(d3d.D3D12_VERSIONED_ROOT_SIGNATURE_DESC),
            C.POINTER(C.c_void_p),
            C.POINTER(C.c_void_p),
        )
        self.d3d12.D3D12SerializeVersionedRootSignature.restype = HRESULT
        self.d3d12.D3D12GetDebugInterface.argtypes = (C.POINTER(GUID), C.POINTER(C.c_void_p))
        self.d3d12.D3D12GetDebugInterface.restype = HRESULT
        self.dxgi.CreateDXGIFactory1.argtypes = (C.POINTER(GUID), C.POINTER(C.c_void_p))
        self.dxgi.CreateDXGIFactory1.restype = HRESULT
        self.kernel32.CreateEventW.argtypes = (C.c_void_p, C.c_int32, C.c_int32, C.c_wchar_p)
        self.kernel32.CreateEventW.restype = C.c_void_p
        self.kernel32.WaitForSingleObject.argtypes = (C.c_void_p, UINT)
        self.kernel32.WaitForSingleObject.restype = UINT
        self.kernel32.CloseHandle.argtypes = (C.c_void_p,)
        self.kernel32.CloseHandle.restype = C.c_int32
        pointer = C.c_void_p()
        check(
            self.d3d12.D3D12GetInterface(C.byref(d3d.CLSID_D3D12SDKConfiguration), C.byref(d3d.IID_ID3D12SDKConfiguration1), C.byref(pointer)),
            "D3D12GetInterface(ID3D12SDKConfiguration1)",
        )
        configuration = Com(pointer)
        pointer = C.c_void_p()
        check(
            configuration.call(
                4,
                HRESULT,
                (UINT, C.c_char_p, C.POINTER(GUID), C.POINTER(C.c_void_p)),
                SDK_VERSION,
                (str(self.sdk_dir.resolve()) + os.sep).encode("mbcs"),
                C.byref(d3d.IID_ID3D12DeviceFactory),
                C.byref(pointer),
            ),
            "CreateDeviceFactory(721)",
        )
        self.factory = Com(pointer)
        configuration.close()
        check(
            self.factory.call(8, HRESULT, (UINT, C.POINTER(GUID), C.c_void_p, C.c_void_p), 1, C.byref(d3d.D3D12ExperimentalShaderModels), None, None),
            "EnableExperimentalFeatures(D3D12ExperimentalShaderModels); enable Windows Developer Mode",
        )
        pointer = C.c_void_p()
        check(self.dxgi.CreateDXGIFactory1(C.byref(dxgi.IID_IDXGIFactory1), C.byref(pointer)), "CreateDXGIFactory1")
        self.dxgi_factory = Com(pointer)
        self.debug_enabled, self._created_device = False, False
        self._creation_lock = threading.Lock()
        self.adapters: list[tuple[Com, AdapterInfo]] = []
        ordinal = 0
        while True:
            pointer = C.c_void_p()
            result = self.dxgi_factory.call(12, HRESULT, (UINT, C.POINTER(C.c_void_p)), ordinal, C.byref(pointer))
            if result & 0xFFFFFFFF == dxgi.DXGI_ERROR_NOT_FOUND:
                break
            check(result, "EnumAdapters1")
            adapter = Com(pointer)
            desc = dxgi.DXGI_ADAPTER_DESC1()
            check(adapter.call(10, HRESULT, (C.POINTER(dxgi.DXGI_ADAPTER_DESC1),), C.byref(desc)), "IDXGIAdapter1.GetDesc1")
            if not desc.Flags & dxgi.DXGI_ADAPTER_FLAG_SOFTWARE:
                self.adapters.append((
                    adapter,
                    AdapterInfo(
                        desc.Description,
                        desc.VendorId,
                        desc.DeviceId,
                        (desc.AdapterLuid.HighPart & 0xFFFFFFFF) << 32 | desc.AdapterLuid.LowPart,
                        desc.DedicatedVideoMemory,
                    ),
                ))
            ordinal += 1

    def create_device(self, ordinal: int, debug: bool = False) -> tuple[Com, AdapterInfo, Capabilities]:
        """Create one adapter-local device and query required and optional scalar features."""
        if ordinal < 0 or ordinal >= len(self.adapters):
            raise IndexError(f"DX12 adapter {ordinal} does not exist ({len(self.adapters)} adapters)")
        with self._creation_lock:
            if debug and not self.debug_enabled:
                if self._created_device:
                    raise RuntimeError("Enable DX12_DEBUG before creating the first DX12 device")
                pointer = C.c_void_p()
                check(
                    self.factory.call(
                        7,
                        HRESULT,
                        (C.POINTER(GUID), C.POINTER(GUID), C.POINTER(C.c_void_p)),
                        C.byref(d3d.CLSID_D3D12Debug),
                        C.byref(d3d.IID_ID3D12Debug),
                        C.byref(pointer),
                    ),
                    "ID3D12DeviceFactory.GetConfigurationInterface(ID3D12Debug)",
                )
                layer = Com(pointer)
                layer.call(3, None, ())
                layer.close()
                self.debug_enabled = True
            adapter, info = self.adapters[ordinal]
            pointer = C.c_void_p()
            check(
                self.factory.call(
                    9,
                    HRESULT,
                    (C.c_void_p, UINT, C.POINTER(GUID), C.POINTER(C.c_void_p)),
                    adapter.ptr,
                    d3d.D3D_FEATURE_LEVEL_11_0,
                    C.byref(d3d.IID_ID3D12Device),
                    C.byref(pointer),
                ),
                "ID3D12DeviceFactory.CreateDevice",
            )
            self._created_device = True
        device = Com(pointer)
        options = self.feature(device, d3d.D3D12_FEATURE_D3D12_OPTIONS, d3d.D3D12_FEATURE_DATA_D3D12_OPTIONS())
        options1 = self.feature(device, d3d.D3D12_FEATURE_D3D12_OPTIONS1, d3d.D3D12_FEATURE_DATA_D3D12_OPTIONS1())
        options4 = self.feature(device, d3d.D3D12_FEATURE_D3D12_OPTIONS4, d3d.D3D12_FEATURE_DATA_D3D12_OPTIONS4())
        options12 = self.feature(device, d3d.D3D12_FEATURE_D3D12_OPTIONS12, d3d.D3D12_FEATURE_DATA_D3D12_OPTIONS12())
        if not options12.EnhancedBarriersSupported:
            raise RuntimeError(f"{info.name} does not support required D3D12 Enhanced Barriers")
        roots = self.feature(device, d3d.D3D12_FEATURE_ROOT_SIGNATURE, d3d.D3D12_FEATURE_DATA_ROOT_SIGNATURE(2))
        if roots.HighestVersion < 2:
            raise RuntimeError(f"{info.name} does not support root signatures 1.1")
        for model in range(0x6A, 0x5F, -1):
            shader = d3d.D3D12_FEATURE_DATA_SHADER_MODEL(model)
            if device.call(13, HRESULT, (UINT, C.c_void_p, UINT), d3d.D3D12_FEATURE_SHADER_MODEL, C.byref(shader), C.sizeof(shader)) >= 0:
                break
        else:
            raise RuntimeError(f"{info.name} does not support shader model 6")
        return (
            device,
            info,
            Capabilities(
                shader.HighestShaderModel,
                bool(options4.Native16BitShaderOpsSupported),
                bool(options.DoublePrecisionFloatShaderOps),
                bool(options1.Int64ShaderOps),
                bool(options1.WaveOps),
                options1.WaveLaneCountMin,
                options1.WaveLaneCountMax,
                bool(options12.EnhancedBarriersSupported),
            ),
        )

    @staticmethod
    def feature[T: C.Structure](device: Com, feature: int, data: T) -> T:
        """Fill a correctly sized feature structure, retaining the HRESULT on failure."""
        check(device.call(13, HRESULT, (UINT, C.c_void_p, UINT), feature, C.byref(data), C.sizeof(data)), f"CheckFeatureSupport({feature})")
        return data

    def serialize_root_signature(self, desc: d3d.D3D12_VERSIONED_ROOT_SIGNATURE_DESC) -> bytes:
        """Serialize root signature 1.1 and copy the blob, including native diagnostics on failure."""
        output, errors = C.c_void_p(), C.c_void_p()
        result = self.d3d12.D3D12SerializeVersionedRootSignature(C.byref(desc), C.byref(output), C.byref(errors))
        diagnostics = blob_bytes(Com(errors)).decode(errors="replace") if errors else ""
        check(result, f"D3D12SerializeVersionedRootSignature {diagnostics}")
        return blob_bytes(Com(output))


def query_matrix_support(
    device: Com, a: int, b: int, accumulator: int, wave_size: int, shape: tuple[int, int, int] = (16, 16, 16), group: bool = False
) -> dict:
    """Query the packaged preview API for one matrix dtype/shape/scope combination.

    Native datatype enum values belong to D3D12, not HLSL's differently numbered
    ComponentType enum. This optional query raises with its HRESULT when unavailable.
    """
    inputs = d3d.D3D12_LINEAR_ALGEBRA_WAVE_MATRIX_MULTIPLY_INPUTS(wave_size, a, b, accumulator)
    data = d3d.D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT(OperationType=2 if group else 1)
    if group:
        data.ThreadGroupMatrixMultiply.WaveInputs = inputs
        data.ThreadGroupMatrixMultiply.Shape = d3d.D3D12_LINEAR_ALGEBRA_MATRIX_MULTIPLY_SHAPE(*shape)
    else:
        data.WaveMatrixMultiply.Inputs = inputs
    NativeRuntime.feature(device, 78, data)
    if group:
        result = data.ThreadGroupMatrixMultiply
        return {
            "flags": result.SupportFlags,
            "min_threads": result.MinThreadGroupSize,
            "max_threads": result.MaxThreadGroupSize,
            "preferred_threads": result.PreferredThreadGroupSize,
        }
    wave_result = data.WaveMatrixMultiply
    if wave_result.NumShapes:
        shapes = (d3d.D3D12_LINEAR_ALGEBRA_MATRIX_MULTIPLY_SHAPE * wave_result.NumShapes)()
        wave_result.Shapes = shapes
        NativeRuntime.feature(device, 78, data)
        return {"flags": wave_result.SupportFlags, "shapes": [(s.M, s.K, s.N) for s in shapes]}
    return {"flags": wave_result.SupportFlags, "shapes": []}


def query_thread_matrix_support(device: Com, vector: int, matrix: int, bias: int, result: int) -> int:
    """Return native support flags for one thread-scope vector/matrix/bias/result datatype combination."""
    data = d3d.D3D12_FEATURE_DATA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT(OperationType=3)
    data.ThreadVectorMatrixMultiply = d3d.D3D12_LINEAR_ALGEBRA_THREAD_VECTOR_MATRIX_MULTIPLY_SUPPORT(vector, matrix, bias, result, 0)
    NativeRuntime.feature(device, 78, data)
    return data.ThreadVectorMatrixMultiply.SupportFlags
