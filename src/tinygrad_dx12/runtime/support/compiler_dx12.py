"""DXC COM compilation with an immutable source/DXIL envelope for specialization."""

import ctypes as C
import functools, hashlib, json, os, struct, sys, threading
from tinygrad.device import Compiler, CompileError
from tinygrad_dx12.runtime.autogen import dxc
from tinygrad_dx12.runtime.autogen.win32 import GUID, HRESULT, UINT
from tinygrad_dx12.runtime.support.dx12 import Com, blob_bytes, check

_MAGIC = b"TGDX12\x01\0"
_HEADER = struct.Struct("<8sI")


def pack_shader(source: str, dxil: bytes) -> bytes:
    """Preserve source metadata and DXIL in a versioned, length-prefixed shader container."""
    encoded = source.encode("utf-8")
    return _HEADER.pack(_MAGIC, len(encoded)) + encoded + dxil


def unpack_shader(binary: bytes) -> tuple[str, bytes]:
    """Recover shader source and DXIL; reject truncated or unknown cached containers."""
    if len(binary) < _HEADER.size:
        raise ValueError("Truncated DX12 shader container")
    magic, size = _HEADER.unpack_from(binary)
    end = _HEADER.size + size
    if magic != _MAGIC or end >= len(binary):
        raise ValueError("Invalid DX12 shader container")
    return binary[_HEADER.size : end].decode("utf-8"), binary[end:]


@functools.cache
def _load_dxc() -> tuple[C.WinDLL, object, str]:
    """Load matching private compiler/validator DLLs and retain dependency-directory handles."""
    if os.name != "nt":
        raise RuntimeError("DXC requires Windows")
    from tinygrad_dx12.setup import architecture, check_pe, install

    path, arch = install(), architecture()
    directory = path / "dxc/build/native/bin" / arch
    for filename in ("dxcompiler.dll", "dxil.dll"):
        check_pe(directory / filename, arch)
    handle = os.add_dll_directory(str(directory.resolve()))
    library = C.WinDLL(str((directory / "dxcompiler.dll").resolve()), winmode=0x1100)
    library.DxcCreateInstance.argtypes = (C.POINTER(GUID), C.POINTER(GUID), C.POINTER(C.c_void_p))
    library.DxcCreateInstance.restype = HRESULT
    return library, handle, str((path / "dxc/build/native/include/hlsl").resolve())


class DX12Compiler(Compiler):
    """Compile HLSL using the pinned DXC, with settings/capabilities in tinygrad's cache key."""

    def __init__(self, target: str = "cs_6_10", native16: bool = False, capabilities: str = "") -> None:
        """Configure compilation lazily; construction requires neither Windows nor installed DLLs."""
        from tinygrad_dx12.setup import PACKAGES

        self.target, self.native16 = target, native16
        self.settings = ("-T", target, "-E", "main", "-HV", "2021", "-O3", "-fno-finite-math-only", "-res_may_alias") + (
            ("-enable-16bit-types",) if native16 else ()
        )
        identity = repr((PACKAGES[1].version, self.settings, capabilities, _MAGIC))
        super().__init__("dx12_" + hashlib.sha256(identity.encode()).hexdigest())
        self.lock = threading.Lock()
        self.diagnostics = ""

    @functools.cached_property
    def _interfaces(self) -> tuple[Com, Com, Com, str]:
        """Create one compiler, utility interface and default include handler for this adapter."""
        library, _, include = _load_dxc()
        interfaces = []
        for clsid, iid in ((dxc.CLSID_DxcUtils, dxc.IID_IDxcUtils), (dxc.CLSID_DxcCompiler, dxc.IID_IDxcCompiler3)):
            pointer = C.c_void_p()
            check(library.DxcCreateInstance(C.byref(clsid), C.byref(iid), C.byref(pointer)), "DxcCreateInstance")
            interfaces.append(Com(pointer))
        pointer = C.c_void_p()
        check(interfaces[0].call(9, HRESULT, (C.POINTER(C.c_void_p),), C.byref(pointer)), "IDxcUtils.CreateDefaultIncludeHandler")
        return interfaces[0], interfaces[1], Com(pointer), include

    def compile_dxil(self, source: str) -> bytes:
        """Compile UTF-8 HLSL to owned DXIL bytes, reporting both native and shader diagnostics."""
        with self.lock:
            if os.getenv("DX12_DUMP_SHADERS", "0") == "1":
                print(source, file=sys.stderr)
            _, compiler, include_handler, include = self._interfaces
            encoded = source.encode("utf-8")
            source_buffer = C.create_string_buffer(encoded)
            buffer = dxc.DxcBuffer(C.cast(source_buffer, C.c_void_p), len(encoded), dxc.DXC_CP_UTF8)
            # Preserve pointwise evaluation and bit-level expansions while freeing generated arithmetic loops.
            strict = source.startswith("// DX12:") and json.loads(source.splitlines()[0][8:]).get("strict_math", False)
            settings = self.settings + (("-Gis",) if strict else ()) + ("-I", include)
            arguments = (C.c_wchar_p * len(settings))(*settings)
            pointer = C.c_void_p()
            check(
                compiler.call(
                    3,
                    HRESULT,
                    (C.POINTER(dxc.DxcBuffer), C.POINTER(C.c_wchar_p), UINT, C.c_void_p, C.POINTER(GUID), C.POINTER(C.c_void_p)),
                    C.byref(buffer),
                    arguments,
                    len(arguments),
                    include_handler.ptr,
                    C.byref(dxc.IID_IDxcResult),
                    C.byref(pointer),
                ),
                "IDxcCompiler3.Compile",
            )
            result = Com(pointer)
            try:
                pointer = C.c_void_p()
                check(result.call(5, HRESULT, (C.POINTER(C.c_void_p),), C.byref(pointer)), "IDxcResult.GetErrorBuffer")
                self.diagnostics = blob_bytes(Com(pointer)).rstrip(b"\0").decode("utf-8", errors="replace") if pointer else ""
                status = HRESULT()
                check(result.call(3, HRESULT, (C.POINTER(HRESULT),), C.byref(status)), "IDxcResult.GetStatus")
                if status.value < 0:
                    raise CompileError(f"DXC {self.target} failed (0x{status.value & 0xFFFFFFFF:08X}):\n{self.diagnostics}")
                pointer = C.c_void_p()
                check(result.call(4, HRESULT, (C.POINTER(C.c_void_p),), C.byref(pointer)), "IDxcResult.GetResult")
                return blob_bytes(Com(pointer))
            finally:
                result.close()

    def compile(self, src: str) -> bytes:
        """Compile a cacheable shader, preserving source for runtime local-size specialization."""
        return pack_shader(src, self.compile_dxil(src))
