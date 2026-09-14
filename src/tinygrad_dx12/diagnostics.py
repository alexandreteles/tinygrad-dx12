"""Report pinned DLLs, device features, and optional shader compiler/PSO acceptance."""

import argparse, ctypes as C, json
from dataclasses import asdict
from pathlib import Path
from tinygrad.device import CompileError, TinyELF
from tinygrad.dtype import dtypes
from tinygrad_dx12.runtime.autogen import d3d12 as d
from tinygrad_dx12.runtime.ops_dx12 import DX12Device, DX12Program
from tinygrad_dx12.runtime.support.dx12 import NativeRuntime, query_matrix_support, system_dll
from tinygrad_dx12.renderer.linalg import linalg_source
from tinygrad_dx12.setup import PACKAGES


class _FixedFileInfo(C.Structure):
    _fields_ = [
        (name, C.c_uint32)
        for name in (
            "signature",
            "structure",
            "version_ms",
            "version_ls",
            "product_ms",
            "product_ls",
            "flags_mask",
            "flags",
            "os",
            "file_type",
            "subtype",
            "date_ms",
            "date_ls",
        )
    ]


def dll_version(path: Path) -> str | None:
    """Read a DLL's embedded Windows version resource, returning None when it is absent."""
    library = system_dll("version.dll")
    library.GetFileVersionInfoSizeW.argtypes, library.GetFileVersionInfoSizeW.restype = (C.c_wchar_p, C.c_void_p), C.c_uint32
    library.GetFileVersionInfoW.argtypes = (C.c_wchar_p, C.c_uint32, C.c_uint32, C.c_void_p)
    library.GetFileVersionInfoW.restype = C.c_int32
    library.VerQueryValueW.argtypes = (C.c_void_p, C.c_wchar_p, C.c_void_p, C.c_void_p)
    library.VerQueryValueW.restype = C.c_int32
    size = library.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return None
    buffer = C.create_string_buffer(size)
    if not library.GetFileVersionInfoW(str(path), 0, size, buffer):
        raise C.WinError()
    pointer, length = C.c_void_p(), C.c_uint32()
    if not library.VerQueryValueW(buffer, "\\", C.byref(pointer), C.byref(length)):
        raise C.WinError()
    info = C.cast(pointer, C.POINTER(_FixedFileInfo)).contents
    return ".".join(str(value) for value in (info.version_ms >> 16, info.version_ms & 65535, info.version_ls >> 16, info.version_ls & 65535))


def probes(device: DX12Device) -> list[dict]:
    """Compile matrix candidates across exposed types and separately attempt their device-local PSOs.

    These diagnostics do not dispatch work or infer native instruction use. A rejected
    candidate describes this source/shape/scope only and does not remove a logical dtype.
    """
    from tinygrad_dx12.renderer.linalg import DATATYPES, _accumulator

    results = []
    for scope in ("Wave", "ThreadGroup"):
        for dtype in (*DATATYPES, dtypes.bfloat16, dtypes.float64):
            shape = (16, 16, 16)
            accumulator = _accumulator(dtype, dtype)
            group_threads = 32
            entry: dict = {"dtype": dtype.name, "scope": scope, "compiler": False, "pso": False}
            if dtype in DATATYPES and accumulator in DATATYPES:
                try:
                    query = query_matrix_support(
                        device.native,
                        DATATYPES[dtype],
                        DATATYPES[dtype],
                        DATATYPES[accumulator],
                        device.caps.wave_min,
                        shape=shape,
                        group=scope == "ThreadGroup",
                    )
                    entry["query"] = query
                    if scope == "Wave" and query["shapes"]:
                        shape = query["shapes"][0]
                    if scope == "ThreadGroup" and query["flags"] & 1:
                        group_threads = query["preferred_threads"] or query["min_threads"]
                except RuntimeError as exc:
                    entry["query_error"] = str(exc)
            m, k, n = shape
            entry["shape"] = {"m": m, "k": k, "n": n}
            entry["threads"] = device.caps.wave_min if scope == "Wave" else group_threads
            try:
                source = linalg_source(
                    dtype,
                    dtype,
                    m,
                    n,
                    k,
                    scope=scope,
                    wave_size=device.caps.wave_min,
                    tile=(m, n, k),
                    group_threads=group_threads,
                    accumulator=accumulator,
                )
            except ValueError as exc:
                entry["error"] = str(exc)
                results.append(entry)
                continue
            try:
                binary = device.compiler.compile(source)
                entry["compiler"] = True
            except (RuntimeError, CompileError) as exc:
                entry["error"] = str(exc)
                results.append(entry)
                continue
            try:
                DX12Program(device, TinyELF(binary, f"probe_{dtype.name}_{scope}", device.renderer.target, ()))
                entry["pso"] = True
            except RuntimeError as exc:
                entry["error"] = str(exc)
            results.append(entry)
    return results


def main() -> None:
    """Print one JSON diagnostic report; optional probes distinguish compiler and PSO acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="compile and create matrix PSOs, including a comparable FP16 control; no dispatch")
    parser.add_argument("--adapter", type=int, help="inspect one device ordinal (default: every hardware adapter)")
    parser.add_argument("--debug", action="store_true", help="enable the optional D3D12 debug layer before device creation")
    args = parser.parse_args()
    if args.debug:
        import os

        os.environ["DX12_DEBUG"] = "1"
    runtime = NativeRuntime.get()
    report = {"packages": [asdict(package) for package in PACKAGES], "dlls": [], "adapters": []}
    runtime.kernel32.GetModuleFileNameW.argtypes = (C.c_void_p, C.c_wchar_p, C.c_uint32)
    runtime.kernel32.GetModuleFileNameW.restype = C.c_uint32
    system_paths = []
    for library in (runtime.d3d12, runtime.dxgi, runtime.kernel32):
        filename = C.create_unicode_buffer(32768)
        if not runtime.kernel32.GetModuleFileNameW(library._handle, filename, len(filename)):
            raise C.WinError()
        system_paths.append(Path(filename.value))
    for path in (*system_paths, runtime.sdk_dir / "D3D12Core.dll", runtime.dxc_dir / "dxcompiler.dll", runtime.dxc_dir / "dxil.dll"):
        report["dlls"].append({"path": str(path), "version": dll_version(path)})
    for ordinal in range(len(runtime.adapters)) if args.adapter is None else (args.adapter,):
        device = DX12Device("DX12" if ordinal == 0 else f"DX12:{ordinal}")
        adapter = {"ordinal": ordinal, "identity": asdict(runtime.adapters[ordinal][1]), "capabilities": asdict(device.caps)}
        try:
            adapter["linalg_tier"] = runtime.feature(
                device.native, d.D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT, d.D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT()
            ).LinearAlgebraTier
        except RuntimeError as exc:
            adapter["linalg_query_error"] = str(exc)
        if args.probe:
            adapter["probes"] = probes(device)
        report["adapters"].append(adapter)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
