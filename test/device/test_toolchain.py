"""Toolchain integrity and native ABI regression checks."""

import ctypes as C
import os, struct, threading, zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest
from tinygrad.device import CompileError
from tinygrad_dx12.setup import check_pe, extract_package
from tinygrad_dx12.runtime.autogen.win32 import GUID
from tinygrad_dx12.runtime.support.dx12 import check
from tinygrad_dx12.runtime.support.compiler_dx12 import DX12Compiler, pack_shader, unpack_shader
from tinygrad_dx12.runtime.ops_dx12 import DX12Buffer


def test_guid_windows_byte_order() -> None:
    """Verify GUID byte order against the SDK configuration's published identifier."""
    guid = GUID("7cda6aca-a03e-49c8-9458-0334d20e07ce")
    assert bytes(guid) == bytes.fromhex("ca6ada7c3ea0c84994580334d20e07ce")
    assert C.sizeof(guid) == 16


@pytest.mark.parametrize("hresult", [-2147024809, 0x80070057])
def test_hresult_signed_failure(hresult: int) -> None:
    """Unsigned Python inputs must still fail according to signed Windows HRESULT rules."""
    with pytest.raises(RuntimeError, match="operation failed: HRESULT 0x80070057"):
        check(hresult, "operation")


@pytest.mark.parametrize("name", ["../escape.dll", "/escape.dll", "C:/escape.dll", "a/../../escape.dll", "..\\escape.dll"])
def test_archive_rejects_escape(tmp_path: Path, name: str) -> None:
    """Reject traversal in any archive member before extracting any accepted payload."""
    archive = tmp_path / "package.nupkg"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("build/native/include/good.h", "payload")
        output.writestr(name, "escape")
    destination = tmp_path / "installed"
    with pytest.raises(ValueError, match="Unsafe archive"):
        extract_package(archive, destination, "x64")
    assert not destination.exists()


def test_pe_requires_process_machine(tmp_path: Path) -> None:
    """An ARM64 DLL must not be accepted into an x64 toolchain despite both using 64-bit pointers."""
    payload = bytearray(134)
    payload[:2] = b"MZ"
    struct.pack_into("<I", payload, 60, 128)
    struct.pack_into("<4sH", payload, 128, b"PE\0\0", 0xAA64)
    image = tmp_path / "arm64.dll"
    image.write_bytes(payload)
    check_pe(image, "arm64")
    with pytest.raises(ValueError, match="Wrong PE architecture"):
        check_pe(image, "x64")


@pytest.mark.parametrize("payload", [b"", b"TGDX12\1\0", struct.pack("<8sI", b"TGDX12\1\0", 100) + b"short"])
def test_reject_truncated_shader(payload: bytes) -> None:
    """Corrupt cached metadata must fail before a native PSO creation call."""
    with pytest.raises(ValueError):
        unpack_shader(payload)


def test_shader_source_preserved() -> None:
    """Local-size specialization retains the full Unicode source and exact compiled bytes."""
    source, dxil = "// \u03bb\n[numthreads(2,1,1)] void main() {}", b"DXBC\0\1\2"
    assert unpack_shader(pack_shader(source, dxil)) == (source, dxil)


def metadata_buffer(size: int, offset: int = 0, owner: DX12Buffer | None = None) -> DX12Buffer:
    """Construct only byte-view metadata, without allocating a native resource or device."""
    buffer = DX12Buffer.__new__(DX12Buffer)
    buffer.size, buffer.offset, buffer.gpu_address, buffer._owner = size, offset, 0x100000000, owner
    return buffer


@pytest.mark.parametrize("size,offset", [(2**32, 0), (2**32 + 1, 3), (128 << 30, 15)])
def test_large_binding_keeps_full_gpu_address(size: int, offset: int) -> None:
    """Logical shader spans no longer impose a four-GiB allocation or view-size cap."""
    buffer = metadata_buffer(size, offset)
    assert buffer.binding_address == 0x100000000 + offset


def test_small_binding_in_large_planned_arena() -> None:
    """A small view can rebase its 64-bit GPU address inside an arena larger than four GiB."""
    owner = metadata_buffer(2**33)
    view = metadata_buffer(16, 2**32 + 129, owner)
    assert view.binding_address == 2**33 + 129


@pytest.mark.skipif(os.name != "nt" or not os.environ.get("TINYGRAD_DX12_CACHE"), reason="requires installed Windows DXC")
def test_native_dxc_reports_errors_and_recovers() -> None:
    """A real shader syntax failure reports diagnostics and does not poison the next compilation."""
    compiler = DX12Compiler(native16=True)
    with pytest.raises(CompileError, match="error:"):
        compiler.compile_dxil("[numthreads(1,1,1)] void main() { not_valid syntax; }")
    binary = compiler.compile_dxil("RWByteAddressBuffer o:register(u0); [numthreads(1,1,1)] void main() { o.Store(0, 123u); }")
    assert binary.startswith(b"DXBC")
    assert len(binary) > 256


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires an installed DX12 device")
def test_shared_program_concurrent_native_caches() -> None:
    """Concurrent graph builders must retain the same native PSO and indirect signature."""
    from tinygrad import Device
    from tinygrad.device import TinyELF
    from tinygrad_dx12.renderer.cstyle import BindingLayout, BufferBinding
    from tinygrad_dx12.runtime.support.dx12 import Com

    device = Device["DX12"]
    layout = BindingLayout((BufferBinding(0, "output", True, 0, 0, 0),), (), 1, 1, False, False)
    source = (
        "// DX12:" + layout.to_json() + "\nRWByteAddressBuffer output:register(u0);\n"
        "cbuffer Params:register(b0) { uint4 args[1]; };\n"
        "#define DX12_LOCAL_X 1\n#define DX12_LOCAL_Y 1\n#define DX12_LOCAL_Z 1\n"
        "[numthreads(DX12_LOCAL_X,DX12_LOCAL_Y,DX12_LOCAL_Z)] void main(uint3 id:SV_GroupThreadID) "
        "{ if(id.x==0) output.Store(args[0].x,123u); }"
    )
    program = device.runtime(TinyELF(device.compiler.compile_cached(source), "cache_lifetime", device.renderer.target, ()))
    barrier = threading.Barrier(8, timeout=30)

    def compile_variant(index: int) -> tuple[Com, Com]:
        """Start both native cache misses together without introducing artificial native delays."""
        barrier.wait()
        pipeline = program.specialize((2, 1, 1))
        barrier.wait()
        return pipeline, program.indirect_signature()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(compile_variant, range(8)))
    pipeline, signature = results[0]
    assert all(pso is pipeline and indirect is signature for pso, indirect in results)
    assert pipeline.ptr and signature.ptr
    output = device.buffer(4)
    program(output, local_size=(2, 1, 1), wait=True)
    result = bytearray(4)
    device.allocator._copyout(memoryview(result), output)
    assert struct.unpack("<I", result)[0] == 123
