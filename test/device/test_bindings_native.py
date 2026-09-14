"""Exercise root descriptors, CBVs, and table overflow against native DXIL execution."""

import os, struct
import pytest
from tinygrad import Device
from tinygrad.device import TinyELF
from tinygrad_dx12.renderer.cstyle import BindingLayout, BufferBinding
from tinygrad_dx12.runtime.ops_dx12 import DX12Device, DX12Program

pytestmark = pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="set DX12_TEST_GPU=1 to execute GPU contracts")


def sum_program(device: DX12Device, count: int) -> DX12Program:
    """Compile an independently specified integer sum to exercise each native binding regime."""
    cbv, table = count > 16, count > 31
    layout = BindingLayout(
        tuple(BufferBinding(i, f"b{i}", i == 0, max(i - 1, 0), 0 if table else i, i) for i in range(count)),
        (),
        count,
        1 if table else count,
        cbv,
        table,
    )
    declarations = [f"{'RW' if i == 0 else ''}ByteAddressBuffer b{i}:register({'u' if i == 0 else 't'}{max(i - 1, 0)});" for i in range(count)]
    readers = [
        f"uint read{i}(uint off) {{ uint v=0; for(uint j=0;j<4;j++) v|=((b{i}.Load((off+j)&~3u)>>(((off+j)&3u)*8u))&255u)<<(8u*j); return v; }}"
        for i in range(1, count)
    ]
    loads = "+".join(f"read{i}(args[{i // 4}][{i % 4}])" for i in range(1, count))
    source = (
        "// DX12:"
        + layout.to_json()
        + "\n"
        + "\n".join(declarations + readers)
        + f"\ncbuffer Params:register(b0) {{ uint4 args[{(count + 3) // 4}]; }};\n"
        "#define DX12_LOCAL_X 1\n#define DX12_LOCAL_Y 1\n#define DX12_LOCAL_Z 1\n"
        f"[numthreads(DX12_LOCAL_X,DX12_LOCAL_Y,DX12_LOCAL_Z)] void main() {{ b0.Store(args[0].x,{loads}); }}"
    )
    return device.runtime(TinyELF(device.compiler.compile_cached(source), "sum_binding_views", device.renderer.target, ()))


@pytest.mark.parametrize("count", [3, 17, 33, 70])
def test_native_binding_layouts(count: int) -> None:
    """Native shader output checks host packing, root serialization, and overflow descriptors together."""
    device = Device["DX12"]
    program = sum_program(device, count)
    buffers = [device.allocator._offset(device.buffer(16), 4, i % 4) for i in range(count)]
    for i, buf in enumerate(buffers):
        device.allocator._copyin(buf, memoryview(struct.pack("<I", i * 17)))
    elapsed = program(*buffers, wait=True)
    result = bytearray(4)
    device.allocator._copyout(memoryview(result), buffers[0])
    assert struct.unpack("<I", result)[0] == sum(i * 17 for i in range(1, count))
    assert elapsed is not None and elapsed > 0
