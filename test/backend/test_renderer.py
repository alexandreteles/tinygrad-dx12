"""Binding ABI regression tests and actual DXC compilation of tinygrad kernels."""

import os, struct
from dataclasses import replace
import pytest
from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.uop.ops import UOp, Ops
from tinygrad_dx12.renderer.cstyle import make_layout, DX12Renderer, parse_source, specialize_source


def test_indirect_published_abi() -> None:
    """Check real argument bytes against the architecture's independently stated ABI."""
    buffers = [(f"data{i}", (UOp.placeholder((8,), dtypes.float, i), i == 2)) for i in range(3)]
    scalar = UOp.variable("count", 0, 100, dtype=dtypes.int32, param=True)
    scalar = scalar.replace(arg=replace(scalar.arg, slot=3))
    layout = make_layout(buffers + [("count", (scalar, False))], (8, 1, 1))
    record = layout.pack_record((0x1000, 0x2000, 0x3000), (1, 2, 3), (17,), (5, 6, 7))
    # The generated layout explicitly puts its three offset words before count.
    assert record == struct.pack("<QQQIIIIIII4x", 0x1000, 0x2000, 0x3000, 1, 2, 3, 17, 5, 6, 7)
    assert layout.stride == 56 and layout.root_dwords == 10


@pytest.mark.parametrize(
    "buffers,cbv,table", [(5, False, False), (21, False, False), (22, True, False), (31, True, False), (32, True, True), (80, True, True)]
)
def test_root_budget(buffers: int, cbv: bool, table: bool) -> None:
    """Exercise both transitions that avoid exceeding D3D12's 64-DWORD root limit."""
    layout = make_layout([(f"data{i}", (UOp.placeholder((8,), dtypes.float, i), i % 2 == 0)) for i in range(buffers)], (1, 1, 1))
    assert (layout.scalar_cbv, layout.table) == (cbv, table)
    assert layout.root_dwords <= 64
    record = layout.pack_record(tuple(4096 * (i + 1) for i in range(buffers)), (0,) * buffers, (), (7, 1, 1), 0x100000 if cbv else 0)
    assert struct.unpack_from("<III", record, layout.dispatch_offset) == (7, 1, 1)
    if cbv:
        assert struct.unpack_from("<Q", record, layout.scalar_offset)[0] == 0x100000


def test_64bit_scalar_words() -> None:
    """Check exact signed/unsigned integer bit patterns and explicit scalar placement."""
    x = UOp.variable("wide", 0, 2**64 - 1, dtype=dtypes.uint64, param=True)
    layout = make_layout([("wide", (x.replace(arg=replace(x.arg, slot=0)), False))], (1, 1, 1))
    assert layout.scalar_data((), (0x0123456789ABCDEF,)) == bytes.fromhex("efcdab8967452301")


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires a DX12 adapter")
@pytest.mark.parametrize("dtype", [dtypes.bfloat16, *dtypes.fp8s])
def test_emulated_max_reduction(dtype: object) -> None:
    """Nonnative floating maxima retain their dtype across local reductions and changed JIT inputs."""

    @TinyJit
    def maxima(value: Tensor) -> Tensor:
        """Capture a row maximum followed by a consumer to exercise shared accumulators."""
        return value.max(axis=1).realize().maximum(0).realize()

    for step in range(4):
        rows = [[(column + row) % 5 - 2 + step for column in range(1024)] for row in range(3)]
        result = maxima(Tensor(rows, device="DX12", dtype=dtype))
        assert result.dtype is dtype and result.tolist() == [2 + step] * 3


def test_capability_target_cache_separation() -> None:
    """Different arithmetic capabilities must select different tinygrad cache keys."""
    from tinygrad.codegen import to_program_key
    from tinygrad.uop.ops import KernelInfo

    a = DX12Renderer(Target("DX12", arch='{"native16":false,"int64":false}'))
    b = DX12Renderer(Target("DX12", arch='{"native16":true,"int64":true}'))
    sink = UOp.sink(arg=KernelInfo())
    assert to_program_key(sink, a) != to_program_key(sink, b)
    assert dtypes.half not in a.supported_dtypes() and dtypes.half in b.supported_dtypes()
    assert dtypes.bfloat16 not in b.supported_dtypes()


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires the installed DX12 preview toolchain")
@pytest.mark.parametrize(
    "dtype",
    [dtypes.float, dtypes.half, dtypes.double, dtypes.bfloat16, dtypes.fp8e4m3, dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint64],
)
def test_compile_reduction(dtype: object) -> None:
    """Compile generated reduction kernels, including locals and nonnative decompositions."""
    renderer = DX12Renderer(Target("DX12", arch='{"native16":true,"doubles":true,"int64":true,"shader_model":106}'))
    value = Tensor.empty((19, 33), device="NULL", dtype=dtype)
    calls = (value + value).sum(1).schedule_linear().src
    programs = [to_program(call.src[0], renderer) for call in calls if call.src[0].op is Ops.SINK]
    assert programs
    for program in programs:
        from tinygrad_dx12.runtime.support.compiler_dx12 import unpack_shader

        source, dxil = unpack_shader(program.src[3].arg)
        assert dxil.startswith(b"DXBC")
        layout = parse_source(source)
        alternate = specialize_source(source, (8, 1, 1))
        assert parse_source(alternate).buffers == layout.buffers
        assert parse_source(alternate).local_size == (8, 1, 1)


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires a DX12 adapter")
@pytest.mark.parametrize("dtype", [dtypes.uint8, dtypes.int8, dtypes.uint16, dtypes.int16, dtypes.float16, dtypes.float32])
def test_concurrent_packed_stores(dtype: object) -> None:
    """Thousands of lanes write neighboring elements, exercising shared-DWORD ownership."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    values = [i % 31 - 15 for i in range(4099)]
    if dtype in (dtypes.uint8, dtypes.uint16):
        values = [x + 15 for x in values]
    source = Tensor(values, device="DX12", dtype=dtype)
    for _ in range(3):
        assert (source + 1).tolist() == [x + 1 for x in values]


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires a DX12 adapter")
def test_double_truncation_keeps_precision() -> None:
    """FP64 truncation cannot silently pass through HLSL's float32 trunc intrinsic."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    values = Tensor([1000000000000.75, -1000000000000.75, 1.75], device="DX12", dtype=dtypes.float64)
    assert values.trunc().tolist() == [1000000000000.0, -1000000000000.0, 1.0]
    assert values.floor().tolist() == [1000000000000.0, -1000000000001.0, 1.0]
    assert values.ceil().tolist() == [1000000000001.0, -1000000000000.0, 2.0]


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires a DX12 adapter")
def test_half_matmul_at_resource_boundary() -> None:
    """Exercise the final half of an exact-size resource; an extra DWORD read faulted the device."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    data = [[(i + j) % 7 - 3 for j in range(512)] for i in range(512)]
    identity = [[int(i == j) for j in range(512)] for i in range(512)]
    from tinygrad.helpers import Context

    with Context(DX12_LINALG=0):
        result = Tensor(data, device="DX12", dtype=dtypes.float16) @ Tensor(identity, device="DX12", dtype=dtypes.float16)
        assert result.tolist() == data


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires the installed DX12 preview toolchain")
def test_double_without_integer64_compiles() -> None:
    """Independent FP64 capability must permit double arithmetic, truncation, and storage."""
    renderer = DX12Renderer(Target("DX12", arch='{"doubles":true,"int64":false,"shader_model":106}'))
    value = Tensor.empty(32, device="NULL", dtype=dtypes.float64)
    calls = ((value + 0.25).trunc() * 2).schedule_linear().src
    assert dtypes.float64 in renderer.supported_dtypes() and dtypes.uint64 not in renderer.supported_dtypes()
    for call in calls:
        program = to_program(call.src[0], renderer)
        assert "int64_t" not in program.src[2].arg


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires a DX12 adapter")
@pytest.mark.parametrize("offset", [1, 2, 3])
def test_unaligned_float_vectors_and_matrix(offset: int) -> None:
    """Native vector loads must shift adjacent DWORD lanes for arbitrary byte-offset views."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    values = [[(i + j) % 7 - 3 for j in range(32)] for i in range(32)]
    raw = [99] * offset + list(struct.pack("<1024f", *[x for row in values for x in row]))
    backing = Tensor(raw, device="DX12", dtype=dtypes.uint8).realize()
    data = Tensor(UOp.from_buffer(backing.uop.buffer.view(1024, dtypes.float32, offset).ensure_allocated())).reshape(32, 32)
    assert (data + 1).tolist() == [[x + 1 for x in row] for row in values]
    identity = Tensor([[int(i == j) for j in range(32)] for i in range(32)], device="DX12", dtype=dtypes.float32)
    assert (data @ identity).tolist() == values


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires a DX12 adapter")
def test_alignment_choice_changes_inside_same_jit() -> None:
    """One captured shader must select aligned and byte-view paths as input buffers change."""
    import tinygrad_dx12

    tinygrad_dx12.register()
    identity = Tensor([[int(i == j) for j in range(128)] for i in range(128)], device="DX12", dtype=dtypes.float32).realize()

    def product(value: Tensor, weight: Tensor) -> Tensor:
        """Capture the matrix and a consumer to exercise native graph replay."""
        return (value @ weight).realize().relu().realize()

    jit, captured = TinyJit(product), None
    for step, offset in enumerate((0, 0, 1, 2, 3, 0)):
        values = [[(i + j + step) % 7 - 3 for j in range(128)] for i in range(128)]
        backing = Tensor(
            [99] * offset + list(struct.pack("<16384f", *[x for row in values for x in row])), device="DX12", dtype=dtypes.uint8
        ).realize()
        view = backing.uop.buffer.view(16384, dtypes.float32, offset).ensure_allocated()
        result = jit(Tensor(UOp.from_buffer(view)).reshape(128, 128), identity)
        assert result.tolist() == [[max(x, 0) for x in row] for row in values]
        if step == 1:
            captured = jit.captured
            assert captured is not None and any(u.op is Ops.SOURCE for u in captured._linear.toposort())
        if step > 1:
            assert jit.captured is captured


@pytest.mark.skipif(os.getenv("DX12_TEST_GPU") != "1", reason="requires a DX12 adapter")
@pytest.mark.parametrize("native16", [False, True])
@pytest.mark.parametrize("dtype", [dtypes.int8, dtypes.uint8, dtypes.int16, dtypes.uint16])
def test_narrow_intermediate_overflow(dtype: object, native16: bool) -> None:
    """Logical narrow overflow must precede wider casts, comparisons, shifts, and chained additions."""
    import tinygrad_dx12
    from tinygrad.engine.realize import run_linear

    tinygrad_dx12.register()
    limit = 1 << (dtype.bitsize - 1 if dtype in dtypes.sints else dtype.bitsize)
    a, b = (
        ([limit - 8, 8 - limit, limit - 1, -limit], [10, -10, 1, -1])
        if dtype in dtypes.sints
        else ([limit - 6, 2, limit - 1, 0], [10, limit - 1, 1, limit - 1])
    )

    def wrap(value: int) -> int:
        """Interpret the stored-width residue using integer arithmetic independent of HLSL shifts."""
        residue = value % (1 << dtype.bitsize)
        return residue - (1 << dtype.bitsize) if dtype in dtypes.sints and residue >= limit else residue

    x, y = Tensor(a, device="DX12", dtype=dtype).realize(), Tensor(b, device="DX12", dtype=dtype).realize()
    sums = [wrap(i + j) for i, j in zip(a, b)]
    expected = [
        sums,
        [wrap(i * j) for i, j in zip(a, b)],
        [int(s < 0) for s in sums],
        [s >> 1 for s in sums],
        [wrap(s + i) for s, i in zip(sums, a)],
        [s + 100000 for s in sums],
    ]
    outputs = [
        (x + y).cast(dtypes.int32),
        (x * y).cast(dtypes.int32),
        ((x + y) < 0).cast(dtypes.int32),
        ((x + y) >> 1).cast(dtypes.int32),
        (x + y + x).cast(dtypes.int32),
        (x + y) + Tensor([100000] * 4, device="DX12", dtype=dtypes.int32),
    ]
    result = outputs[0].stack(*outputs[1:])
    renderer = DX12Renderer(Target("DX12", arch=f'{{"native16":{str(native16).lower()},"int64":true,"shader_model":106}}'))
    linear = result.schedule_linear()
    run_linear(
        linear.replace(src=tuple(c.replace(src=(to_program(c.src[0], renderer), *c.src[1:])) if c.src[0].op is Ops.SINK else c for c in linear.src))
    )
    assert result.tolist() == expected
