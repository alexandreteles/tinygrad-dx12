"""Small external integration hooks for the installed tinygrad source API."""

import importlib, sys, weakref
from dataclasses import replace
from functools import wraps
from tinygrad.device import Buffer
from tinygrad.uop.ops import AxisType, Ops, UOp

_registered = False


def resolve_graph_input(source: UOp, input_uops: tuple[UOp, ...]) -> Buffer:
    """Resolve an existing JIT buffer/view template against the current inputs.

    This includes MSELECT(PARAM), which appears after tuple-device splitting
    and is not included in the base runner's direct-PARAM replacement list.
    """
    from tinygrad.engine.realize import _resolve

    buf = _resolve(source, input_uops).buffer
    if not isinstance(buf, Buffer):
        raise RuntimeError("DX12 graph input did not resolve to one device")
    return buf.ensure_allocated()


def _local_call(call: UOp, lane: int) -> UOp:
    """Select a tuple-device lane and specialize its original device index.

    Device-range values are lane indices, not adapter ordinals. The selected
    buffers keep their complete adapter identifiers. Rebuild PROGRAM metadata
    from its retained sink so a previously compiled binary cannot leak between
    adapters with different capabilities.
    """
    ast = call.src[0]
    sink = ast.src[0] if ast.op is Ops.PROGRAM else ast
    replacements = {
        u: u.const_like(lane)
        if (u.op is Ops.RANGE and u.arg[-1] is AxisType.DEVICE) or (u.op is Ops.PARAM and u.arg.name == "_device_num")
        else u.replace(arg=replace(u.arg, device=u.arg.device[lane]))
        for u in sink.toposort()
        if (u.op is Ops.RANGE and u.arg[-1] is AxisType.DEVICE)
        or (u.op is Ops.PARAM and (u.arg.name == "_device_num" or isinstance(u.arg.device, tuple)))
    }
    sink = sink.substitute(replacements)
    body = ast.replace(src=(sink,), arg=None) if ast.op is Ops.PROGRAM else sink
    return call.replace(src=(body, *(b.mselect(lane) if isinstance(b.device, tuple) else b for b in call.src[1:])))


def split_dx12_calls(linear: UOp) -> UOp:
    """Expand DX12 tuple kernels in place before renderer selection.

    Copies and non-DX12 calls retain their order and identity. Each expanded
    kernel is compiled with its selected adapter's renderer, including when
    different adapter ordinals appear in a reordered device tuple.
    """

    def expand(call: UOp) -> tuple[UOp, ...]:
        """Return the device-local calls for one sequential operation."""
        if call.op is not Ops.CALL or call.src[0].op not in (Ops.SINK, Ops.PROGRAM):
            return (call,)
        devices = next((b.device for b in call.src[1:] if isinstance(b.device, tuple)), None)
        if devices is None or not all(d.split(":")[0] == "DX12" for d in devices):
            return (call,)
        return tuple(_local_call(call, lane) for lane in range(len(devices)))

    return linear.replace(src=tuple(local for call in linear.src for local in expand(call)))


def register() -> None:
    """Register DX12 for explicit use and automatic device selection.

    Tinygrad tries DX12 first when no DEV override or cached default exists.
    If DX12 cannot open, automatic selection tries the other runtimes.
    Explicit DX12 requests report initialization errors instead of falling back.
    Registration opens no hardware and downloads no files. Repeated calls have
    no effect. Existing tensors and a previously selected default stay unchanged.
    The installed tinygrad files are not changed.
    """
    global _registered
    if _registered:
        return
    from tinygrad import codegen, schedule
    from tinygrad.device import ALL_DEVICES, Device
    from tinygrad.dtype import dtypes
    from tinygrad.engine import jit, realize
    from tinygrad.helpers import Context
    from tinygrad.renderer import Renderer
    from tinygrad.schedule import memory
    from tinygrad_dx12.codegen.matmul import lower_matmul
    from tinygrad_dx12.runtime.ops_dx12 import DX12Program
    from tinygrad_dx12.renderer.cstyle import DX12Renderer, dtype_of

    runtime = importlib.import_module("tinygrad_dx12.runtime.ops_dx12")
    sys.modules["tinygrad.runtime.ops_dx12"] = runtime
    if "DX12" not in Device._devices:
        Device._devices.append("DX12")
    original = realize.lower_and_compile
    original_plan = memory.memory_plan_rewrite
    original_jit = jit.jit_lower
    original_kernel = realize.exec_kernel
    original_rewrite = codegen.full_rewrite_to_sink
    kernels: weakref.WeakKeyDictionary[UOp, tuple] = weakref.WeakKeyDictionary()

    @wraps(original_rewrite)
    def full_rewrite_to_sink(ast: UOp, ren: Renderer, optimize: bool = True) -> UOp:
        """Let tinygrad reduce nonnative floats before narrowing the final maximum.

        Widening MAX preserves every representable input and avoids the shared
        dtype decomposition's invalid AFTER(CAST) accumulator. This runs inside
        tinygrad's compilation cache, with no per-dispatch normalization.
        """
        if isinstance(ren, DX12Renderer):
            ast = ast.substitute({
                u: u.replace(src=(u.src[0].cast(dtypes.float32), *u.src[1:])).cast(dtype_of(u))
                for u in ast.toposort()
                if u.op is Ops.REDUCE and u.arg[0] is Ops.MAX and u.dtype in (*dtypes.fp8s, dtypes.bfloat16)
            })
        return original_rewrite(ast, ren, optimize)

    @wraps(original_kernel)
    def exec_kernel(ctx: realize.ExecContext, call: UOp, ast: UOp, devices: tuple[str, ...] | None = None) -> list[float | None]:
        """Reuse DX12 call metadata and views while checking current storage on every launch.

        Input identity changes resolve the complete view templates again. Buffer's
        own storage-version check handles reallocation of an unchanged input UOp.
        Weak keys/references keep this cache from retaining calls or input graphs;
        active native storage remains owned by the runtime submission fence.
        """
        device = call.device
        if devices is not None or not ctx.cache or not isinstance(device, str) or device.split(":")[0] != "DX12":
            return original_kernel(ctx, call, ast, devices)
        cached = kernels.get(call)
        if cached is None or len(cached[0]) != len(ctx.input_uops) or any(ref() is not u for ref, u in zip(cached[0], ctx.input_uops)):
            args = realize.get_call_arg_uops(call)
            bufs = tuple(resolve_graph_input(args[i], ctx.input_uops) for i in ast.arg.globals)
            cached = (tuple(weakref.ref(u) for u in ctx.input_uops), bufs, realize.get_runtime(device, ast))
            kernels[call] = cached
        global_size, local_size = ast.arg.launch_dims(ctx.var_vals)
        program = cached[2]
        if not isinstance(program, DX12Program):
            raise RuntimeError("DX12 call resolved to another runtime")
        return [
            program(
                *(b.ensure_allocated().get_buf(device) for b in cached[1]),
                global_size=global_size,
                local_size=local_size,
                vals=ast.arg.vals(ctx.var_vals),
                wait=ctx.wait,
                timeout=ctx.timeout,
            )
        ]

    @wraps(original_plan)
    def memory_plan_rewrite(linear: UOp, held_bufs: set[UOp] | None = None) -> UOp:
        """Expose matrix conversions to tinygrad before its existing memory planner."""
        return original_plan(lower_matmul(split_dx12_calls(linear)), held_bufs)

    @wraps(original)
    def lower_and_compile(linear: UOp) -> UOp:
        """Use tinygrad's compiler after expanding adapter-specific calls."""
        return original(lower_matmul(split_dx12_calls(linear)))

    @wraps(original_jit)
    def jit_lower(linear: UOp, held_bufs: set[UOp], input_uops: list[UOp]) -> UOp:
        """Prove matrix buffer alignment before JIT replaces concrete inputs with parameters."""
        linear = lower_matmul(split_dx12_calls(linear), tuple(input_uops))
        with Context(DX12_LINALG=0):
            return original_jit(linear, held_bufs, input_uops)

    memory.memory_plan_rewrite = schedule.memory_plan_rewrite = jit.memory_plan_rewrite = memory_plan_rewrite
    realize.lower_and_compile = lower_and_compile
    jit.jit_lower = jit_lower
    realize.exec_kernel = exec_kernel
    codegen.full_rewrite_to_sink = full_rewrite_to_sink
    if "DX12" not in ALL_DEVICES:
        ALL_DEVICES.insert(0, "DX12")
    _registered = True
