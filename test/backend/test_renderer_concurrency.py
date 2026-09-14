"""A device-local renderer must preserve independent concurrent compilation state."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from tinygrad import dtypes
from tinygrad.helpers import Target
from tinygrad.uop.ops import KernelInfo, UOp
from tinygrad_dx12.renderer.cstyle import DX12Renderer


def test_shared_renderer_concurrent_sources() -> None:
    """Concurrent callers receive exactly their sequential shader, including dtype-specific helpers."""
    renderer = DX12Renderer(Target("DX12", arch='{"doubles":true,"int64":true,"native16":true}'))
    cases = []
    for slot in range(8):
        dtype = (dtypes.float32, dtypes.float64, dtypes.int32, dtypes.float16)[slot % 4]
        zero = UOp.const(0, dtypes.int32)
        source, output = UOp.param(1, dtype, slot + 1, "DX12"), UOp.param(0, dtype, slot + 1, "DX12")
        value = loaded = source.index(zero).load()
        for _ in range(2048):
            value = value * value + loaded
        cases.append(list(output.index(zero).store(value).sink(arg=KernelInfo(name=f"thread_{slot}")).toposort()))
    expected = [renderer.render(uops) for uops in cases]
    barrier = Barrier(len(cases))

    def render(slot: int) -> str:
        """Start independent renders together without changing thread scheduling or adding sleeps."""
        barrier.wait()
        try:
            return renderer.render(cases[slot])
        except Exception as error:
            return type(error).__name__

    with ThreadPoolExecutor(max_workers=len(cases)) as workers:
        actual = list(workers.map(render, range(len(cases))))
    assert all(result == reference for result, reference in zip(actual, expected))
