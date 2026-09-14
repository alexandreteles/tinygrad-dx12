"""Register the DX12 runtime on import, without opening hardware.

Import this package before creating tensors. Tinygrad then tries DX12 first
when it selects a default device. An explicit DEV setting takes precedence.
Use Tensor(..., device="DX12") to require DX12 for one tensor.
"""

from tinygrad_dx12.compat import register as register

register()


def main() -> None:
    """Print the installed toolchain and adapter diagnostics."""
    from tinygrad_dx12.diagnostics import main as diagnostics

    diagnostics()
