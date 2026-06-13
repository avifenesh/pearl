"""Pearl mining plugin for vLLM.

Top-level attributes (``register_pearl_miner_layer``, ``PearlKernel``,
``pearl_gemm_noisy``, ``pearl_gemm_vanilla``) are imported lazily so that
pure-Python helpers in this package (e.g. :mod:`vllm_miner.piecewise`) can be
imported on hosts without the compiled ``pearl_gemm`` CUDA extension. The heavy
imports only fire when one of these names is actually accessed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "register_pearl_miner_layer",
    "PearlKernel",
    "pearl_gemm_vanilla",
    "pearl_gemm_noisy",
]

if TYPE_CHECKING:
    from .gemm_operators import pearl_gemm_noisy, pearl_gemm_vanilla
    from .register import register_pearl_miner_layer
    from .vllm_kernels import PearlKernel

_LAZY = {
    "register_pearl_miner_layer": ("register", "register_pearl_miner_layer"),
    "PearlKernel": ("vllm_kernels", "PearlKernel"),
    "pearl_gemm_noisy": ("gemm_operators", "pearl_gemm_noisy"),
    "pearl_gemm_vanilla": ("gemm_operators", "pearl_gemm_vanilla"),
}


def __getattr__(name: str):
    """PEP 562 lazy attribute access for the heavy (CUDA-dependent) exports."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{target[0]}", __name__)
    return getattr(module, target[1])


def __dir__() -> list[str]:
    return sorted(__all__)
