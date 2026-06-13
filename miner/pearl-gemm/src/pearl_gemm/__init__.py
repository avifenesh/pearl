"""
pearl_gemm package

This package provides CUDA kernels for Pearl GEMM with noising/denoising and PoW extraction.

On hosts without the compiled ``pearl_gemm_cuda`` extension (e.g. non-sm_90 dev
machines), a pure-Python reference backend is installed as ``pearl_gemm_cuda``
so the package still imports and the mining path runs (bit-exact for PoW, NOT
performance-representative). Force the reference with ``PEARL_GEMM_REFERENCE=1``.
"""

from . import _reference_cuda as _reference_cuda

# Install the reference backend if the real CUDA extension is unavailable
# (or when explicitly forced). The real extension always wins otherwise.
_reference_cuda.install()

# Re-export pearl_gemm_cuda utilities for cleaner API
from pearl_gemm_cuda import (
    HostSignalStatus,
    get_host_signal_header,
    get_host_signal_header_size,
    get_host_signal_sync_size,
    get_required_scratchpad_bytes,
    kEALScaleFactorDenoise,
    kEBRScaleFactorDenoise,
)

from . import pearl_gemm_interface
from .helpers import (
    HostSignalHeaderPinnedPool,
    ProofTileIndices,
    extract_indices,
    make_pow_target_tensor,
)
from .pearl_gemm_interface import (
    commitment_hash_from_merkle_roots,
    denoise_converter,
    gemm,
    noise_A,
    noise_B,
    noise_gen,
    noisy_gemm,
    quantize,
    tensor_hash,
)

__all__ = [
    "HostSignalHeaderPinnedPool",
    "HostSignalStatus",
    "ProofTileIndices",
    "commitment_hash_from_merkle_roots",
    "denoise_converter",
    "extract_indices",
    "gemm",
    "get_host_signal_header",
    "get_host_signal_header_size",
    "get_host_signal_sync_size",
    "get_required_scratchpad_bytes",
    "kEALScaleFactorDenoise",
    "kEBRScaleFactorDenoise",
    "make_pow_target_tensor",
    "noise_A",
    "noise_B",
    "noise_gen",
    "noisy_gemm",
    "pearl_gemm_interface",
    "quantize",
    "tensor_hash",
]
