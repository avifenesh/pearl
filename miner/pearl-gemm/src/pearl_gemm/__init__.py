"""
pearl_gemm package

This package provides CUDA kernels for Pearl GEMM with noising/denoising and PoW extraction.
"""

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
    bpeb_ear_product,
    commitment_hash_from_merkle_roots,
    denoise_converter,
    gemm,
    noise_A,
    noise_B,
    noise_B_side_product,
    noise_gen,
    noisy_gemm,
    quantize,
    tensor_hash,
)

__all__ = [
    "HostSignalHeaderPinnedPool",
    "HostSignalStatus",
    "ProofTileIndices",
    "bpeb_ear_product",
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
    "noise_B_side_product",
    "noise_gen",
    "noisy_gemm",
    "pearl_gemm_interface",
    "quantize",
    "tensor_hash",
]
