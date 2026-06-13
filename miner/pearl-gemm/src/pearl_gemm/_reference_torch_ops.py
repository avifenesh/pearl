"""Register the ``pearl_gemm::*`` torch library ops in pure Python.

The C++ extension defines these ops via ``TORCH_LIBRARY(pearl_gemm, ...)``.
Without the extension, ``pearl_gemm_interface.py``'s ``register_fake`` calls
fail because the ops don't exist. This module declares the same op schemas
(copied verbatim from ``csrc/gemm/pearl_gemm_api.cpp``) and binds CPU/CUDA
implementations to the reference ops, so the package imports and runs.
"""

from __future__ import annotations

import torch

from . import _reference_ops as ops

# Schemas copied verbatim from csrc/gemm/pearl_gemm_api.cpp TORCH_LIBRARY block.
_SCHEMAS = {
    "noisy_gemm": (
        "noisy_gemm(Tensor A, Tensor B, Tensor EAL, Tensor? EAL_fp16, Tensor EBR, "
        "Tensor? EBR_fp16, Tensor? EAR_R_major, Tensor? EBL_R_major, Tensor? EAR_K_major, "
        "Tensor? EBL_K_major, Tensor(AxEBL_fp16!) AxEBL_fp16, Tensor(EARxBpEB_fp16!) EARxBpEB_fp16, "
        "Tensor(ApEA!) ApEA, Tensor(BpEB!) BpEB, Tensor A_scales, Tensor B_scales, Tensor(C!) C, "
        "Tensor(host_signal_header_pinned!) host_signal_header_pinned, "
        "Tensor(host_signal_sync!) host_signal_sync, Tensor pow_target, Tensor pow_key, "
        "Tensor(AxEBL_int32!)? AxEBL_int32 = None, Tensor(EARxBpEB_int32!)? EARxBpEB_int32 = None, "
        "int tile_size_m = 128, int tile_size_n = 256, int tile_size_k = 128, "
        "int cluster_size_m = 1, int cluster_size_n = 1, int? pipeline_stages = None, "
        "int? swizzle = None, bool swizzle_n_maj = True, int? tile_size_m_noising_A = None, "
        "int? tile_size_n_noising_B = None, int? tile_size_k_noising_A = None, "
        "int? tile_size_k_noising_B = None, int pipeline_stages_noising_A = 2, "
        "int pipeline_stages_noising_B = 2, int? k_blocks_per_split_noising_A = None, "
        "int? k_blocks_per_split_noising_B = None, bool run_noising_A = True, "
        "bool run_noising_B = False, bool skip_reduction = True, bool skip_denoising = False, "
        "Tensor(inner_hash_counter!)? inner_hash_counter = None, bool enable_debug = False) -> ()"
    ),
    "gemm": (
        "gemm(Tensor A, Tensor B, Tensor A_scales, Tensor B_scales, Tensor(C!) C, "
        "int tile_size_m = 128, int tile_size_n = 256, int tile_size_k = 128, "
        "int cluster_size_m = 1, int cluster_size_n = 1, int? pipeline_stages = None, "
        "int? swizzle = None, bool swizzle_n_maj = True) -> ()"
    ),
    "noise_A": (
        "noise_A(Tensor A, Tensor EAL, Tensor(AxEBL!) AxEBL, Tensor(ApEA!) ApEA, "
        "Tensor? EAR, Tensor? EBL, int? tile_size_m = None, int? tile_size_k = None, "
        "int pipeline_stages = 2, int? k_blocks_per_split = None) -> ()"
    ),
    "noise_B": (
        "noise_B(Tensor B, Tensor EBR, Tensor(EARxBpEB!) EARxBpEB, Tensor(BpEB!) BpEB, "
        "Tensor? EAR, Tensor? EBL, int? tile_size_n = None, int? tile_size_k = None, "
        "int pipeline_stages = 2, int? k_blocks_per_split = None) -> ()"
    ),
    "denoise_converter": (
        "denoise_converter(Tensor? EARxBpEB_in, Tensor? AxEBL_in, "
        "Tensor(EARxBpEB_out!)? EARxBpEB_out, Tensor(AxEBL_out!)? AxEBL_out) -> ()"
    ),
    "noise_gen": (
        "noise_gen(int R, int num_threads = 64, Tensor(EAL!)? EAL = None, "
        "Tensor(EAL_fp16!)? EAL_fp16 = None, Tensor(EAR_R_major!)? EAR_R_major = None, "
        "Tensor(EAR_K_major!)? EAR_K_major = None, Tensor(EBL_R_major!)? EBL_R_major = None, "
        "Tensor(EBL_K_major!)? EBL_K_major = None, Tensor(EBR!)? EBR = None, "
        "Tensor(EBR_fp16!)? EBR_fp16 = None, Tensor? key_A = None, Tensor? key_B = None, "
        "Tensor(aux_buffer!)? aux_buffer = None) -> ()"
    ),
    "inner_hash": "inner_hash(Tensor input_buffer, int iterations = 1) -> Tensor",
    "tensor_hash": (
        "tensor_hash(Tensor data, Tensor key, Tensor(out!) out, Tensor(roots!) roots, "
        "int num_threads = 128, int num_stages = 2, int leaves_per_mt_block = 512) -> ()"
    ),
    "quantize": (
        "quantize(Tensor input, Tensor(output!) output, Tensor(scales!) scales, "
        "int max_val = 63, Tensor? smooth_scale = None, bool fast_math = False) -> ()"
    ),
    "commitment_hash_from_merkle_roots": (
        "commitment_hash_from_merkle_roots(Tensor A_merkle_root, Tensor B_merkle_root, "
        "Tensor key, Tensor(A_commitment_hash!) A_commitment_hash, "
        "Tensor(B_commitment_hash!) B_commitment_hash, Tensor? routing_root = None, "
        "Tensor? offsets_hash = None) -> ()"
    ),
    "build_routing_data": (
        "build_routing_data(Tensor topk_ids, Tensor(routing_data!) routing_data, "
        "Tensor(slot_indices!) slot_indices, Tensor(routing_offsets!) routing_offsets, "
        "Tensor(scratchpad!) scratchpad, int num_experts) -> ()"
    ),
}

_registered = False


def register() -> bool:
    """Define pearl_gemm::* ops + bind reference impls. Idempotent. No-op if real ext present."""
    global _registered
    if _registered:
        return True
    # If the real library already defined these (extension loaded), do nothing.
    if hasattr(torch.ops, "pearl_gemm") and hasattr(torch.ops.pearl_gemm, "noisy_gemm"):
        try:
            torch.ops.pearl_gemm.noisy_gemm  # noqa: B018
            _registered = True
            return False
        except Exception:
            pass

    lib = torch.library.Library("pearl_gemm", "DEF")
    impl = torch.library.Library("pearl_gemm", "IMPL")
    for name, schema in _SCHEMAS.items():
        lib.define(schema)
        fn = getattr(ops, name)
        # Bind for CPU and CUDA so calls dispatch to the reference regardless of
        # input device (reference itself moves tensors as needed).
        impl.impl(name, fn, "CompositeExplicitAutograd")

    # Keep refs alive for the process lifetime.
    register._lib = lib
    register._impl = impl
    _registered = True
    return True
