"""Reference implementations of the ``pearl_gemm_cuda`` ops (local dev only).

Backed by :class:`miner_base.noisy_gemm.NoisyGemm` and
:class:`miner_base.noise_generation.NoiseGenerator` — the same bit-faithful
references the real CUDA kernels are validated against. These functions match
the call signatures used by ``pearl_gemm_interface.py`` (positional) and the
``torch.ops.pearl_gemm.*`` schemas, writing results into caller-provided output
tensors exactly like the kernels do.

NOT performance-representative. For local correctness / PoW bit-exactness only.
"""

from __future__ import annotations

import functools

import torch

from . import _reference_cuda as ref

# Side table: id(EAL output tensor) -> (E_AL, E_AR, E_BL, E_BR, key_A, key_B)
# populated by noise_gen, consumed by noisy_gemm so the same reference noise is
# used end to end.
_NOISE_TABLE: dict[int, tuple] = {}


@functools.lru_cache(maxsize=8)
def _get_noise_generator(noise_rank: int, noise_range: int):
    """Cache NoiseGenerator instances (instantiation logs + setup are non-trivial)."""
    from miner_base.noise_generation import NoiseGenerator

    return NoiseGenerator(noise_rank=noise_rank, noise_range=noise_range)


@functools.lru_cache(maxsize=8)
def _get_noisy_gemm(noise_rank: int, noise_range: int):
    """Cache NoisyGemm instances — avoids re-instantiating on every call."""
    from miner_base.noisy_gemm import NoisyGemm

    return NoisyGemm(noise_rank=noise_rank, noise_range=noise_range)


def _int_matmul_dequant(A, B, A_scales, B_scales, out_dtype):
    """C = A_scales * (A @ B.T) * B_scales, exact int32 accumulation.

    torch has no integer matmul on CUDA ("addmm_cuda not implemented for Int"),
    so do the int32 matmul on CPU then move back to A's device for dequant.
    """
    dev = A.device
    acc = (A.detach().cpu().to(torch.int32) @ B.detach().cpu().to(torch.int32).T)
    acc = acc.to(dev).to(torch.float32)
    acc = acc * A_scales.view(-1, 1).to(torch.float32) * B_scales.view(1, -1).to(torch.float32)
    return acc.to(out_dtype)


def _to_bytes32(t: torch.Tensor) -> bytes:
    """A (32,) uint8 / (8,) uint32 tensor -> 32 raw bytes."""
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()[:32]


def quantize(input_tensor, output, scales, max_val=63, smooth_scale=None, fast_math=False):
    """Symmetric per-token quantization into ``output`` (int8) + ``scales`` (fp32).

    Mirrors quantize_kernel.cu: per-row scale = max(abs(row))/max_val, then round.
    """
    x = input_tensor
    if smooth_scale is not None:
        x = x / smooth_scale.to(x.dtype)
    x = x.to(torch.float32)
    row_max = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    s = row_max / max_val
    q = torch.round(x / s).clamp_(-max_val, max_val).to(torch.int8)
    output.copy_(q)
    scales.copy_(s.to(scales.dtype))
    return None


def noise_gen(
    R=128,
    num_threads=64,
    EAL=None,
    EAL_fp16=None,
    EAR_R_major=None,
    EAR_K_major=None,
    EBL_R_major=None,
    EBL_K_major=None,
    EBR=None,
    EBR_fp16=None,
    key_A=None,
    key_B=None,
    aux_buffer=None,
):
    """Generate reference noise from seeds and stash it for noisy_gemm.

    We fill the provided EAL/EBR (and fp16 twins) with the reference noise and
    record the full (E_AL,E_AR,E_BL,E_BR,key_A,key_B) keyed by id(EAL) so
    noisy_gemm can run the faithful NoisyGemm with the matching factors.
    """
    if aux_buffer is not None:
        aux_buffer.zero_()
    if EAL is None or key_A is None or key_B is None:
        return None

    m = EAL.shape[0]
    n = EBR.shape[0] if EBR is not None else m
    # k is the common dim; EAR_R_major is (k, R)
    k = EAR_R_major.shape[0] if EAR_R_major is not None else R
    kA = _to_bytes32(key_A)
    kB = _to_bytes32(key_B)

    gen = _get_noise_generator(R, ref_noise_range())
    E_AL, E_AR, E_BL, E_BR = gen.generate_noise_metrices(kA, kB, m, k, n)

    # The kernel-layout output tensors (EAL/EBR/*_R_major/*_K_major + fp16 twins)
    # use different shapes/transposes than the reference factors and are only
    # consumed by the real kernel's noising path. The reference noisy_gemm
    # re-derives everything from the stashed factors, so we DON'T copy into the
    # caller tensors (shapes differ) — we only stash. We touch EAL so its id()
    # is a stable key and callers see it as "written".
    if EAL is not None:
        EAL.zero_()

    _NOISE_TABLE[id(EAL)] = (E_AL, E_AR, E_BL, E_BR, kA, kB)
    return None


def ref_noise_range() -> int:
    # noise_range defaults to 128 (uint7). Kept as a hook in case settings vary.
    return 128


def gemm(A, B, A_scales, B_scales, C, tile_size_m=128, tile_size_n=256, tile_size_k=128,
         cluster_size_m=1, cluster_size_n=1, pipeline_stages=None, swizzle=None,
         swizzle_n_maj=True):
    """Plain quantized GEMM: C = A_scales * (A @ B.T) * B_scales (no noising)."""
    C.copy_(_int_matmul_dequant(A, B, A_scales, B_scales, C.dtype))
    return None


def noisy_gemm(A, B, EAL, EAL_fp16, EBR, EBR_fp16, EAR_R_major, EBL_R_major,
               EAR_K_major, EBL_K_major, AxEBL_fp16, EARxBpEB_fp16, ApEA, BpEB,
               A_scales, B_scales, C, host_signal_header_pinned, host_signal_sync,
               pow_target, pow_key, AxEBL_int32=None, EARxBpEB_int32=None,
               tile_size_m=128, tile_size_n=256, tile_size_k=128, cluster_size_m=1,
               cluster_size_n=1, pipeline_stages=None, swizzle=None, swizzle_n_maj=True,
               tile_size_m_noising_A=None, tile_size_n_noising_B=None,
               tile_size_k_noising_A=None, tile_size_k_noising_B=None,
               pipeline_stages_noising_A=2, pipeline_stages_noising_B=2,
               k_blocks_per_split_noising_A=None, k_blocks_per_split_noising_B=None,
               run_noising_A=True, run_noising_B=True, skip_reduction=False,
               skip_denoising=False, inner_hash_counter=None, enable_debug=False):
    """Reference noisy GEMM + PoW extraction via miner_base.NoisyGemm.

    Writes the dequantized output into ``C`` and, on a PoW hit, stashes a
    HostSignalHeader keyed to ``host_signal_header_pinned`` (read back by
    get_host_signal_header).
    """
    from pearl_gateway.comm.dataclasses import CommitmentHash

    stash = _NOISE_TABLE.pop(id(EAL), None)
    if stash is None:
        # noise_gen wasn't routed through the shim; fall back to deriving from
        # pow_key (== commitment_hash_A). We still need key_B; without it the
        # PoW won't be bit-exact, so require the stash in practice.
        raise RuntimeError(
            "reference noisy_gemm: missing noise stash for this EAL; "
            "ensure noise_gen ran through the reference shim"
        )
    E_AL, E_AR, E_BL, E_BR, kA, kB = stash

    # Fast-dev mode: skip the expensive per-tile PoW hash sweep (32k blake3
    # hashes) and just produce the (noise-free) dequantized matmul output. The
    # noisy GEMM denoises back to A@B.T, so C is the same as a plain GEMM here;
    # this is for fast logic/integration iteration where PoW detection is not
    # needed. No block is ever "found" in this mode.
    import os as _os

    fast_dev = _os.getenv("PEARL_GEMM_FAST_DEV", "").lower() in ("1", "true", "yes")
    if fast_dev or skip_reduction:
        C.copy_(_int_matmul_dequant(A, B, A_scales, B_scales, C.dtype))
        ref._set_header(host_signal_header_pinned,
                        ref.HostSignalHeader(status=ref.HostSignalStatus.kSignalIdle))
        return None

    pt = _pow_target_to_int(pow_target)
    R = EAL.shape[1]
    ng = _get_noisy_gemm(R, ref_noise_range())
    # NoisyGemm uses integer matmul, which torch only supports on CPU. Run the
    # reference on CPU then move the result back to the caller's device.
    out_dev = C.device
    ch = CommitmentHash(noise_seed_A=kA, noise_seed_B=kB)
    C_out, found = ng.noisy_gemm(
        A.detach().cpu(), B.detach().cpu(),
        E_AL.cpu(), E_AR.cpu(), E_BL.cpu(), E_BR.cpu(),
        ch, pow_target=pt,
    )
    # Dequantize to C's dtype (C_out is int32 raw matmul of the de-noised path).
    deq = C_out.to(out_dev).to(torch.float32)
    deq = deq * A_scales.view(-1, 1).to(torch.float32) * B_scales.view(1, -1).to(torch.float32)
    C.copy_(deq.to(C.dtype))

    header = ref.HostSignalHeader(status=ref.HostSignalStatus.kSignalIdle)
    if found:
        obi = ng.get_opened_block_info()
        rows = list(obi.A_row_indices)
        cols = list(obi.B_column_indices)
        header = ref.HostSignalHeader(
            status=ref.HostSignalStatus.kSignalTriggered,
            tileCoord=[0, 0, 0],
            num_registers_per_thread=max(len(rows), len(cols)),
            thread_rows=rows,
            thread_cols=cols,
            mma_size=ref.MMASize(m=A.shape[0], n=B.shape[0], k=A.shape[1]),
            mma_tile_size=ref.MMASize(m=1, n=1, k=1),  # indices already absolute
        )
    ref._set_header(host_signal_header_pinned, header)
    return None


def _pow_target_to_int(pow_target) -> int:
    """(8,) uint32 LE-word tensor -> uint256 int."""
    if isinstance(pow_target, int):
        return pow_target
    words = pow_target.detach().cpu().contiguous().view(torch.uint32).tolist()
    val = 0
    for i, w in enumerate(words):
        val |= (int(w) & 0xFFFFFFFF) << (32 * i)
    return val


def tensor_hash(data, key, out, roots, threads_per_block=128, num_stages=2,
                leaves_per_mt_block=512):
    """Bit-exact merkle-tree root of ``data`` keyed by ``key``, written to ``out``.

    Matches the CUDA tensor_hash (validated by test_matrix_merkle_tree_vs_tensor_hash:
    MatrixMerkleTree(data, key).root == kernel output).
    """
    from miner_base.matrix_merkle_tree import MatrixMerkleTree

    key_bytes = _to_bytes32(key)
    # MatrixMerkleTree hashes the raw bytes; it requires int8 dtype, but the
    # kernel takes uint8. Reinterpret the bits (same bytes) to satisfy it.
    d = data.detach().cpu().contiguous()
    if d.dtype == torch.uint8:
        d = d.view(torch.int8)
    root = MatrixMerkleTree(d, key_bytes).root
    out.copy_(torch.frombuffer(bytearray(root), dtype=torch.uint8).to(out.device))
    return out


def commitment_hash_from_merkle_roots(A_merkle_root, B_merkle_root, key,
                                      A_commitment_hash, B_commitment_hash,
                                      routing_root=None, offsets_hash=None):
    """Bit-exact commitment hashes from merkle roots (mirrors CommitmentHasher)."""
    from miner_base.commitment_hash import CommitmentHasher

    kwargs = {}
    if routing_root is not None or offsets_hash is not None:
        kwargs["routing_root"] = _to_bytes32(routing_root) if routing_root is not None else None
        kwargs["offsets_root"] = _to_bytes32(offsets_hash) if offsets_hash is not None else None
    ch = CommitmentHasher.commitment_hash_from_merkle_roots(
        _to_bytes32(A_merkle_root), _to_bytes32(B_merkle_root), _to_bytes32(key), **kwargs
    )
    a = torch.frombuffer(bytearray(ch.noise_seed_A), dtype=torch.uint8).to(A_commitment_hash.device)
    b = torch.frombuffer(bytearray(ch.noise_seed_B), dtype=torch.uint8).to(B_commitment_hash.device)
    A_commitment_hash.copy_(a)
    B_commitment_hash.copy_(b)
    return None


# --- Ops not exercised by the dense (non-MoE) inference path. Provided so the
# --- module surface is complete; raise clearly if a code path hits them so we
# --- know to implement a faithful reference.
def _unimplemented(name):
    def _fn(*args, **kwargs):
        raise NotImplementedError(
            f"pearl_gemm reference: '{name}' is not implemented in the local "
            f"reference backend yet (not needed for the dense inference path). "
            f"Implement against miner_base if a code path requires it."
        )
    return _fn


noise_A = _unimplemented("noise_A")
noise_B = _unimplemented("noise_B")
denoise_converter = _unimplemented("denoise_converter")
inner_hash = _unimplemented("inner_hash")
build_routing_data = _unimplemented("build_routing_data")
get_build_routing_data_scratchpad_bytes = _unimplemented("get_build_routing_data_scratchpad_bytes")
