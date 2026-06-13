"""Single custom op wrapping the mining GEMM, for piecewise CUDA-graph splitting.

vLLM's piecewise compiler splits the model FX graph at declared op names. For
the mining GEMM to be a split point it must appear in the graph as one
registered custom op. This module registers ``vllm_miner::noisy_gemm_mining``
and routes it to the existing :func:`vllm_miner.gemm_operators.pearl_gemm_noisy`
implementation.

Mutation contract (intentional, see PR plan Task 0.4): the op's *tensor
arguments* (``a``, ``b``, ``scale_a``, ``scale_b``) are read-only, so
``mutates_args=()`` is correct. All output/scratch tensors are allocated
*inside* ``pearl_gemm_noisy`` and never passed in, so the op's only
graph-visible effect is its returned tensor ``C``. The side effects on global
mining state and pinned host buffers are deliberately opaque to the compiler;
that is fine because this op is declared a *splitting op* and runs eager.

The real implementation imports the CUDA-backed kernel lazily so this module can
be imported (and the op registered + fake-traced) on a CPU-only host.
"""

from __future__ import annotations

import torch

# Must match vllm_miner.piecewise.PEARL_MINING_SPLIT_OP.
PEARL_MINING_SPLIT_OP = "vllm_miner::noisy_gemm_mining"


@torch.library.custom_op(PEARL_MINING_SPLIT_OP, mutates_args=())
def noisy_gemm_mining(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    submit_block: bool,
) -> torch.Tensor:
    """Quantized noisy GEMM with proof-of-work mining: ``C = a @ b.T``.

    Args:
        a: ``(m, k)`` int8 quantized activations.
        b: ``(n, k)`` int8 quantized weights.
        scale_a: ``(m,)`` fp32 activation scales.
        scale_b: ``(n,)`` fp32 weight scales.
        out_dtype: output dtype (bf16 or fp16).
        submit_block: whether to schedule block submission on a PoW hit.

    Returns:
        ``(m, n)`` output tensor in ``out_dtype``.
    """
    # Lazy import: keeps this module importable without the compiled CUDA ext.
    from .gemm_operators import pearl_gemm_noisy

    return pearl_gemm_noisy(
        a,
        b,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
        layer=None,
        submit_block=submit_block,
    )


@noisy_gemm_mining.register_fake
def _noisy_gemm_mining_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    submit_block: bool,
) -> torch.Tensor:
    """Shape/dtype propagation for compile/trace: ``(m,k) x (n,k) -> (m,n)``."""
    m = a.shape[0]
    n = b.shape[0]
    return torch.empty((m, n), dtype=out_dtype, device=a.device)
