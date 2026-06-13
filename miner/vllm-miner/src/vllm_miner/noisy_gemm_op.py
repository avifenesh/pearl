"""Single custom op wrapping the mining GEMM, for piecewise CUDA-graph splitting.

vLLM's piecewise compiler splits the model FX graph at declared op names. For
the mining GEMM to be a split point it must appear in the graph as one
registered custom op. This module registers ``vllm_miner::noisy_gemm_mining``
and routes it to the existing :func:`vllm_miner.gemm_operators.pearl_gemm_noisy`
implementation.

Op shape (IMPORTANT): this is a **void** op that writes its result into a
caller-provided output tensor ``c`` and returns ``None`` — mirroring vLLM's own
attention splitting ops (``vllm::unified_attention_with_output`` etc.), which all
write an out-tensor and return nothing. vLLM 0.20's piecewise split codegen
(``vllm/compilation/codegen.py``) wires submodule results by integer index and
asserts ``isinstance(index, int)``; a splitting op that *returns* a tensor
produces a non-int getitem and trips that assert. A void out-param op avoids it.

Mutation contract: ``c`` is declared mutated (``mutates_args=("c",)``); the
input tensors are read-only. Side effects on global mining state and pinned host
buffers are deliberately opaque to the compiler — fine, since this op runs eager
as a splitting op.

The real implementation imports the CUDA-backed kernel lazily so this module can
be imported (and the op registered + fake-traced) on a CPU-only host.
"""

from __future__ import annotations

import torch

# Must match vllm_miner.piecewise.PEARL_MINING_SPLIT_OP.
PEARL_MINING_SPLIT_OP = "vllm_miner::noisy_gemm_mining"


@torch.library.custom_op(PEARL_MINING_SPLIT_OP, mutates_args=("c",))
def noisy_gemm_mining(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    c: torch.Tensor,
    submit_block: bool,
) -> None:
    """Quantized noisy GEMM with proof-of-work mining: writes ``c = a @ b.T``.

    Args:
        a: ``(m, k)`` int8 quantized activations.
        b: ``(n, k)`` int8 quantized weights.
        scale_a: ``(m,)`` fp32 activation scales.
        scale_b: ``(n,)`` fp32 weight scales.
        c: ``(m, n)`` preallocated output tensor (written in place).
        submit_block: whether to schedule block submission on a PoW hit.
    """
    # Lazy import: keeps this module importable without the compiled CUDA ext.
    from .gemm_operators import pearl_gemm_noisy

    pearl_gemm_noisy(
        a,
        b,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=c.dtype,
        layer=None,
        submit_block=submit_block,
        out=c,
    )


@noisy_gemm_mining.register_fake
def _noisy_gemm_mining_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    c: torch.Tensor,
    submit_block: bool,
) -> None:
    """Void op: no return value (out tensor ``c`` is provided by the caller)."""
    return None


def noisy_gemm_mining_out(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    submit_block: bool,
) -> torch.Tensor:
    """Allocate the output tensor and invoke the void mining op, returning ``c``.

    Convenience wrapper for the call site so the linear layer keeps a functional
    ``C = f(...)`` shape while the registered op stays void (graph-capturable).
    """
    c = torch.empty((a.shape[0], b.shape[0]), dtype=out_dtype, device=a.device)
    torch.ops.vllm_miner.noisy_gemm_mining(a, b, scale_a, scale_b, c, submit_block)
    return c
