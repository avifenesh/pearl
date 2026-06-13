"""Piecewise CUDA-graph support for the Pearl mining plugin (PR 0).

vLLM's v1 engine can capture *piecewise* CUDA graphs: the model's FX graph is
split at a set of declared "splitting ops", the subgraphs between splits are
captured/replayed, and the splitting ops themselves run eager. Attention works
exactly this way.

Pearl's mining GEMM does per-forward-pass host work (reads the current mining
job, schedules CUDA-event callbacks, allocates scratch tensors), which is not
CUDA-graph-replayable. Declaring the mining op as a splitting op lets the rest
of the model graph (norms, rotary, attention, small GEMMs, sampler) be captured
while the mining layers stay eager — recovering most of the ``--enforce-eager``
throughput tax with *zero* change to mining semantics.

This module is intentionally free of any CUDA/torch imports so the recipe can be
constructed and tested on a CPU-only host. The actual op registration lives in
the (GPU-only) kernel wrapper.
"""

from __future__ import annotations

# Registered custom-op name, in the ``namespace::opname`` form that vLLM's
# compiler matches against FX-graph call sites. Must equal the name the mining
# GEMM wrapper registers via ``torch.library.custom_op`` (see Task 0.4).
PEARL_MINING_SPLIT_OP = "vllm_miner::noisy_gemm_mining"


def pearl_splitting_ops() -> list[str]:
    """Return vLLM's default attention splitting ops plus the Pearl mining op.

    The defaults are preserved (deduplicated, order-stable) so attention still
    runs eager between captured subgraphs; the Pearl op is appended so the
    mining GEMM is also treated as a split point.
    """
    from vllm.config.compilation import CompilationConfig

    ops = list(CompilationConfig._attention_ops)
    if PEARL_MINING_SPLIT_OP not in ops:
        ops.append(PEARL_MINING_SPLIT_OP)
    return ops


def piecewise_compilation_config():
    """Build a ``CompilationConfig`` wired for piecewise capture with mining.

    vLLM only auto-populates ``splitting_ops`` when it is ``None``; by passing an
    explicit list (attention ops + the Pearl op) we keep both intact after
    ``set_splitting_ops_for_v1`` runs during config finalization.
    """
    from vllm.config.compilation import (
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
    )

    return CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        splitting_ops=pearl_splitting_ops(),
    )
