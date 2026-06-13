"""CPU-only tests for the piecewise CUDA-graph recipe (PR 0).

These verify that the documented ``--compilation-config`` recipe keeps vLLM's
default attention splitting ops *and* adds Pearl's mining GEMM op, so that the
mining layers run eager (as splitting ops) while the rest of the model graph is
captured into piecewise CUDA graphs. No GPU is required: we only exercise the
``CompilationConfig`` plumbing.
"""

from vllm.config.compilation import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
)
from vllm_miner.piecewise import (
    PEARL_MINING_SPLIT_OP,
    pearl_splitting_ops,
    piecewise_compilation_config,
)


def test_pearl_split_op_name_is_namespaced():
    """The split-op name must be the registered ``namespace::opname`` form."""
    assert PEARL_MINING_SPLIT_OP == "vllm_miner::noisy_gemm_mining"


def test_pearl_splitting_ops_extends_attention_ops():
    """Helper returns the default attention ops plus the Pearl mining op."""
    ops = pearl_splitting_ops()
    # Pearl op present.
    assert PEARL_MINING_SPLIT_OP in ops
    # Default attention ops preserved (do not clobber vLLM's defaults).
    for attn_op in CompilationConfig._attention_ops:
        assert attn_op in ops
    # No duplicates.
    assert len(ops) == len(set(ops))


def test_piecewise_config_recipe_contains_pearl_split_op():
    """The recipe survives ``set_splitting_ops_for_v1`` with the Pearl op intact.

    vLLM only auto-populates ``splitting_ops`` when it is ``None``; a
    pre-populated list is preserved. This asserts the recipe is stable and that
    both the attention ops and the Pearl mining op remain after config
    finalization.
    """
    cc = piecewise_compilation_config()
    assert cc.mode == CompilationMode.VLLM_COMPILE
    assert cc.cudagraph_mode == CUDAGraphMode.PIECEWISE

    cc.set_splitting_ops_for_v1(all2all_backend="naive", data_parallel_size=1)

    assert PEARL_MINING_SPLIT_OP in cc.splitting_ops
    assert "vllm::unified_attention_with_output" in cc.splitting_ops
