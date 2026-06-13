"""CPU/fake-tensor traceability test for the mining-GEMM custom op (PR 0, Task 0.4).

For vLLM's piecewise compiler to treat the mining GEMM as a splitting op, the
call site must be a single registered custom op (``vllm_miner::noisy_gemm_mining``)
that the FX graph can see by name, and it must be traceable under FakeTensorMode
(no real CUDA needed) via its registered fake/meta implementation.

These tests run on CPU: they never invoke the real CUDA kernel, only the
registered fake, so they validate op registration + shape/dtype propagation.
"""

import torch

# Importing the wrapper module registers the custom op as a side effect.
from vllm_miner.noisy_gemm_op import (
    PEARL_MINING_SPLIT_OP,
    noisy_gemm_mining,  # noqa: F401  (import ensures registration)
)


def test_op_is_registered_under_expected_name():
    """The op must exist in the torch op registry under the split-op name."""
    assert PEARL_MINING_SPLIT_OP == "vllm_miner::noisy_gemm_mining"
    # torch.ops.vllm_miner.noisy_gemm_mining must resolve.
    op = torch.ops.vllm_miner.noisy_gemm_mining
    assert op is not None


def test_mining_gemm_traceable_with_fake_tensors():
    """The fake impl propagates shape/dtype: (m,k) x (n,k) -> (m,n) in out_dtype."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    m, n, k = 256, 512, 128
    with FakeTensorMode():
        a = torch.empty((m, k), dtype=torch.int8)
        b = torch.empty((n, k), dtype=torch.int8)
        scale_a = torch.empty((m,), dtype=torch.float32)
        scale_b = torch.empty((n,), dtype=torch.float32)

        out = torch.ops.vllm_miner.noisy_gemm_mining(a, b, scale_a, scale_b, torch.bfloat16, True)

    assert tuple(out.shape) == (m, n)
    assert out.dtype == torch.bfloat16


def test_op_appears_as_single_node_in_fx_graph():
    """The op is captured as one ``call_function`` node, not inlined Python.

    This is the property piecewise splitting relies on: the mining GEMM shows up
    as a single ``vllm_miner.noisy_gemm_mining`` call site in the FX graph. We
    trace with ``make_fx`` under fake tensors so no real CUDA kernel runs (the op
    body resolves to its registered fake), which is exactly how vLLM's compiler
    sees the op when deciding split points.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    def f(a, b, sa, sb):
        return torch.ops.vllm_miner.noisy_gemm_mining(a, b, sa, sb, torch.bfloat16, True)

    m, n, k = 128, 256, 128
    a = torch.empty((m, k), dtype=torch.int8)
    b = torch.empty((n, k), dtype=torch.int8)
    sa = torch.empty((m,), dtype=torch.float32)
    sb = torch.empty((n,), dtype=torch.float32)

    gm = make_fx(f, tracing_mode="fake")(a, b, sa, sb)

    targets = [str(n.target) for n in gm.graph.nodes if n.op == "call_function"]
    mining_nodes = [t for t in targets if "noisy_gemm_mining" in t]
    assert len(mining_nodes) == 1, targets
