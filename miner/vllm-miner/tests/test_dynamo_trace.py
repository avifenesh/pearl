"""CPU/fake-tensor traceability test for the mining-GEMM custom op (PR 0, Task 0.4).

For vLLM's piecewise compiler to treat the mining GEMM as a splitting op, the
call site must be a single registered custom op (``vllm_miner::noisy_gemm_mining``)
that the FX graph can see by name, and it must be traceable under FakeTensorMode
(no real CUDA needed) via its registered fake/meta implementation.

The op is **void**: it writes into a caller-provided output tensor ``c`` and
returns ``None`` (mirroring vLLM's attention splitting ops), so the piecewise
split codegen does not produce a non-int getitem on its result.

These tests run on CPU: they never invoke the real CUDA kernel, only the
registered fake, so they validate op registration + the void out-param shape.
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
    op = torch.ops.vllm_miner.noisy_gemm_mining
    assert op is not None


def test_mining_gemm_is_void_and_traceable_with_fake_tensors():
    """The void op runs under fake tensors writing into a caller-provided ``c``."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    m, n, k = 256, 512, 128
    with FakeTensorMode():
        a = torch.empty((m, k), dtype=torch.int8)
        b = torch.empty((n, k), dtype=torch.int8)
        scale_a = torch.empty((m,), dtype=torch.float32)
        scale_b = torch.empty((n,), dtype=torch.float32)
        c = torch.empty((m, n), dtype=torch.bfloat16)

        ret = torch.ops.vllm_miner.noisy_gemm_mining(a, b, scale_a, scale_b, c, True)

    # Void op: returns None, output is the caller-provided tensor.
    assert ret is None
    assert tuple(c.shape) == (m, n)
    assert c.dtype == torch.bfloat16


def test_op_appears_as_single_node_in_fx_graph():
    """The op is captured as one ``call_function`` node, not inlined Python.

    This is the property piecewise splitting relies on: the mining GEMM shows up
    as a single ``vllm_miner.noisy_gemm_mining`` call site in the FX graph. We
    trace with ``make_fx`` under fake tensors so no real CUDA kernel runs (the op
    body resolves to its registered fake), which is how vLLM's compiler sees the
    op when deciding split points.
    """
    from torch.fx.experimental.proxy_tensor import make_fx

    def f(a, b, sa, sb, c):
        torch.ops.vllm_miner.noisy_gemm_mining(a, b, sa, sb, c, True)
        return c

    m, n, k = 128, 256, 128
    a = torch.empty((m, k), dtype=torch.int8)
    b = torch.empty((n, k), dtype=torch.int8)
    sa = torch.empty((m,), dtype=torch.float32)
    sb = torch.empty((n,), dtype=torch.float32)
    c = torch.empty((m, n), dtype=torch.bfloat16)

    gm = make_fx(f, tracing_mode="fake")(a, b, sa, sb, c)

    targets = [str(node.target) for node in gm.graph.nodes if node.op == "call_function"]
    mining_nodes = [t for t in targets if "noisy_gemm_mining" in t]
    assert len(mining_nodes) == 1, targets
