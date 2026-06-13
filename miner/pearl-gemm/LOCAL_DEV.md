# Local development without the sm_90 kernels

The `pearl-gemm` CUDA kernels are **sm_90a (Hopper) only**. On any other GPU
(e.g. a Blackwell `sm_120` dev box) the compiled `pearl_gemm_cuda` extension
cannot be built or imported. A pure-Python **reference backend** fills that gap
so the whole miner plugin imports and runs locally for development.

## How it works

`pearl_gemm/__init__.py` calls `_reference_cuda.install()` before importing
`pearl_gemm_cuda`. When the real extension is absent (or you force it), a
reference module is installed as `pearl_gemm_cuda`, backed by the bit-faithful
references in `miner_base` (`NoisyGemm`, `NoiseGenerator`, `MatrixMerkleTree`,
`CommitmentHasher`) — the same references the real kernels are validated against.

- **Auto:** activates when `import pearl_gemm_cuda` fails.
- **Force:** `PEARL_GEMM_REFERENCE=1`.

Files: `_reference_cuda.py` (types/sizes/install), `_reference_ops.py` (the ops),
`_reference_torch_ops.py` (declares `pearl_gemm::*` torch ops so `register_fake`
in `pearl_gemm_interface.py` works).

## Setup

```bash
# from repo root; skip the sm_90 CUDA build, install everything else
PEARL_GEMM_SKIP_CUDA_BUILD=TRUE uv sync --all-packages
```

## What runs locally

- `import pearl_gemm` and the full `vllm_miner` plugin import.
- `PearlKernel.apply_weights` (mining + non-mining), `pearl_gemm_noisy`,
  `tensor_hash`, `commitment_hash_from_merkle_roots`, `noise_gen` — full chain,
  including PoW-hit detection with valid winning A-row/B-column indices.
- `pytest miner/vllm-miner/tests/test_vllm_interface.py` → 19/22 pass.
  The 3 failures call `vllm_ops.cutlass_scaled_mm` (Int8 GEMM, SM<100 only) as
  their comparison reference and cannot run on Blackwell — a test-harness arch
  limit, not a backend issue.

## Modes / speed

The reference runs the int matmul + per-tile BLAKE3 PoW sweep on CPU
(~610 ms at 1024³). For fast logic/integration iteration where PoW detection is
not needed:

```bash
PEARL_GEMM_FAST_DEV=1   # skip the ~32k-hash PoW sweep -> ~3x faster (213 ms)
```

In fast-dev mode `noisy_gemm` returns just the dequantized matmul output and
never "finds" a block. Leave it off for bit-exact PoW behaviour.

## Scope / caveats

- **Bit-exact for PoW** (hashing/commitment proven equal to the kernels by the
  repo's `test_reference_vs_kernels.py`); **NOT performance-representative** —
  measure GEMM performance only on the real `sm_90` kernels (H100).
- MoE / `noise_A` / `noise_B` / `denoise_converter` / `inner_hash` ops raise
  `NotImplementedError` (not needed for the dense inference path). Implement
  against `miner_base` if a code path requires them.
- `torch` has no integer matmul on CUDA, so the reference does int32
  accumulation on CPU regardless of input device.
