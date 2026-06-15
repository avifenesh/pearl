# Noised-weight (BpEB) cache — caching the static-weight noising per mining job

## What
In mining mode the noisy GEMM noises both operands. `B` is the layer WEIGHT (a
persistent `nn.Parameter`), and its noised form `BpEB = B + E_BL @ E_BR` depends
ONLY on the mining-job key and the weight — NOT on the activation `A`:

    commitment_B = blake3(job_key || merkle_root(B))   # A-INDEPENDENT
    E_BL, E_BR   = noise factors seeded by commitment_B
    BpEB         = B + E_BL @ E_BR                      # A-INDEPENDENT

The mining job changes ~1/s (gateway poll) while a forward pass runs in ms, so
within one job `BpEB` is recomputed byte-for-byte identically on every forward
pass — the dominant, m-independent half of the mining tax on the large FFN
matmuls (Step-0: gate_up B-side +0.167 ms of the +0.20 ms noising; `BpEB` = 117 MB
> L2, so its formation + HBM write is real bandwidth paid every forward). Over a
long prefill that cost amortizes to ~0 when cached.

This caches `BpEB` per `(weight, commitment_B)`. On a cache hit the main GEMM runs
with `run_noising_B=False` (an existing kernel flag) and consumes the cached
weight instead of re-noising it; the main GEMM already reads its B operand from
`ptr_BpEB`. Off by default (`MINER_CACHE_NOISED_WEIGHT=1`).

## The A-dependent side-product (why this is not just `run_noising_B=False`)
The noisingB kernel produces TWO things: `BpEB` (A-independent, cached) and the
denoise side-product `EARxBpEB = BpEB @ E_AR` (A-DEPENDENT — `E_AR` is seeded by
`commitment_A`, which depends on the activation). The main GEMM's denoise step
needs `EARxBpEB` every call, so it is NEVER cached — it is recomputed fresh from
the cached `BpEB` each forward pass.

`E_AR` is a structured-sparse selection matrix and `BpEB` is int8, so the entries
of `EARxBpEB` are small integers that fit exactly in fp32's 24-bit mantissa.
CUDA has no integer matmul, but an **fp32 matmul with TF32 disabled is bit-exact**
for these values and runs on the GPU (no CPU round-trip). We feed it as int32 and
let the kernel's existing denoise converter scale int32 -> fp16, matching the
miner_base reference `noise_B` exactly.

## Why it's correct (PoW-equivalent) — validated bit-exact on H100
A cache HIT yields a `BpEB` byte-identical to recomputing, and `EARxBpEB` is
recomputed fresh every call, so the GEMM's output `C` and its proof-of-work input
are unchanged. Validated on H100 (sm_90, real kernels): driving `noisy_gemm` with
a precomputed BpEB + run_noising_B=False vs full noising gives **C exact-equal,
max|dC| = 0.000e+00**. Every uncertainty degrades to a redundant recompute, never
a wrong weight:
- `commitment_B` bytes are part of the cache key -> a new mining job misses;
- a `weakref` to the exact weight is checked on a hit -> a freed/reused `data_ptr`
  (ABA) misses and recomputes;
- `device` (type + index) is in the key -> CUDA `data_ptr` ints that collide
  across GPUs cannot produce a wrong hit.
Valid under the immutable-weight invariant (Pearl weights are persistent
`nn.Parameter`s set once at load); `clear_cache()` for any weight-reload path.

## Honest scope / caveats
- Memory: a cached `BpEB` is a full `n x k` int8 tensor (~117 MB for a gate_up
  weight). Apply only to the large FFN weights where it pays; a byte budget
  (default 2 GiB) bounds the cache as a backstop.
- We replace the BpEB FORMATION + the `n x k` int8 HBM write (the expensive part)
  with a cache read; the cheaper `EARxBpEB` reduction stays per-call.
- On a cache MISS, `BpEB` is formed via the standalone `noise_B` op and stored;
  the win is on subsequent hits within the same job.
- The win is amortized over the prefill share of compute (big on prefill-heavy
  serving, ~0 on decode-diluted) — report per workload, like the weight-hash PR.

## Validation
- Cache semantics: `tests/test_bpeb_cache.py` (CPU, no kernels) — miss/hit/
  job-change/distinct-weights/weakref-evict/byte-budget. 8/8 pass.
- Invariant (BpEB ⟂ A, deterministic in commitment_B): proven vs miner_base in
  pearl-pr-plan/proofs/prove_bpeb_invariant.py.
- Bit-exact equivalence on H100 (kernel level + the shipped pearl_gemm_noisy with
  the flag on vs off): pearl-pr-plan/modal_bpeb_validate.py.
- Pending for the PR description: the amortized prefill req/s A/B (cache off vs on).
