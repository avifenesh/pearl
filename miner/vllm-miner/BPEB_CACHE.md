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

This caches `BpEB` per `(weight, job_key)`, which is equivalent to `(weight,
commitment_B)` under the immutable-weight invariant because
`commitment_B = blake3(job_key || merkle_root(B))`. Keying on the already-CPU job
key avoids a GPU sync just to copy `commitment_B` back to Python. On a cache hit
the main GEMM runs with `run_noising_B=False` (an existing kernel flag) and
consumes the cached weight instead of re-noising it; the main GEMM already reads
its B operand from `ptr_BpEB`. Off by default; enable with
`MINER_CACHE_NOISED_WEIGHT=1`.

For serving experiments, `MINER_CACHE_NOISED_WEIGHT_READY_FILE=/path/to/file`
can defer cache use until after vLLM startup/profiling. If set, cache hits are
disabled until the file exists; this avoids charging cached tensors against the
KV-cache memory budget during vLLM profiling.

`MINER_CACHE_NOISED_WEIGHT_LOG_INTERVAL=N` logs JSON cache counters every N
lookups. Use this only while benchmarking.

## The A-dependent side-product (why this is not just `run_noising_B=False`)
The noisingB kernel produces TWO things: `BpEB` (A-independent, cached) and the
denoise side-product `EARxBpEB = BpEB @ E_AR` (A-DEPENDENT — `E_AR` is seeded by
`commitment_A`, which depends on the activation). The main GEMM's denoise step
needs `EARxBpEB` every call, so it is NEVER cached — it is recomputed fresh from
the cached `BpEB` each forward pass.

`E_AR` is a structured-sparse selection matrix and `BpEB` is int8, so the cache
hit path recomputes `EARxBpEB` with a dedicated CUDA product:
`BpEB(n x k) @ EAR_K_major(r x k).T -> int32(n x r)`. The result is fed to the
kernel's existing denoise converter, which scales int32 -> fp16, matching the
miner_base reference `noise_B` exactly.

## Why it's correct (PoW-equivalent) — validated bit-exact on H100
A cache HIT yields a `BpEB` byte-identical to recomputing, and `EARxBpEB` is
recomputed fresh every call, so the GEMM's output `C` and its proof-of-work input
are unchanged. Validated on H100 (sm_90, real kernels): driving `noisy_gemm` with
a precomputed BpEB + run_noising_B=False vs full noising gives **C exact-equal,
max|dC| = 0.000e+00**. Every uncertainty degrades to a redundant recompute, never
a wrong weight:
- mining-job key bytes are part of the cache key -> a new mining job misses;
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
  with a cache read; the smaller `EARxBpEB` product stays per-call and now runs
  through the dedicated CUDA side-product op.
- On a cache MISS, `BpEB` is formed via the standalone `noise_B` op and stored;
  the win is on subsequent hits within the same job.
- The win is amortized over the prefill share of compute (big on prefill-heavy
  serving, ~0 on decode-diluted) — report per workload, like the weight-hash PR.

## Validation
- Cache semantics: `tests/test_bpeb_cache.py` (CPU, no kernels) — miss/hit/
  job-change/distinct-weights/weakref-evict/byte-budget/admission stats. 9/9 pass.
- Invariant (BpEB ⟂ A, deterministic in commitment_B): proven vs miner_base in
  pearl-pr-plan/proofs/prove_bpeb_invariant.py.
- Bit-exact equivalence on H100 (kernel level + the shipped pearl_gemm_noisy with
  the flag on vs off): pearl-pr-plan/modal_bpeb_validate.py.
- H100 kernel microbench with the CUDA side-product: steady-state cache hits are
  faster on the tested prefill-shaped GEMMs; report exact workload numbers in the
  PR text, not as a default claim.

## Serving benchmark result
Modal H100, tp=1, `max_model_len=4096`, `gpu_memory_utilization=0.93`,
2770 prompt tokens/request, `PearlKernel` plugin hits = 200 per request.

Early cache allocation (plain `MINER_CACHE_NOISED_WEIGHT=1`) reduced vLLM KV
capacity and still lost latency (Modal result
`/tmp/pearl-modal-results/bpeb-server-H100-tp1-ctx4096-20260617T022327Z.json`):
- cache off: median 0.561552 s, mean 0.562682 s, KV 3.69 GiB / 12,080 tokens /
  2.95x concurrency
- cache on: median 0.606051 s, mean 0.609670 s, KV 1.54 GiB / 5,024 tokens /
  1.23x concurrency
- result: -7.34% median, -7.71% mean

Deferred cache allocation (`MINER_CACHE_NOISED_WEIGHT_READY_FILE` created after
`/v1/models` readiness) preserved the KV budget but still lost latency (Modal
run `ap-eEiAEEjLpqxc2K3duiEGGa`, result
`/tmp/pearl-modal-results/bpeb-server-H100-tp1-ctx4096-20260617T042007Z.json`):
- cache off: median 0.517003 s, mean 0.519665 s, KV 3.69 GiB / 12,080 tokens /
  2.95x concurrency
- cache on: median 0.558280 s, mean 0.557966 s, KV 3.69 GiB / 12,080 tokens /
  2.95x concurrency
- result: -7.39% median, -6.86% mean

Follow-up H100 kernel-only bench showed an isolated cache hit is not the problem
for a representative prefill-shaped layer (`m=2770, n=14336, k=8192, r=128`;
Modal result
`/tmp/pearl-modal-results/bpeb-hit-path-H100-20260617T044025Z.json`):
- `noise_B_fp16`: median 0.180192 ms
- `bpeb_ear_product_int32`: median 0.092320 ms
- `denoise_converter_ear_only`: median 0.017584 ms
- full `noisy_gemm`: median 0.808656 ms
- cached-hit sequence (`bpeb_ear_product` + skip-B `noisy_gemm`): median
  0.711248 ms
- result: isolated cached hit is +13.7% faster for this shape

A smaller-object/fusion probe added a no-store B-side noising mode that still
forms each `BpEB` tile in shared memory and computes `EARxBpEB`, but skips the
global `n x k` `BpEB` write. It is bit-exact against `noise_B` for
`EARxBpEB`. On the same H100 shape (Modal run `ap-jXFvq8kpU0ZNbyoFwA7Yox`,
result `/tmp/pearl-modal-results/bpeb-hit-path-H100-20260617T062620Z.json`):
- `noise_B_fp16`: median 0.182208 ms
- `noise_B_side_product_fp16`: median 0.160432 ms
- result: no-store side-product is +13.6% faster inside `noise_B`, but only
  saves 0.021776 ms absolute for this layer

This says the standalone `BpEB` write is measurable but not the dominant cost.
A true mainloop fusion could also remove the later `BpEB` read, but the current
no-store probe by itself is not a production path because the existing main GEMM
still needs a full `BpEB` operand.

But the serving cache had effectively no hits. A stats run with deferred cache,
one warmup plus one measured request, and `MINER_CACHE_NOISED_WEIGHT_LOG_INTERVAL=25`
(Modal run `ap-lg090XVkcxjFvTb33oZPGp`, result
`/tmp/pearl-modal-results/bpeb-server-H100-tp1-ctx4096-20260617T051832Z.json`)
ended with:
- lookups: 400
- hits: 0
- misses/stores: 400
- over-budget clears: 46
- final entries: 4
- final bytes: 1.02 GiB of a 2 GiB budget
- serving result: cache off 0.516696 s, cache on 0.546583 s (-5.47%)

Selective admission (preserve the resident subset and skip admission instead of
clearing when the byte budget is full) fixed the zero-hit thrash but still lost
latency. The deferred-cache 2 GiB serving run with
`MINER_CACHE_NOISED_WEIGHT_MAX_BYTES=2147483648` and
`MINER_CACHE_NOISED_WEIGHT_LOG_INTERVAL=25` (Modal run
`ap-GQSW2nETj3AgguX0ot9J9p`, result
`/tmp/pearl-modal-results/bpeb-server-H100-tp1-ctx4096-20260617T054756Z.json`)
ended with:
- cache off: median/mean 0.538894 s
- cache on: median/mean 0.584089 s
- result: -7.74% median/mean
- final cache lookups: 400
- final hits: 8
- final misses: 392
- final stores: 8
- admission skips: 384
- over-budget clears: 0
- final entries: 8
- final bytes: exactly 2.00 GiB of a 2 GiB budget

Doubling that to 4 GiB doubled the resident subset and hit count but made latency
worse (Modal run `ap-82mknOM6h03jLN0MAuuA7k`, result
`/tmp/pearl-modal-results/bpeb-server-H100-tp1-ctx4096-20260617T055408Z.json`):
- cache off: median/mean 0.569111 s
- cache on: median/mean 0.632656 s
- result: -10.04% median/mean
- final cache lookups: 400
- final hits: 16
- final misses: 384
- final stores: 16
- admission skips: 368
- over-budget clears: 0
- final entries: 16
- final bytes: exactly 4.00 GiB of a 4 GiB budget

Conclusion: allocation timing is a real hidden parameter for vLLM capacity, and
the isolated full-`BpEB` hit path can be faster. Selective admission proves that
a bounded resident full-`BpEB` cache can avoid destructive clears, but the useful
reuse is far too sparse at realistic budgets: 2 GiB gives 8 repeated hits and
4 GiB gives 16, while hundreds of lookups still pay miss cost. Full `BpEB`
caching should not be a PR as-is; a plausible next target must reduce the cached
object size or avoid the full-weight working-set problem, not merely tune the
hit-path kernel.
