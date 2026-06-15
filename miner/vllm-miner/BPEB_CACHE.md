# Noised-weight (BpEB) cache — PR notes

## What
In mining mode the noisy GEMM noises both operands. `B` is the layer WEIGHT
(a persistent `nn.Parameter`), and its noised form `BpEB = B + E_BL @ E_BR`
depends ONLY on the mining-job key and the weight — NOT on the activation `A`:

    commitment_B = blake3(job_key || merkle_root(B))   # A-INDEPENDENT
    E_BL, E_BR   = noise factors seeded by commitment_B
    BpEB         = B + E_BL @ E_BR                      # A-INDEPENDENT

The mining job changes ~1/s (gateway poll); a forward pass runs in ms. So within
one job `BpEB` is recomputed byte-for-byte identically every forward pass — the
dominant, m-independent half of the mining tax on the large FFN matmuls (forming
`E_BL@E_BR` + a full `n x k` int8 write that overflows L2; ~0.17 ms/layer on
gate_up, H100). Over a long prefill that cost amortizes to ~0 if cached.

This caches `BpEB` per `(weight, commitment_B)` and, on a hit, runs the main GEMM
with `run_noising_B=False` (already a kernel flag) so it consumes the cached
weight instead of re-noising. Off by default (`MINER_CACHE_NOISED_WEIGHT=1`).

## Why it's correct (PoW-equivalent)
A cache HIT yields a `BpEB` byte-identical to recomputing (proven A-independent +
deterministic in `commitment_B`; see proof). Every uncertainty degrades to a
redundant recompute, never a wrong weight:
- `commitment_B` bytes are part of the cache key  -> a new job misses;
- a weakref to the exact weight is checked        -> a freed/reused data_ptr misses (ABA-safe);
- device (type+index) is in the key               -> cross-GPU data_ptr collisions can't false-hit.

The activation-dependent side-product `EARxBpEB = E_AR @ BpEB` (E_AR seeded by
`commitment_A`) is NEVER cached — it is recomputed fresh every call from the
cached `BpEB` (int32 reduction; the kernel's denoise converter scales int32->fp16,
matching miner_base reference `noise_B` exactly). So the proof-of-work path is
bit-unchanged. Valid under the immutable-weight invariant; `clear_cache()` on any
weight reload.

## Honest scope / caveats
- Memory: a cached `BpEB` is a full `n x k` int8 tensor (~117 MB for a Llama
  gate_up weight). Apply only to the large FFN weights where it pays; a byte
  budget (default 2 GiB) bounds the cache as a backstop.
- We save the BpEB FORMATION + the `n x k` write, not the whole B path: the
  cheap `EARxBpEB` reduction stays per-call.
- The win is amortized over the prefill share of compute (big on prefill-heavy
  serving, ~0 on decode-diluted) — report per workload, like the weight-hash PR.
- Cache MISS currently recomputes EARxBpEB once redundantly (noise_B's discarded
  scratch + the fresh matmul); misses are once-per-job so this is negligible. A
  follow-up could capture BpEB from the normal run_noising_B=True path on a miss
  to avoid even that.

## Validation
- Cache semantics: tests/test_bpeb_cache.py (CPU, no kernels) — miss/hit/job-change/
  ABA/weakref-evict/byte-budget. 8/8 pass.
- Invariant (BpEB ⟂ A, deterministic in commitment_B): proven vs miner_base in
  pearl-pr-plan/proofs/prove_bpeb_invariant.py.
- REQUIRED before merge (needs H100, sm_90): bit-exact C and identical
  blocks-found vs master over a run with the flag on, + amortized prefill req/s.
