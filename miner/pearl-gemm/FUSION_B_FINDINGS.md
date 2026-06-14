# B′ noise fusion — findings (spike, NOT a perf win)

Status: **negative result.** The `PEARL_FUSE_NOISE_B` path on this branch
(`feat/fuse-noise-b`) is **functionally correct but ~8× slower** than stock. Do
not ship it as a throughput optimization. This document records what was tried,
what was measured, why it lost, and what is salvageable.

## Goal (what we were chasing)
Mining roughly doubles GEMM cost (kernel-level +92% median) and costs ~+83%
serving throughput on Llama-3.1-8B (H100, warm, submission held constant). A
phase decomposition pinned most of the noisy-GEMM tax on the **noise injection**
step, and specifically the **B-side**: `noising_B` materializes
`BpEB = B + int8(EBL·EBR)` (n×k) to HBM, then the main GEMM reads it back. On
large FFN weights `BpEB` overflows the 50 MB L2 (gate_up: 117 MB), so that is a
genuine HBM write+read round-trip (~0.17 ms/layer of the ~0.20 ms noising tax).
Thesis: form `BpEB` on-chip in the GEMM mainloop and delete the round-trip.

## What was built
A build-flag-gated (`PEARL_FUSE_NOISE_B`, default OFF → byte-identical stock)
fused path: the GEMM mainloop loads the int8 noise factors `EBL` (per k-tile)
and `EBR` (once) via TMA into SMEM, then a pre-pass forms `BpEB` in-place in the
swizzled `sB` (EBR·EBL WGMMA → narrow int8 → add B → write back) before the main
GEMM consumes it. Restricted to R=64 (R=128's fp16 denoise SMEM arm + the int8
EBL/EBR overflow the 227 KB SMEM limit).

## Correctness: PASS
`TestNoisyGEMM::test_int7_noisy_gemm` (R=64) = **384 passed, 0 failed**. The
denoised output `C` matches the reference `A·Bᵀ` (scaled) within the test
tolerance across all shapes/dtypes. The fusion math is right.

Two real bugs were found and fixed getting there:
1. **Deadlock** (`consumer_wait` hung forever): `producer_acquire` arms the
   mainloop mbarrier for A+B bytes only, and `PipelineTmaAsync::producer_commit`'s
   byte arg is a NOP for TMA. The extra EBL/EBR TMA bytes overshot the barrier's
   transaction count. Fix: `pipeline.producer_expect_transaction(state, EBL[+EBR])`
   after `producer_acquire` (leader-only). This `expect_tx` pattern is the
   reusable lesson for adding any extra TMA load to an existing CUTLASS pipeline.
2. **SMEM aliasing hang**: placing EBL/EBR in the A/B union arm made them alias
   the denoise-factor arm, which the producer's `load_denoise` overwrites
   mid-mainloop. Fix: separate SMEM region (only fits at R=64).

## Performance: FAIL (~8× slower)
`noisy_gemm` median ms, R=64, warm (50 iter), STOCK build vs FUSED build:

| shape (m×n×k)        | stock | fused | ratio        |
|----------------------|-------|-------|--------------|
| 4096×28672×4096 (gate_up) | 1.31  | 10.19 | **7.8× slower** |
| 4096×4096×14336 (down)    | 0.64  | 5.17  | 8.1× slower  |
| 1024×28672×4096           | 0.46  | 2.76  | 6.0× slower  |
| 8192×4096×4096            | 0.42  | 3.02  | 7.2× slower  |

And dropping the (now-redundant) `noising_B` launch — the best case of the
planned "candidate-2" — does **not** help: gate_up 10.195 ms (noising_B on) vs
10.096 ms (off). ≈0 change.

## Why it lost
The bottleneck is the **in-mainloop pre-pass itself**, not the redundant
`noising_B` launch. The pre-pass runs on a **single warpgroup**, **serialized
before each k-tile's main GEMM**, with the other warpgroup idle and **no overlap**
with the main WGMMA. That serialization starves the GEMM pipeline and costs far
more (~9 ms) than the HBM round-trip it removes (~0.17 ms). The standalone
`noising_B` kernel is fully parallel and cheap by comparison.

The Step-0 thesis correctly identified the HBM *waste*; the fusion *mechanism*
(serial single-WG pre-pass) is simply the wrong trade.

## Salvageable
- The `producer_expect_transaction` recipe for extending a CUTLASS TMA pipeline.
- The build-marker mechanism (`_build_marker.py`) for deterministic
  test/runtime kernel-config selection independent of env timing.
- A bit-exact reference of on-chip `BpEB` formation (if a *parallel* fusion is
  ever attempted: both warpgroups, or pipelined/epilogue-overlapped noise).

## Verdict
Keep this branch as a documented spike. Not a PR. The real mining-tax
optimization, if any, lies elsewhere — see the broader noisy-path measurement
(option C) for the next target.
