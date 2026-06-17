# B-side noise fusion: form BpEB on chip

This change adds an experimental build option to move B-side noising into the
GEMM mainloop:

```text
PEARL_FUSE_NOISE_B=1
```

The stock path computes and materializes the noised B operand, `BpEB`, before the
main GEMM. The main GEMM then reads that full `n x k` int8 noised-weight tile
from HBM. For large decoder/FFN matrices, this materialized-`BpEB` HBM traffic is
the expensive part.

The fused path keeps the skinny B-side noise factors (`EBL`, `EBR`) and forms:

```text
BpEB = B + int8(EBR * EBL)
```

inside the GEMM mainloop, in shared memory, before the main WGMMA consumes the B
operand. In other words, it moves the noising work on chip and avoids the main
GEMM's HBM read of materialized `BpEB`.

The current implementation still runs the separate `noising_B` launch to produce
`EARxBpEB` for denoising correctness, and that launch still materializes `BpEB`.
In a fused build the main GEMM ignores that materialized `BpEB` and re-forms the
B operand on chip. A future follow-up can remove the remaining `BpEB` write once
`EARxBpEB` is also produced without the materialized noised-B output.

## Why this matters for decoding

Decoder serving is often memory-bound on weight movement. The fused path adds
small on-chip compute but removes full noised-B HBM traffic. During decode and
speculative decode, that extra compute can be overlapped/hidden behind the
serving loop while avoiding the memory round trip.

This is the implementation novelty. Benchmark evidence is separate: real-prompt
EAGLE runs show that when the mining/noisy path is used in speculative decoding,
the noising becomes serving-throughput-free or better in the validated regimes.

## Scope

- Default build remains unchanged: `PEARL_FUSE_NOISE_B=0`.
- The fused build is currently restricted to `R=64`.
- The fused path writes `BpEB` in-place into the existing B shared-memory tile.
  This keeps shared-memory usage below the H100 launch limit; an earlier
  decoupled-buffer experiment compiled but requested too much dynamic shared
  memory to launch.
- The separate `noising_B` launch remains for `EARxBpEB` denoising correctness;
  the fused GEMM ignores the materialized HBM `BpEB` and re-forms it on chip.
- If a fused build is invoked without B-side noise factors, the host wiring falls
  back to the materialized `BpEB` operand rather than raw `B`, so the fused build
  cannot silently drop B-side noise.
- `tests/test_pearl_gemm.py::TestFusedNoiseB` is skipped by default and runs only
  in a fused build. It precomputes `EARxBpEB`, poisons materialized `BpEB`, skips
  `noising_B`, and verifies the fused GEMM still produces the correct output —
  proving the GEMM re-forms `BpEB` on chip instead of reading the HBM buffer.
