#pragma once

namespace pearl {
static constexpr int kAxEBLScaleFactor = 1 << 14;
static constexpr int kEARxBpEBScaleFactor = 1 << 12;
// Divide int tensors by a constant factor of 1 << 12 for fp16 tensor core MMA for denoising.
static constexpr int kIntToFp16ScaleFactor = 1 << 12;
// fp16 denoise factors AxEBL, EARxBpEB were already scaled in noising kernels.
// int32 denoise factors AxEBL, EARxBpEB were already converted and scaled by denoise conversion kernel.
// fp16 AxEBL was actually divided by (1 << 14); hence we multiply EBR by 4 = (1<<14) / (1<<12)
//  in this case to adjust everything to a common 1 << 12 scaling.
// Also multiply by -1 because we will subtract when doing denoising mma.
static constexpr int kEBRScaleFactorDenoise =
    -1 * kAxEBLScaleFactor / kIntToFp16ScaleFactor;
static constexpr int kEALScaleFactorDenoise =
    -1 * kEARxBpEBScaleFactor / kIntToFp16ScaleFactor;

// --- B-side noise fusion (B') -------------------------------------------------
// When kFuseNoiseB is true, the main GEMM forms the noised weights BpEB = B + E_B
// (E_B = EBL*EBR truncated to int8) on-chip into the pipeline SMEM, instead of
// reading a pre-materialized BpEB from HBM (which round-trips full n*k weights
// that overflow L2 on large FFN matrices — the measured ~0.17ms/layer tax).
// Default OFF: the build and numerics are byte-identical to the stock path unless
// PEARL_FUSE_NOISE_B is defined, so this cannot regress the existing kernels.
#ifdef PEARL_FUSE_NOISE_B
static constexpr bool kFuseNoiseB = true;
#else
static constexpr bool kFuseNoiseB = false;
#endif
}  // namespace pearl
