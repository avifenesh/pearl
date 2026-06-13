"""
Default compiled kernels configuration.

Minimal set of kernels for development and PR CI.
This replaces default_compiled_kernels.jsonnet with Python + pydantic.
"""

from pearl_gemm_build_utils.kernel_configs import (
    KernelCompilationGrid,
    MatmulKernelConfig,
    NoisingAKernelConfig,
    NoisingBKernelConfig,
)

# Build matmul kernels
import os as _os

_matmul_kernels = []


def _fuse_noise_b_enabled() -> bool:
    """Whether the fused B' build is active.

    Read a build-written marker FIRST (so the test/runtime config always matches
    what was actually compiled, regardless of the current env), then fall back to
    the env var (build time), then default off. This avoids the trap where the
    GEMM was built fused (2 stages) but a later process reads the env unset and
    requests an un-compiled 3-stage kernel (or vice versa)."""
    try:
        from pearl_gemm_build_utils.kernel_configs import _build_marker

        return bool(_build_marker.FUSE_NOISE_B)
    except Exception:
        pass
    return _os.environ.get("PEARL_FUSE_NOISE_B", "").upper() in (
        "1",
        "TRUE",
        "YES",
        "ON",
    )


# B' fusion stages the int8 EBL ring in SMEM alongside A/B; drop the main GEMM
# to 2 pipeline stages so the fused build fits the H100 SMEM limit.
_matmul_stages = 2 if _fuse_noise_b_enabled() else 3

# 128x256x128, R=64/128
for R in [64, 128]:
    _matmul_kernels.append(
        MatmulKernelConfig(
            tile_size_m=128,
            tile_size_n=256,
            tile_size_k=128,
            R=R,
            pipeline_stages=_matmul_stages,
            cM=1,
            cN=1,
        )
    )

# Noising A: 64x64, fp16/int32
_noising_a_kernels = [
    NoisingAKernelConfig(
        tile_size_m=64,
        tile_size_k=64,
        R=R,
        pipeline_stages=2,
        AxEBL_type=dtype,
    )
    for R in [64, 128]
    for dtype in ["fp16", "int32"]
]

# Noising B: 64x64, fp16/int32
_noising_b_kernels = [
    NoisingBKernelConfig(
        tile_size_n=64,
        tile_size_k=64,
        R=R,
        pipeline_stages=2,
        EARxBpEB_type=dtype,
    )
    for R in [64, 128]
    for dtype in ["fp16", "int32"]
]

KERNEL_CONFIGS = KernelCompilationGrid(
    matmul_kernels=_matmul_kernels,
    noising_a_kernels=_noising_a_kernels,
    noising_b_kernels=_noising_b_kernels,
)
