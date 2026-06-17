"""
Default compiled kernels configuration.

Minimal set of kernels for development and PR CI.
This replaces default_compiled_kernels.jsonnet with Python + pydantic.
"""

import os as _os
import sys as _sys
from pathlib import Path as _Path

from pearl_gemm_build_utils.kernel_configs import (
    KernelCompilationGrid,
    MatmulKernelConfig,
    NoisingAKernelConfig,
    NoisingBKernelConfig,
)

# Build matmul kernels
_matmul_kernels = []


def _read_bool_marker(path: _Path) -> bool | None:
    if not path.exists():
        return None
    namespace: dict[str, object] = {}
    exec(path.read_text(), namespace)  # noqa: S102 - trusted local build marker
    return bool(namespace.get("FUSE_NOISE_B", False))


def _fuse_noise_b_enabled() -> bool:
    """Whether the fused B' build is active.

    Prefer build-written markers, then fall back to the env var used at build
    time. The pearl_gemm marker is packaged with the compiled extension so a
    fused wheel keeps selecting its restricted R=64 kernel set even if the env is
    unset later.
    """
    marker = _read_bool_marker(_Path(__file__).with_name("_build_marker.py"))
    if marker is not None:
        return marker
    for entry in _sys.path:
        marker = _read_bool_marker(_Path(entry) / "pearl_gemm" / "_build_marker.py")
        if marker is not None:
            return marker
    return _os.environ.get("PEARL_FUSE_NOISE_B", "").upper() in (
        "1",
        "TRUE",
        "YES",
        "ON",
    )


# B' fusion: keep 3 pipeline stages, but the separate int8 EBL/EBR SMEM region
# only fits the H100 limit at R=64 (at R=128 the fp16 denoise arm is already
# ~192KB and EBL+EBR add ~48KB on top -> overflow). So when fused, build only
# the R=64 matmul kernel. (R=128 fused needs a non-aliasing SMEM redesign;
# tracked separately.) Stock (non-fused) build keeps both R=64 and R=128.
_fused = _fuse_noise_b_enabled()
_matmul_stages = 3
_matmul_Rs = [64] if _fused else [64, 128]

# 128x256x128
for R in _matmul_Rs:
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
