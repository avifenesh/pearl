"""Pure-Python ``pearl_gemm_cuda`` replacement for local development.

The real ``pearl_gemm_cuda`` is a CUDA extension whose GEMM kernels are
``sm_90a`` (Hopper) only, so it cannot be built or imported on other GPUs
(e.g. Blackwell ``sm_120`` dev machines). This module provides a drop-in,
bit-faithful **reference** implementation of the same surface, backed by the
existing :class:`miner_base.noisy_gemm.NoisyGemm` reference (which mirrors
``pow_utils.hpp`` exactly), so the whole plugin + mining path runs locally.

It is installed as ``pearl_gemm_cuda`` in ``sys.modules`` by
:func:`install` when the real extension is unavailable. Activation is
also forced by setting ``PEARL_GEMM_REFERENCE=1``.

Scope: correctness / PoW bit-exactness for development. NOT performance —
the reference is slow and single-threaded-ish; GEMM perf must be measured on
the real ``sm_90`` kernels (H100).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import torch

# ---------------------------------------------------------------------------
# Constants mirrored from csrc/gemm/{host_signal_header,pearl_gemm_constants}.hpp
# ---------------------------------------------------------------------------
MAX_NUM_REGISTERS_PER_THREAD = 256
CHAINING_VALUE_SIZE_U32 = 8  # blake3 256-bit digest = 8 x u32
CHUNK_SIZE = 1024  # blake3::CHUNK_SIZE
CHAINING_VALUE_SIZE = 32  # bytes

# kAxEBLScaleFactor=1<<14, kEARxBpEBScaleFactor=1<<12, kIntToFp16ScaleFactor=1<<12
kEALScaleFactorDenoise = -1 * (1 << 12) // (1 << 12)  # = -1
kEBRScaleFactorDenoise = -1 * (1 << 14) // (1 << 12)  # = -4

# Header byte layout (reference-internal; only this module reads it back via
# get_host_signal_header, so the exact C++ struct layout is not required).
_HEADER_SIZE_BYTES = 2 * 128  # matches the kernel's reserved 2x128B header


class HostSignalStatus(enum.IntEnum):
    kSignalIdle = 0
    kSignalTriggered = 1


@dataclass
class MMASize:
    m: int = 0
    n: int = 0
    k: int = 0


@dataclass
class HostSignalHeader:
    """Reference mirror of the C++ HostSignalHeader (fields used in Python)."""

    status: HostSignalStatus = HostSignalStatus.kSignalIdle
    tileCoord: list = field(default_factory=lambda: [0, 0, 0])
    threadIdx: list = field(default_factory=lambda: [0, 0, 0])
    num_registers_per_thread: int = 0
    thread_rows: list = field(default_factory=list)
    thread_cols: list = field(default_factory=list)
    mma_size: MMASize = field(default_factory=MMASize)
    mma_tile_size: MMASize = field(default_factory=MMASize)
    target: list = field(default_factory=lambda: [0] * CHAINING_VALUE_SIZE_U32)


# A pinned-header tensor's id() -> the HostSignalHeader the reference wrote into
# it. The real kernel writes raw bytes the C++ get_host_signal_header decodes;
# here we keep a side-table keyed by tensor identity (the reference owns both
# the writer and get_host_signal_header reader, so this is self-consistent).
_HEADER_TABLE: dict[int, HostSignalHeader] = {}


def get_host_signal_header_size() -> int:
    return _HEADER_SIZE_BYTES


def get_host_signal_sync_size() -> int:
    return 8  # sizeof(HostSignalSync): int + enum, 8-byte aligned


def get_required_scratchpad_bytes(matrix_bytes: int, threads_per_block: int = 128) -> int:
    bytes_per_block = threads_per_block * CHUNK_SIZE
    required_blocks = (matrix_bytes + bytes_per_block - 1) // bytes_per_block
    return required_blocks * CHAINING_VALUE_SIZE


def get_host_signal_header(host_signal_header_pinned: torch.Tensor) -> HostSignalHeader:
    """Return the HostSignalHeader the reference stashed for this pinned buffer."""
    h = _HEADER_TABLE.get(id(host_signal_header_pinned))
    if h is None:
        return HostSignalHeader(status=HostSignalStatus.kSignalIdle)
    return h


def _set_header(host_signal_header_pinned: torch.Tensor, header: HostSignalHeader) -> None:
    _HEADER_TABLE[id(host_signal_header_pinned)] = header


def build_module():
    """Construct a module object exposing the full ``pearl_gemm_cuda`` surface."""
    import types

    from . import _reference_ops as ops

    mod = types.ModuleType("pearl_gemm_cuda")
    # Types / enums / constants
    mod.HostSignalStatus = HostSignalStatus
    mod.HostSignalHeader = HostSignalHeader
    mod.MMASize = MMASize
    mod.kEALScaleFactorDenoise = kEALScaleFactorDenoise
    mod.kEBRScaleFactorDenoise = kEBRScaleFactorDenoise
    # Sizes / helpers
    mod.get_host_signal_header = get_host_signal_header
    mod.get_host_signal_header_size = get_host_signal_header_size
    mod.get_host_signal_sync_size = get_host_signal_sync_size
    mod.get_required_scratchpad_bytes = get_required_scratchpad_bytes
    # Ops
    for name in (
        "quantize", "noise_gen", "gemm", "noisy_gemm", "tensor_hash",
        "commitment_hash_from_merkle_roots", "noise_A", "noise_B",
        "denoise_converter", "inner_hash", "build_routing_data",
        "get_build_routing_data_scratchpad_bytes",
    ):
        setattr(mod, name, getattr(ops, name))
    mod.__pearl_reference__ = True
    return mod


def install(force: bool = False) -> bool:
    """Install the reference module as ``pearl_gemm_cuda`` if the real ext is absent.

    Returns True if the reference backend was installed. The real extension
    always wins unless ``force`` (or env ``PEARL_GEMM_REFERENCE=1``).
    """
    import os
    import sys

    if "pearl_gemm_cuda" in sys.modules and not force:
        return getattr(sys.modules["pearl_gemm_cuda"], "__pearl_reference__", False)

    if not force and os.getenv("PEARL_GEMM_REFERENCE", "").lower() not in ("1", "true", "yes"):
        # Only auto-install if the real extension cannot be imported.
        try:
            import pearl_gemm_cuda  # noqa: F401  (real ext)
            return False
        except Exception:
            pass

    sys.modules["pearl_gemm_cuda"] = build_module()
    # Also define the pearl_gemm::* torch library ops so register_fake (in
    # pearl_gemm_interface) and any torch.ops.pearl_gemm.* call sites work.
    from . import _reference_torch_ops

    _reference_torch_ops.register()
    return True
