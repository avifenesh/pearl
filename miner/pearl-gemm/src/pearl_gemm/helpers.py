import atexit
import os
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock, Semaphore

import torch
from blake3 import blake3
from loguru import logger
from pearl_gemm_cuda import HostSignalHeader, get_host_signal_header_size

SIZE_U32 = 4
BLAKE3_DIGEST_SIZE_U32 = blake3.digest_size // SIZE_U32

# Opt-in pinned-pool profiling. When PEARL_PROFILE_PINNED_POOL is truthy, the pool
# records acquire-wait latency + outstanding high-water mark and dumps a summary at
# process exit. Off by default => zero added work on the hot path.
_PROFILE_PINNED_POOL = os.getenv("PEARL_PROFILE_PINNED_POOL", "").lower() in (
    "1",
    "true",
    "yes",
)

# Wait-time histogram bucket edges in milliseconds (upper-inclusive); the final
# bucket catches everything larger.
_WAIT_BUCKETS_MS = (0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)


def make_pow_target_tensor(value: int, device="cuda") -> torch.Tensor:
    """Create a pow_target tensor from a uint256 integer value."""

    result = torch.empty((BLAKE3_DIGEST_SIZE_U32,), dtype=torch.uint32, device=device)
    for i in range(BLAKE3_DIGEST_SIZE_U32):
        result[i] = value & 0xFFFFFFFF
        value >>= 32
    return result


_LOGGER = logger.bind(name=__name__)


@dataclass
class _PoolStats:
    """Thread-safe counters for pinned-pool contention profiling."""

    pool_size: int
    acquires: int = 0
    blocked_acquires: int = 0  # acquires that had to wait (pool empty at request time)
    total_wait_s: float = 0.0
    max_wait_s: float = 0.0
    outstanding: int = 0
    high_water: int = 0
    wait_buckets_ms: list = field(default_factory=lambda: [0] * (len(_WAIT_BUCKETS_MS) + 1))
    _lock: Lock = field(default_factory=Lock)

    def record_acquire(self, wait_s: float, was_blocked: bool) -> None:
        with self._lock:
            self.acquires += 1
            if was_blocked:
                self.blocked_acquires += 1
            self.total_wait_s += wait_s
            if wait_s > self.max_wait_s:
                self.max_wait_s = wait_s
            self.outstanding += 1
            if self.outstanding > self.high_water:
                self.high_water = self.outstanding
            wait_ms = wait_s * 1000.0
            placed = False
            for i, edge in enumerate(_WAIT_BUCKETS_MS):
                if wait_ms <= edge:
                    self.wait_buckets_ms[i] += 1
                    placed = True
                    break
            if not placed:
                self.wait_buckets_ms[-1] += 1

    def record_release(self) -> None:
        with self._lock:
            self.outstanding -= 1

    def summary(self) -> str:
        with self._lock:
            avg_ms = (self.total_wait_s / self.acquires * 1000.0) if self.acquires else 0.0
            edges = [*(f"<={e}ms" for e in _WAIT_BUCKETS_MS), "more"]
            hist = ", ".join(
                f"{lbl}:{n}" for lbl, n in zip(edges, self.wait_buckets_ms, strict=True) if n
            )
            return (
                f"[pinned-pool profile] pool_size={self.pool_size} acquires={self.acquires} "
                f"blocked={self.blocked_acquires} "
                f"({100.0 * self.blocked_acquires / max(self.acquires, 1):.1f}%) "
                f"avg_wait={avg_ms:.3f}ms max_wait={self.max_wait_s * 1000.0:.3f}ms "
                f"high_water={self.high_water}/{self.pool_size}\n"
                f"  wait_hist: {hist}"
            )


class HostSignalHeaderPinnedPool:
    def __init__(self, pool_size: int = 128) -> None:
        self._pool_size = pool_size
        self._available_buffers = deque()
        self._used_buffers: set[int] = set()
        self._lock = Lock()
        self._semaphore = Semaphore(self._pool_size)

        self._stats: _PoolStats | None = None
        if _PROFILE_PINNED_POOL:
            self._stats = _PoolStats(pool_size=pool_size)
            atexit.register(lambda: _LOGGER.info(self._stats.summary()))
            _LOGGER.info(f"Pinned-pool profiling ENABLED (pool_size={pool_size})")

        host_signal_header_size = get_host_signal_header_size()

        # pre-allocate pinned buffer
        for _ in range(self._pool_size):
            self._available_buffers.append(
                torch.zeros((host_signal_header_size,), dtype=torch.int8, pin_memory=True),
            )

    def acquire(self) -> torch.Tensor:
        if not self._available_buffers:
            _LOGGER.warning(f"Pool size exceeded, {self._pool_size=}")

        if self._stats is not None:
            # Non-blocking probe first to classify "had to wait", then time the real wait.
            t0 = time.perf_counter()
            got = self._semaphore.acquire(blocking=False)
            was_blocked = not got
            if not got:
                self._semaphore.acquire()  # the real (blocking) wait
            wait_s = time.perf_counter() - t0
            self._stats.record_acquire(wait_s, was_blocked)
        else:
            self._semaphore.acquire()

        with self._lock:
            buffer = self._available_buffers.popleft()
            if id(buffer) in self._used_buffers:
                raise AssertionError("Unexpectedly found available buffer in _used_buffers")
            self._used_buffers.add(id(buffer))
            return buffer

    def release(self, buffer: torch.Tensor) -> None:
        with self._lock:
            if id(buffer) not in self._used_buffers:
                raise ValueError("Attempted to release unused buffer")
            self._used_buffers.remove(id(buffer))
            self._available_buffers.append(buffer)
            buffer.zero_()

        if self._stats is not None:
            self._stats.record_release()
        self._semaphore.release()


@dataclass
class ProofTileIndices:
    A_row_indices: list[int]
    B_column_indices: list[int]


def extract_indices(header: HostSignalHeader) -> ProofTileIndices:
    row_tile_coord = header.tileCoord[0] * header.mma_tile_size.m
    col_tile_coord = header.tileCoord[1] * header.mma_tile_size.n
    thread_rows = sorted(set(header.thread_rows))
    thread_cols = sorted(set(header.thread_cols))

    return ProofTileIndices(
        A_row_indices=[row_tile_coord + r for r in thread_rows],
        B_column_indices=[col_tile_coord + c for c in thread_cols],
    )
