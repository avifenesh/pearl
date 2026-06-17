"""Cache for the noised mining WEIGHT ``BpEB = B + E_BL @ E_BR``.

The noisy mining GEMM noises both operands. The ``B`` operand is the layer
weight (a persistent ``nn.Parameter``, constant for the process lifetime), and
its noised form ``BpEB`` is a deterministic function of ONLY the mining-job key
and the weight -- NOT of the activation ``A``:

    commitment_B = blake3(key || merkle_root(B))        # A-INDEPENDENT
    E_BL, E_BR   = noise factors seeded by commitment_B # A-INDEPENDENT
    BpEB         = B + E_BL @ E_BR                       # A-INDEPENDENT

The mining job changes ~1/s (gateway poll); a forward pass runs in
milliseconds. So within one job ``BpEB`` is recomputed byte-for-byte identically
on every forward pass -- the dominant, m-independent half of the mining tax on
large FFN weights (forming ``E_BL @ E_BR`` plus a full ``n x k`` int8 write that
overflows L2; ~0.17 ms/layer on gate_up on H100).

This caches ``BpEB`` keyed by ``(weight, mining-job key)``. For an immutable
weight this is equivalent to ``(weight, commitment_B)`` because
``commitment_B = blake3(key || merkle_root(B))``; using the already-CPU job key
avoids synchronizing the GPU just to copy ``commitment_B`` back to Python. A
cached value is ALWAYS bit-identical to recomputing, so the proof-of-work path is
unchanged; any uncertainty degrades to a redundant recompute, never a wrong
weight:

* the mining-job key bytes are part of the cache key -> a new job misses and
  recomputes;
* a ``weakref`` to the exact weight tensor is checked on a hit -> a freed/reused
  ``data_ptr`` (ABA) misses and recomputes;
* ``device`` (type + index) is part of the key -> CUDA ``data_ptr`` ints that
  collide across GPUs cannot produce a wrong hit.

The activation-dependent side-product ``EARxBpEB = E_AR @ BpEB`` is NOT cached
(``E_AR`` is seeded by ``commitment_A`` and changes every forward); it is always
recomputed fresh from the cached ``BpEB``. So only the A-independent weight
noising is reused -- the proof-of-work is bit-unchanged.

Entries self-evict via a ``weakref`` finalizer when their weight is garbage
collected. A byte-budget backstop bounds memory for non-collectable churn, since
each cached ``BpEB`` is a full ``n x k`` int8 tensor (e.g. ~117 MB for a Llama
gate_up weight) -- materially larger than a 32-byte hash, so the cache should be
applied only to the large FFN weights where it pays.

Dict operations are atomic under the GIL, and all mining ops run on the default
CUDA stream, so the cached tensor's producing kernel is ordered before any
consumer on the same stream -- identical to the un-cached code's own same-stream
assumption.

Validity assumes the weight is IMMUTABLE (Pearl weights are persistent
``nn.Parameter``s set once at load and never mutated in place). The cache cannot
detect in-place mutation of the same object (``copy_``/``set_``/adapter swaps);
call :func:`clear_cache` on any such weight-reload path.
"""

import weakref
from collections.abc import Callable

import torch

# cache_key -> (job key bytes, weakref(weight), BpEB tensor, nbytes)
_CACHE: dict[tuple, tuple] = {}

# Backstop for non-collectable churn. BpEB tensors are large (n*k int8), so bound
# total cached bytes rather than entry count. 2 GiB ~= a handful of big FFN weights.
_MAX_BYTES = 2 * 1024 * 1024 * 1024
_cur_bytes = 0
_STATS: dict[str, int] = {
    "lookups": 0,
    "hits": 0,
    "misses": 0,
    "stores": 0,
    "evictions": 0,
    "clears": 0,
    "over_budget_clears": 0,
    "admission_skips": 0,
    "oversize_skips": 0,
    "stale_replacements": 0,
    "weakref_skips": 0,
}


def cached_noised_weight(
    weight: torch.Tensor,
    job_key: bytes,
    compute: Callable[[], torch.Tensor],
    max_bytes: int | None = None,
) -> torch.Tensor:
    """Return the noised weight ``BpEB`` for ``weight`` under the current mining
    job, reusing a cached value only when the exact same (still-alive) weight
    tensor and mining-job key are unchanged.

    On any miss it calls ``compute()`` (which must produce ``BpEB``) and caches
    the result; the returned tensor is always identical to recomputing.

    ``compute`` MUST return the freshly-noised ``BpEB``; the caller is
    responsible for separately recomputing the activation-dependent
    ``EARxBpEB`` on every call (it is never cached).
    """
    global _cur_bytes
    byte_budget = _MAX_BYTES if max_bytes is None else max_bytes
    _STATS["lookups"] += 1
    cache_key = (
        weight.device.type,
        weight.device.index,
        weight.data_ptr(),
        tuple(weight.shape),
        weight.dtype,
    )
    entry = _CACHE.get(cache_key)
    if entry is not None:
        cached_job_key, weak_weight, cached_bpeb, _nbytes = entry
        if cached_job_key == job_key and weak_weight() is weight:
            _STATS["hits"] += 1
            return cached_bpeb

    _STATS["misses"] += 1
    result = compute()

    # Register a finalizer so the entry is dropped as soon as `weight` is GC'd
    # instead of lingering until the byte budget clears the whole cache. The
    # `entry[1] is ref` check ensures a finalizer only removes ITS OWN entry and
    # never a newer one that reused the same cache_key (e.g. data_ptr reuse).
    def _evict(ref: weakref.ref, key: tuple = cache_key) -> None:
        global _cur_bytes
        e = _CACHE.get(key)
        if e is not None and e[1] is ref:
            _cur_bytes -= e[3]
            _CACHE.pop(key, None)
            _STATS["evictions"] += 1

    try:
        weak_weight = weakref.ref(weight, _evict)
    except TypeError:
        _STATS["weakref_skips"] += 1
        return result  # not weak-referenceable: skip caching (still correct)

    nbytes = result.element_size() * result.nelement()
    # If a stale entry exists for this key (job change), reclaim its bytes first.
    old = _CACHE.get(cache_key)
    if old is not None:
        _cur_bytes -= old[3]
        _CACHE.pop(cache_key, None)
        _STATS["stale_replacements"] += 1

    if nbytes > byte_budget:
        _STATS["admission_skips"] += 1
        _STATS["oversize_skips"] += 1
        return result
    if _cur_bytes + nbytes > byte_budget:
        # Selective admission: preserve the resident subset and compute this
        # weight uncached. Clearing here made sequential model traversals chase
        # the current layer and produced zero hits on the next request.
        _STATS["admission_skips"] += 1
        return result

    _CACHE[cache_key] = (job_key, weak_weight, result, nbytes)
    _cur_bytes += nbytes
    _STATS["stores"] += 1
    return result


def clear_cache() -> None:
    """Drop all cached noised weights (test/maintenance helper)."""
    global _cur_bytes
    _CACHE.clear()
    _cur_bytes = 0
    _STATS["clears"] += 1


def cache_size() -> int:
    """Number of live cache entries (introspection/test helper)."""
    return len(_CACHE)


def cache_bytes() -> int:
    """Total bytes held by cached BpEB tensors (introspection/test helper)."""
    return _cur_bytes


def cache_stats() -> dict[str, int]:
    """Return live noised-weight cache counters (introspection/debug helper)."""
    return {
        **_STATS,
        "entries": len(_CACHE),
        "bytes": _cur_bytes,
        "default_max_bytes": _MAX_BYTES,
    }


def reset_cache_stats() -> None:
    """Reset noised-weight cache counters (test helper)."""
    for key in _STATS:
        _STATS[key] = 0
