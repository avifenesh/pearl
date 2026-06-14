"""Cache for the keyed merkle hash of a static mining weight.

The noisy mining GEMM hashes BOTH operands (``tensor_hash``) to seed the PoW
commitment. The ``B`` operand is the layer weight: constant for the process
lifetime, while the hash key only changes when the mining job changes (gateway
polled ~1/s) and a forward pass runs in milliseconds. So ``tensor_hash(weight)``
recomputes the identical 32-byte root every forward pass -- pure waste
(0.13-0.41 ms/layer on H100, the dominant per-layer cost at decode batch sizes).

This caches that root. A cached value is ALWAYS bit-identical to recomputing, so
the proof-of-work path is unchanged; any uncertainty degrades to a redundant
recompute, never a wrong hash:

* the mining-job ``key`` bytes are part of the cache key -> a new job misses;
* a ``weakref`` to the exact tensor is checked on a hit -> a freed/reused
  ``data_ptr`` (ABA) misses and recomputes;
* ``device`` (type + index) is part of the key -> CUDA ``data_ptr`` ints that
  collide across GPUs cannot produce a wrong hit.

Entries self-evict via a ``weakref`` finalizer when their weight is garbage
collected (a size cap is kept only as a backstop for non-collectable churn).

Dict operations are atomic under the GIL, and all mining ops run on the default
CUDA stream (no explicit streams in the plugin), so the cached tensor's producing
``tensor_hash`` is ordered before any consumer on the same stream — identical to
the un-cached code's own same-stream assumption.

Validity assumes the weight is IMMUTABLE (Pearl weights are persistent
``nn.Parameter``s set once at load and never mutated in place). The cache cannot
detect in-place mutation of the same object (``copy_``/``set_``/adapter swaps);
call :func:`clear_cache` on any such weight-reload path.
"""

import weakref
from collections.abc import Callable

import torch

_CACHE: dict[tuple, tuple] = {}
_MAX_ENTRIES = 4096


def cached_weight_hash(
    weight: torch.Tensor,
    key_bytes: bytes,
    compute: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Return the keyed merkle hash of ``weight``, reusing a cached value only
    when the exact same (still-alive) tensor and mining-job key are unchanged.

    On any miss it calls ``compute()`` (which must produce the hash) and caches
    the result; the returned bytes are always identical to recomputing.
    """
    cache_key = (
        weight.device.type,
        weight.device.index,
        weight.data_ptr(),
        tuple(weight.shape),
        weight.dtype,
    )
    entry = _CACHE.get(cache_key)
    if entry is not None:
        cached_key_bytes, weak_weight, cached_hash = entry
        if cached_key_bytes == key_bytes and weak_weight() is weight:
            return cached_hash

    result = compute()

    # Register a finalizer so the entry is dropped as soon as `weight` is GC'd,
    # instead of lingering until the size cap clears the whole cache. The
    # `entry[1] is ref` check ensures a finalizer only removes ITS OWN entry and
    # never a newer one that reused the same cache_key (e.g. data_ptr reuse).
    def _evict(ref: weakref.ref, key: tuple = cache_key) -> None:
        entry = _CACHE.get(key)
        if entry is not None and entry[1] is ref:
            _CACHE.pop(key, None)

    try:
        weak_weight = weakref.ref(weight, _evict)
    except TypeError:
        return result  # not weak-referenceable: skip caching (still correct)
    if len(_CACHE) >= _MAX_ENTRIES:  # backstop for non-collectable churn
        _CACHE.clear()
    _CACHE[cache_key] = (key_bytes, weak_weight, result)
    return result


def clear_cache() -> None:
    """Drop all cached weight hashes (test/maintenance helper)."""
    _CACHE.clear()


def cache_size() -> int:
    """Number of live cache entries (introspection/test helper)."""
    return len(_CACHE)
