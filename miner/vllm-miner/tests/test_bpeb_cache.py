"""Unit tests for the noised-weight (BpEB) cache.

These test the CACHE SEMANTICS in isolation (no GPU / no kernels): a miss
computes, a repeat hit returns the same object without recomputing, a mining-job
change (different job key) recomputes, ABA (a freed/reused data_ptr) does
not false-hit, and the byte budget bounds memory. The numerical
bit-exactness of reusing BpEB across forward passes within a job is proven
separately against miner_base in pearl-pr-plan/proofs/prove_bpeb_invariant.py
and is exercised end-to-end by the GPU equivalence test.

Run: cd miner/vllm-miner && uv run pytest tests/test_bpeb_cache.py -v
"""

import torch
from vllm_miner.bpeb_cache import (
    cache_bytes,
    cache_size,
    cache_stats,
    cached_noised_weight,
    clear_cache,
    reset_cache_stats,
)


def _weight(n=8, k=16):
    return torch.zeros((n, k), dtype=torch.int8)


def setup_function(_):
    clear_cache()
    reset_cache_stats()


def test_miss_computes_then_hit_does_not_recompute():
    w = _weight()
    cb = b"job-A" + b"\x00" * 27
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return torch.full_like(w, 7)

    out1 = cached_noised_weight(w, cb, compute)
    assert calls["n"] == 1
    out2 = cached_noised_weight(w, cb, compute)
    assert calls["n"] == 1, "a hit must NOT recompute"
    assert out2 is out1, "a hit returns the cached object"
    assert torch.equal(out1, torch.full_like(w, 7))


def test_job_change_recomputes():
    w = _weight()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return torch.full_like(w, calls["n"])

    a = cached_noised_weight(w, b"job-A" + b"\x00" * 27, compute)
    b = cached_noised_weight(w, b"job-B" + b"\x00" * 27, compute)
    assert calls["n"] == 2, "a new job key must recompute"
    assert not torch.equal(a, b)
    # cache holds the latest; the stale job-A bytes were reclaimed (one live entry).
    assert cache_size() == 1


def test_different_weights_do_not_alias():
    w1 = _weight()
    w2 = _weight()
    cb = b"job-A" + b"\x00" * 27

    o1 = cached_noised_weight(w1, cb, lambda: torch.full_like(w1, 1))
    o2 = cached_noised_weight(w2, cb, lambda: torch.full_like(w2, 2))
    assert torch.equal(o1, torch.full_like(w1, 1))
    assert torch.equal(o2, torch.full_like(w2, 2))
    assert cache_size() == 2


def test_aba_freed_and_reused_data_ptr_does_not_false_hit():
    """If a weight is freed and a new tensor reuses the same data_ptr, the
    weakref check must miss and recompute rather than return the stale BpEB."""
    cb = b"job-A" + b"\x00" * 27
    calls = {"n": 0}

    def make_and_cache(val):
        w = _weight()
        calls["n"] += 1

        def compute(v=val, ww=w):
            return torch.full_like(ww, v)

        return cached_noised_weight(w, cb, compute), w

    out1, w1 = make_and_cache(1)
    ptr1 = w1.data_ptr()
    del w1, out1  # free the weight -> finalizer should evict its entry

    # Allocate until we (likely) reuse the address; correctness does not depend on
    # actually hitting the same data_ptr -- only that no false hit ever occurs.
    for _ in range(64):
        out2, w2 = make_and_cache(2)
        if w2.data_ptr() == ptr1:
            assert torch.equal(out2, torch.full_like(w2, 2)), "must be the NEW weight's BpEB"
            break
        del w2, out2


def test_weakref_finalizer_evicts_on_gc():
    cb = b"job-A" + b"\x00" * 27
    w = _weight()
    cached_noised_weight(w, cb, lambda ww=w: torch.full_like(ww, 3))
    assert cache_size() == 1
    del w
    import gc

    gc.collect()
    assert cache_size() == 0, "entry must self-evict when its weight is GC'd"
    assert cache_bytes() == 0


def test_byte_budget_is_tracked_and_reclaimed():
    cb = b"job-A" + b"\x00" * 27
    w = _weight(n=8, k=16)  # 128 int8 bytes
    out = cached_noised_weight(w, cb, lambda: torch.zeros_like(w))
    assert cache_bytes() == out.element_size() * out.nelement() == 128
    clear_cache()
    assert cache_bytes() == 0 and cache_size() == 0


def test_cache_stats_count_misses_hits_and_stores():
    w = _weight()
    cb = b"job-A" + b"\x00" * 27
    cached_noised_weight(w, cb, lambda: torch.zeros_like(w))
    cached_noised_weight(w, cb, lambda: torch.full_like(w, 1))

    stats = cache_stats()
    assert stats["lookups"] == 2
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["stores"] == 1
    assert stats["entries"] == 1
    assert stats["bytes"] == 128


def test_over_budget_miss_skips_admission_and_preserves_resident_entry():
    cb = b"job-A" + b"\x00" * 27
    w1 = _weight()
    w2 = _weight()
    calls = {"w1": 0, "w2": 0}

    def compute_w1():
        calls["w1"] += 1
        return torch.full_like(w1, 1)

    def compute_w2():
        calls["w2"] += 1
        return torch.full_like(w2, 2)

    out1 = cached_noised_weight(w1, cb, compute_w1, max_bytes=128)
    out2 = cached_noised_weight(w2, cb, compute_w2, max_bytes=128)
    out1_again = cached_noised_weight(w1, cb, compute_w1, max_bytes=128)

    assert calls == {"w1": 1, "w2": 1}
    assert out1_again is out1
    assert torch.equal(out2, torch.full_like(w2, 2))
    assert cache_size() == 1
    assert cache_bytes() == 128

    stats = cache_stats()
    assert stats["lookups"] == 3
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["stores"] == 1
    assert stats["admission_skips"] == 1
    assert stats["over_budget_clears"] == 0


def test_oversize_result_is_not_cached():
    cb = b"job-A" + b"\x00" * 27
    w = _weight()
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return torch.full_like(w, calls["n"])

    first = cached_noised_weight(w, cb, compute, max_bytes=64)
    second = cached_noised_weight(w, cb, compute, max_bytes=64)

    assert calls["n"] == 2
    assert not torch.equal(first, second)
    assert cache_size() == 0
    stats = cache_stats()
    assert stats["admission_skips"] == 2
    assert stats["oversize_skips"] == 2
