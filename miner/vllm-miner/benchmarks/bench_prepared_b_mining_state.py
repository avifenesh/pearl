"""Benchmark cold vs warm PreparedBMiningState for dense noisy GEMM.

Example:
    uv run --package vllm-miner python \
        miner/vllm-miner/benchmarks/bench_prepared_b_mining_state.py
"""

from __future__ import annotations

import argparse
import statistics

import torch
from miner_base.settings import MinerSettings
from vllm_miner.gemm_operators import pearl_gemm_noisy
from vllm_miner.mining_state import delete_state, init_async_manager, init_pinned_pool
from vllm_miner.prepared_b_mining_state import (
    clear_prepared_b_cache,
    prepared_b_cache_bytes,
    prepared_b_cache_size,
    prepared_b_cache_stats,
    reset_prepared_b_cache_stats,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dense noisy GEMM with cold and warm PreparedBMiningState."
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=28672)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cache-bytes", type=int, default=1 << 30)
    parser.add_argument("--allow-non-h100", action="store_true")
    return parser.parse_args()


def _require_gpu(allow_non_h100: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    name = torch.cuda.get_device_name(0)
    if "H100" not in name and not allow_non_h100:
        raise RuntimeError(f"Expected H100; got {name}. Pass --allow-non-h100 to override.")


def _make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    a = torch.randint(-63, 64, (args.m, args.k), dtype=torch.int8, device="cuda")
    b = torch.randint(-63, 64, (args.n, args.k), dtype=torch.int8, device="cuda")
    scale_a = torch.ones((args.m,), dtype=torch.float32, device="cuda")
    scale_b = torch.ones((args.n,), dtype=torch.float32, device="cuda")
    return a, b, scale_a, scale_b


def _one_call(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> None:
    out = pearl_gemm_noisy(a, b, scale_a, scale_b, torch.bfloat16, submit_block=False)
    # Keep the output live until after kernel launch.
    del out


def _measure(
    mode: str,
    args: argparse.Namespace,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> dict[str, float | int | str]:
    clear_prepared_b_cache()

    if mode == "warm":
        _one_call(a, b, scale_a, scale_b)
        torch.cuda.synchronize()
        reset_prepared_b_cache_stats()
    elif mode != "cold":
        raise ValueError(f"unknown mode: {mode}")

    samples = []
    for _ in range(args.warmup):
        if mode == "cold":
            clear_prepared_b_cache(reset_stats=False)
        _one_call(a, b, scale_a, scale_b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iterations):
        if mode == "cold":
            clear_prepared_b_cache(reset_stats=False)
        start.record()
        _one_call(a, b, scale_a, scale_b)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))

    stats = prepared_b_cache_stats()
    return {
        "mode": mode,
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "avg_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "cache_entries": prepared_b_cache_size(),
        "cache_bytes": prepared_b_cache_bytes(),
        "hits": stats.hits,
        "misses": stats.misses,
        "evictions": stats.evictions,
        "oversize": stats.oversize,
    }


def _print_rows(rows: list[dict[str, float | int | str]]) -> None:
    print(
        "mode,m,n,k,avg_ms,median_ms,cache_entries,cache_bytes,"
        "hits,misses,evictions,oversize",
        flush=True,
    )
    for row in rows:
        print(
            f"{row['mode']},{row['m']},{row['n']},{row['k']},"
            f"{row['avg_ms']:.3f},{row['median_ms']:.3f},"
            f"{row['cache_entries']},{row['cache_bytes']},"
            f"{row['hits']},{row['misses']},{row['evictions']},{row['oversize']}",
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    _require_gpu(args.allow_non_h100)
    settings = MinerSettings(
        no_gateway=True,
        skip_block_submission=True,
        prepared_b_cache_bytes=args.cache_bytes,
    )
    init_async_manager(settings)
    init_pinned_pool(settings.pinned_pool_size)
    try:
        a, b, scale_a, scale_b = _make_inputs(args)
        rows = [
            _measure("cold", args, a, b, scale_a, scale_b),
            _measure("warm", args, a, b, scale_a, scale_b),
        ]
        _print_rows(rows)
    finally:
        clear_prepared_b_cache()
        delete_state()


if __name__ == "__main__":
    main()
