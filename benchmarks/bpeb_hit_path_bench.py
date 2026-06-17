from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch
from pearl_gemm import (
    bpeb_ear_product,
    denoise_converter,
    noise_B,
    noise_B_side_product,
    noisy_gemm,
)
from pearl_gemm.testing import GEMMParam, GemmTensorGenerator


def _sync() -> None:
    torch.cuda.synchronize()


def _measure_ms(fn: Callable[[], None], *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    _sync()

    times: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _summary(times: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(times)
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p10_ms": ordered[max(0, int(0.10 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(len(ordered) - 1, int(0.90 * (len(ordered) - 1)))],
        "times_ms": times,
    }


def _new_generator(args: argparse.Namespace) -> tuple[GEMMParam, GemmTensorGenerator]:
    params = GEMMParam(
        m=args.m,
        n=args.n,
        k=args.k,
        R=args.r,
        tile_size_m=args.tile_size_m,
        tile_size_n=args.tile_size_n,
        tile_size_k=args.tile_size_k,
        pipeline_stages=args.pipeline_stages,
        tile_size_m_noising_A=args.tile_size_m_noising_a,
        tile_size_n_noising_B=args.tile_size_n_noising_b,
        tile_size_k_noising_A=args.tile_size_k_noising_a,
        tile_size_k_noising_B=args.tile_size_k_noising_b,
        pipeline_stages_noising_A=args.pipeline_stages_noising_a,
        pipeline_stages_noising_B=args.pipeline_stages_noising_b,
        AxEBL_type_noising=torch.float16,
        EARxBpEB_type_noising=torch.float16,
        k_blocks_per_split_noising_A=args.k_blocks_per_split_noising_a,
        k_blocks_per_split_noising_B=args.k_blocks_per_split_noising_b,
        skip_reduction=args.skip_reduction,
    )
    generator = GemmTensorGenerator(params)
    generator.generate(pow_target=args.pow_target)
    return params, generator


def _run_noise_b(params: GEMMParam, tg: GemmTensorGenerator) -> None:
    noise_B(
        tg.B,
        tg.EBR,
        tg.EARxBpEB,
        tg.BpEB,
        EAR=tg.EAR_K_major,
        EBL=tg.EBL_R_major,
        tile_size_n=params.tile_size_n_noising_B,
        tile_size_k=params.tile_size_k_noising_B,
        pipeline_stages=params.pipeline_stages_noising_B,
        k_blocks_per_split=params.k_blocks_per_split_noising_B,
    )


def _run_noise_b_side_product(params: GEMMParam, tg: GemmTensorGenerator) -> None:
    noise_B_side_product(
        tg.B,
        tg.EBR,
        tg.EARxBpEB,
        EAR=tg.EAR_K_major,
        EBL=tg.EBL_R_major,
        tile_size_n=params.tile_size_n_noising_B,
        tile_size_k=params.tile_size_k_noising_B,
        pipeline_stages=params.pipeline_stages_noising_B,
        k_blocks_per_split=params.k_blocks_per_split_noising_B,
    )


def _run_bpeb_ear_product(params: GEMMParam, tg: GemmTensorGenerator) -> None:
    assert tg.EARxBpEB_int32 is not None
    bpeb_ear_product(
        tg.BpEB,
        tg.EAR_K_major,
        tg.EARxBpEB_int32,
        tile_size_n=params.tile_size_n_noising_B,
        tile_size_k=params.tile_size_k_noising_B,
        pipeline_stages=params.pipeline_stages_noising_B,
        k_blocks_per_split=params.k_blocks_per_split_noising_B,
    )


def _run_denoise_converter(tg: GemmTensorGenerator) -> None:
    denoise_converter(
        EARxBpEB_in=tg.EARxBpEB_int32,
        EARxBpEB_out=tg.EARxBpEB_fp16,
    )


def _run_noisy_gemm(
    params: GEMMParam,
    tg: GemmTensorGenerator,
    *,
    run_noising_b: bool,
    use_ear_int32: bool,
) -> None:
    noisy_gemm(
        A=tg.A,
        B=tg.B,
        EAL=tg.EAL,
        EAL_fp16=tg.EAL_fp16,
        EBR=tg.EBR,
        EBR_fp16=tg.EBR_fp16,
        EAR_R_major=tg.EAR_R_major,
        EBL_R_major=tg.EBL_R_major,
        EAR_K_major=tg.EAR_K_major,
        EBL_K_major=tg.EBL_K_major,
        AxEBL_fp16=tg.AxEBL_fp16,
        EARxBpEB_fp16=tg.EARxBpEB_fp16,
        ApEA=tg.ApEA,
        BpEB=tg.BpEB,
        A_scales=tg.A_scales,
        B_scales=tg.B_scales,
        C=tg.C,
        host_signal_header_pinned=tg.host_signal_header_pinned,
        host_signal_sync=tg.host_signal_sync,
        pow_target=tg.pow_target,
        pow_key=tg.pow_key,
        AxEBL_int32=tg.AxEBL_int32,
        EARxBpEB_int32=tg.EARxBpEB_int32 if use_ear_int32 else None,
        tile_size_m=params.tile_size_m,
        tile_size_n=params.tile_size_n,
        tile_size_k=params.tile_size_k,
        pipeline_stages=params.pipeline_stages,
        cluster_size_m=params.cluster_size_m,
        cluster_size_n=params.cluster_size_n,
        swizzle=params.swizzle,
        swizzle_n_maj=params.swizzle_n_maj,
        tile_size_m_noising_A=params.tile_size_m_noising_A,
        tile_size_n_noising_B=params.tile_size_n_noising_B,
        tile_size_k_noising_A=params.tile_size_k_noising_A,
        tile_size_k_noising_B=params.tile_size_k_noising_B,
        pipeline_stages_noising_A=params.pipeline_stages_noising_A,
        pipeline_stages_noising_B=params.pipeline_stages_noising_B,
        k_blocks_per_split_noising_A=params.k_blocks_per_split_noising_A,
        k_blocks_per_split_noising_B=params.k_blocks_per_split_noising_B,
        run_noising_A=True,
        run_noising_B=run_noising_b,
        skip_reduction=params.skip_reduction,
        skip_denoising=False,
    )


def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)

    full_params, full_tg = _new_generator(args)
    cached_params, cached_tg = _new_generator(args)

    _run_noise_b(full_params, full_tg)
    expected_ear = full_tg.EARxBpEB.clone()
    full_tg.EARxBpEB.zero_()
    _run_noise_b_side_product(full_params, full_tg)
    torch.testing.assert_close(full_tg.EARxBpEB, expected_ear, rtol=0, atol=0)
    del expected_ear

    cached_tg._EARxBpEB_int32 = torch.empty(
        (cached_tg.num_stages, args.n, args.r), dtype=torch.int32, device="cuda"
    )
    cached_tg._EARxBpEB_fp16 = torch.empty(
        (cached_tg.num_stages, args.n, args.r), dtype=torch.float16, device="cuda"
    )
    cached_tg._EARxBpEB = cached_tg._EARxBpEB_int32

    # Seed the cached path with the exact BpEB tensor it would keep resident.
    _run_noise_b(full_params, cached_tg)
    _run_bpeb_ear_product(cached_params, cached_tg)
    _run_denoise_converter(cached_tg)
    _sync()

    benchmarks: dict[str, Callable[[], None]] = {
        "noise_B_fp16": lambda: _run_noise_b(full_params, full_tg),
        "noise_B_side_product_fp16": lambda: _run_noise_b_side_product(
            full_params, full_tg
        ),
        "bpeb_ear_product_int32": lambda: _run_bpeb_ear_product(
            cached_params, cached_tg
        ),
        "denoise_converter_ear_only": lambda: _run_denoise_converter(cached_tg),
        "noisy_gemm_full": lambda: _run_noisy_gemm(
            full_params, full_tg, run_noising_b=True, use_ear_int32=False
        ),
        "noisy_gemm_skip_b": lambda: _run_noisy_gemm(
            cached_params, cached_tg, run_noising_b=False, use_ear_int32=True
        ),
        "cached_hit_sequence": lambda: (
            _run_bpeb_ear_product(cached_params, cached_tg),
            _run_noisy_gemm(
                cached_params, cached_tg, run_noising_b=False, use_ear_int32=True
            ),
        ),
    }

    result = {
        "device": torch.cuda.get_device_name(args.device),
        "shape": {"m": args.m, "n": args.n, "k": args.k, "r": args.r},
        "config": {
            "tile_size_m": args.tile_size_m,
            "tile_size_n": args.tile_size_n,
            "tile_size_k": args.tile_size_k,
            "pipeline_stages": args.pipeline_stages,
            "tile_size_m_noising_A": args.tile_size_m_noising_a,
            "tile_size_n_noising_B": args.tile_size_n_noising_b,
            "tile_size_k_noising_A": args.tile_size_k_noising_a,
            "tile_size_k_noising_B": args.tile_size_k_noising_b,
            "pipeline_stages_noising_A": args.pipeline_stages_noising_a,
            "pipeline_stages_noising_B": args.pipeline_stages_noising_b,
            "k_blocks_per_split_noising_A": args.k_blocks_per_split_noising_a,
            "k_blocks_per_split_noising_B": args.k_blocks_per_split_noising_b,
            "skip_reduction": args.skip_reduction,
        },
        "correctness": {"noise_B_side_product_matches_noise_B": True},
        "benchmarks": {},
    }

    for name, fn in benchmarks.items():
        result["benchmarks"][name] = _summary(
            _measure_ms(fn, warmup=args.warmup, iters=args.iters)
        )

    full_ms = result["benchmarks"]["noisy_gemm_full"]["median_ms"]
    cached_ms = result["benchmarks"]["cached_hit_sequence"]["median_ms"]
    noise_b_ms = result["benchmarks"]["noise_B_fp16"]["median_ms"]
    side_product_ms = result["benchmarks"]["noise_B_side_product_fp16"]["median_ms"]
    result["comparison"] = {
        "cached_minus_full_median_ms": cached_ms - full_ms,
        "cached_vs_full_speedup": full_ms / cached_ms,
        "noise_B_side_product_minus_full_median_ms": side_product_ms - noise_b_ms,
        "noise_B_side_product_vs_full_speedup": noise_b_ms / side_product_ms,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2770)
    parser.add_argument("--n", type=int, default=14336)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--r", type=int, default=128)
    parser.add_argument("--tile-size-m", type=int, default=128)
    parser.add_argument("--tile-size-n", type=int, default=256)
    parser.add_argument("--tile-size-k", type=int, default=128)
    parser.add_argument("--pipeline-stages", type=int, default=3)
    parser.add_argument("--tile-size-m-noising-a", type=int, default=None)
    parser.add_argument("--tile-size-n-noising-b", type=int, default=None)
    parser.add_argument("--tile-size-k-noising-a", type=int, default=None)
    parser.add_argument("--tile-size-k-noising-b", type=int, default=None)
    parser.add_argument("--pipeline-stages-noising-a", type=int, default=2)
    parser.add_argument("--pipeline-stages-noising-b", type=int, default=2)
    parser.add_argument("--k-blocks-per-split-noising-a", type=int, default=None)
    parser.add_argument("--k-blocks-per-split-noising-b", type=int, default=None)
    parser.add_argument("--skip-reduction", action="store_true")
    parser.add_argument("--pow-target", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
