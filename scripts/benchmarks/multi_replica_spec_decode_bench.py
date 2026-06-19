#!/usr/bin/env python3
"""Multi-replica real-prompt speculative decoding benchmark.

Starts one vLLM replica per GPU for each case and load-balances the same prompt
bucket workload across replicas. This is for deployment-style p5 assessment
where total throughput matters, not one-GPU proxy throughput.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

BASE_SCRIPT = Path(__file__).with_name("real_prompt_spec_decode_bench.py")
spec = importlib.util.spec_from_file_location("real_prompt_spec_decode_bench", BASE_SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load {BASE_SCRIPT}")
rb = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rb
spec.loader.exec_module(rb)


def log(msg: str) -> None:
    rb.log(msg)


class Server:
    def __init__(self, gpu: str, port: int, process: subprocess.Popen[Any], base_url: str, log_path: Path):
        self.gpu = gpu
        self.port = port
        self.process = process
        self.base_url = base_url
        self.log_path = log_path


def metric_sum(diffs: list[dict[str, Any]]) -> dict[str, Any]:
    out = {name: 0.0 for name in rb.SPEC_METRIC_NAMES}
    per_pos: dict[int, float] = {}
    for diff in diffs:
        for name in rb.SPEC_METRIC_NAMES:
            out[name] += float(diff.get(name, 0.0))
        for pos, val in diff.get("per_pos", {}).items():
            per_pos[int(pos)] = per_pos.get(int(pos), 0.0) + float(val)
    out["per_pos"] = per_pos
    return out


def start_servers(args: argparse.Namespace, case: Any, case_idx: int) -> list[Server]:
    servers: list[Server] = []
    case_dir = Path(args.output_dir) / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    for replica_idx, gpu in enumerate(args.gpus_list):
        port = args.port_base + case_idx * 100 + replica_idx
        base_url = f"http://{args.host}:{port}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if args.miner_no_gateway:
            env["MINER_NO_GATEWAY"] = "1"
        if args.skip_block_submission:
            env["MINER_SKIP_BLOCK_SUBMISSION"] = "1"
        if case.mining:
            env.pop("MINER_NO_MINING", None)
        else:
            env["MINER_NO_MINING"] = "1"
        cmd = rb.build_server_command(args, case, port)
        log_path = case_dir / f"server_gpu{gpu}.log"
        log(f"case={case.name} replica={replica_idx} gpu={gpu} starting vLLM port={port}")
        log("server command: " + " ".join(cmd))
        fh = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, text=True)
        # Keep file handle attached via process object for CPython lifetime; close parent copy.
        fh.close()
        servers.append(Server(str(gpu), port, proc, base_url, log_path))
    for server in servers:
        rb.wait_for_server(server.base_url, server.process, args.server_timeout_s)
        log(f"case={case.name} gpu={server.gpu} server ready")
    return servers


def stop_servers(case_name: str, servers: list[Server]) -> None:
    for server in servers:
        if server.process.poll() is None:
            log(f"case={case_name} gpu={server.gpu} stopping vLLM")
            server.process.terminate()
    for server in servers:
        try:
            server.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=20)


def run_case_multi(args: argparse.Namespace, case: Any, prompts_by_group: dict[str, list[Any]], case_idx: int) -> dict[str, Any]:
    servers = start_servers(args, case, case_idx)
    try:
        group_summaries: dict[str, dict[str, Any]] = {}
        all_results: list[dict[str, Any]] = []
        total_concurrency = max(1, args.per_gpu_concurrency * len(servers))
        for group, records in prompts_by_group.items():
            befores = [rb.fetch_metrics(s.base_url) for s in servers]
            start = time.time()
            workers = min(total_concurrency, len(records))
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = []
                for i, record in enumerate(records):
                    server = servers[i % len(servers)]
                    futures.append(
                        pool.submit(
                            rb.request_completion,
                            base_url=server.base_url,
                            model_name=args.served_model_name,
                            prompt=record,
                            max_tokens=args.max_tokens,
                            request_timeout_s=args.request_timeout_s,
                        )
                    )
                results = [future.result() for future in futures]
            elapsed_s = time.time() - start
            afters = [rb.fetch_metrics(s.base_url) for s in servers]
            diffs = [rb.metric_diff(b, a) for b, a in zip(befores, afters)]
            metrics = metric_sum(diffs)
            summary = rb.summarize_group(
                case=case,
                group=group,
                records=records,
                results=results,
                elapsed_s=elapsed_s,
                metrics=metrics,
                max_tokens=args.max_tokens,
                concurrency=workers,
            )
            summary["num_replicas"] = len(servers)
            summary["gpus"] = args.gpus_list
            summary["per_gpu_concurrency"] = args.per_gpu_concurrency
            summary["total_requested_concurrency"] = total_concurrency
            group_summaries[group] = summary
            all_results.extend(results)
            log(
                f"case={case.name} group={group} replicas={len(servers)} "
                f"per_gpu_c={args.per_gpu_concurrency} total_c={workers} "
                f"req/s={summary['requests_per_second']:.3f} "
                f"out_tok/s={summary['output_tokens_per_second']:.1f} "
                f"acc_len={summary['acceptance_length']}"
            )
        case_dir = Path(args.output_dir) / case.name
        (case_dir / "summary.json").write_text(
            json.dumps({"case": asdict(case), "groups": group_summaries, "results": all_results}, indent=2),
            encoding="utf-8",
        )
        return group_summaries
    finally:
        stop_servers(case.name, servers)


def write_outputs(output_dir: Path, all_summaries: dict[str, dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "combined_summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    fields = [
        "case", "group", "mining", "spec_k", "num_replicas", "per_gpu_concurrency", "total_requested_concurrency",
        "num_requests", "avg_prompt_len", "requests_per_second", "output_tokens_per_second", "total_tokens_per_second",
        "acceptance_length", "avg_draft_acceptance_rate", "acceptance_at_pos_0", "acceptance_at_pos_1", "acceptance_at_pos_2", "acceptance_at_pos_3",
    ]
    with (output_dir / "combined_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for case_groups in all_summaries.values():
            for summary in case_groups.values():
                row = {k: summary.get(k) for k in fields}
                pos = summary.get("acceptance_at_pos", {}) or {}
                for i in range(4):
                    row[f"acceptance_at_pos_{i}"] = pos.get(str(i))
                writer.writerow(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-replica real prompt spec decode bench")
    p.add_argument("--model", required=True)
    p.add_argument("--served-model-name", default="pearl")
    p.add_argument("--eagle-checkpoint", required=True)
    p.add_argument("--eagle-method", default="eagle3")
    p.add_argument("--vllm-bin", default="vllm")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dataset-repo", default=rb.DEFAULT_SUBSETS and "RedHatAI/speculator_benchmarks")
    p.add_argument("--subsets", default=rb.DEFAULT_SUBSETS)
    p.add_argument("--hf-cache-dir", default=None)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--bucket-lengths", default=None)
    p.add_argument("--samples-per-subset", type=int, default=24)
    p.add_argument("--samples-per-bucket", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=96)
    p.add_argument("--per-gpu-concurrency", type=int, default=4)
    p.add_argument("--cases", default="eagle_k3_mining,eagle_k4_mining,eagle_k5_mining")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port-base", type=int, default=11000)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    p.add_argument("--server-timeout-s", type=int, default=1800)
    p.add_argument("--request-timeout-s", type=int, default=900)
    p.add_argument("--seed", type=int, default=20260630)
    p.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--disable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--miner-no-gateway", dest="miner_no_gateway", action="store_true", default=True)
    p.add_argument("--with-gateway", dest="miner_no_gateway", action="store_false")
    p.add_argument("--skip-block-submission", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args(argv)
    args.gpus_list = [x.strip() for x in args.gpus.split(",") if x.strip()]
    # rb.build_server_command expects these names.
    args.concurrency = args.per_gpu_concurrency * len(args.gpus_list)
    args.gpu = args.gpus
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [rb.parse_case(c) for c in args.cases.split(",") if c.strip()]
    prompts = rb.load_prompt_records(args)
    prompts_by_group: dict[str, list[Any]] = {}
    for prompt in prompts:
        prompts_by_group.setdefault(prompt.group, []).append(prompt)
    (output_dir / "prompt_manifest.json").write_text(json.dumps([asdict(p) for p in prompts], indent=2), encoding="utf-8")
    log("selected prompts=" + str(len(prompts)) + " groups=" + ",".join(f"{g}:{len(r)}" for g, r in prompts_by_group.items()))
    all_summaries: dict[str, dict[str, Any]] = {}
    for idx, case in enumerate(cases, start=1):
        summaries = run_case_multi(args, case, prompts_by_group, idx)
        all_summaries[case.name] = summaries
        write_outputs(output_dir, all_summaries)
    log("multi-replica combined results:")
    for case_name, groups in all_summaries.items():
        for group, summary in groups.items():
            log(f"{case_name}/{group}: req/s={summary['requests_per_second']:.3f} out_tok/s={summary['output_tokens_per_second']:.1f} acc_len={summary['acceptance_length']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
