# Pearl Miner

Mining infrastructure for the Pearl network. Currently only mining via vLLM is supported, in the future we hope to supply plugins for other LLM inference libraries, like SGLang, TensorRT-LLM, Ollama, ...

## Directory Structure

```
miner/
├── miner-utils/            # Shared utilities (logging, common helpers)
├── miner-base/             # Core mining logic: async loop manager, block submission,
│                           #   commitment hashes, Merkle trees, gateway client
├── pearl-gemm-build-utils/ # Build-time config and code generation for pearl-gemm kernels
├── pearl-gemm/             # CUDA kernels: NoisyGEMM, noising/denoising, PoW extraction
│   └── csrc/               #   C++/CUDA source (uses NVIDIA CUTLASS)
├── pearl-gateway/          # Bridge between a Pearl full node and the miner process
│                           #   (work distribution, block submission, JSON-RPC)
└── vllm-miner/             # vLLM plugin that replaces quantized linear ops with NoisyGEMM
    └── Dockerfile          #   Production Docker image
```

## Prerequisites

- Python 3.12
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA toolkit and NVIDIA GPU (for `pearl-gemm` and running the miner)
- Rust toolchain (for `py-pearl-mining` dependencies)

## Build

All packages are managed as a uv workspace. Run the following from the **repository root**:

```bash
# Install all packages
uv sync

# Install only vllm-miner and its transitive dependencies
uv sync --package vllm-miner
```

The `pearl-gemm` CUDA kernels are compiled automatically during sync. Set the `PEARL_GEMM_FORCE_BUILD` environment variable to `TRUE` to force a rebuild, or set `PEARL_GEMM_SKIP_CUDA_BUILD=TRUE` to skip the CUDA build entirely.

## Tests

All commands run from the **repository root**. Use `-n auto` to run tests in parallel (requires `pytest-xdist`).

> **GPU requirement:** `pearl-gemm` and `vllm-miner` tests require an NVIDIA GPU. Currently only **sm90** (H100 / H200) GPUs are supported.

### Basic tests

Run the fast, self-contained unit tests (excludes `slow`, `integration`, `performance` markers and the vLLM execution suite):

```bash
uv run pytest -n auto \
  -m "not slow and not integration and not performance" \
  --ignore=miner/vllm-miner/tests/test_vllm_execution.py \
  miner/
```

To run tests for a single package:

```bash
uv run pytest -n auto miner/miner-base/tests/
uv run pytest -n auto miner/pearl-gateway/tests/
uv run pytest -n auto miner/pearl-gemm/tests/
uv run pytest -n auto miner/vllm-miner/tests/ \
  --ignore=miner/vllm-miner/tests/test_vllm_execution.py
```

### Slow tests

Tests marked `slow` typically take >30 seconds (large model loads, extended GPU workloads):

```bash
uv run pytest -n auto -m slow miner/
```

### Performance tests

Throughput and latency benchmarks:

```bash
uv run pytest -m performance miner/
```

### vLLM execution tests

End-to-end tests that start a full vLLM server with the Pearl plugin. These are heavyweight and run sequentially:

```bash
uv run pytest -v miner/vllm-miner/tests/test_vllm_execution.py
```

### Integration tests

Integration tests require a **local `pearld` node** running in simnet mode. See [`.github/workflows/integration_tests_ci.yml`](.github/workflows/integration_tests_ci.yml) for a full working example (node startup flags, environment variables, test invocation).

```bash
uv run pytest -v -m integration miner/
```

## Docker

### Build

Run from the **repository root** (the Dockerfile expects the full monorepo as build context):

```bash
docker buildx build -t vllm_miner . -f miner/vllm-miner/Dockerfile
```

### Run

The container starts `pearl-gateway` in the background and then launches `vllm serve`.

```bash
docker run --rm -it --gpus all \
  -p 8000:8000 -p 8337:8337 -p 8339:8339 \
  -e PEARLD_RPC_URL=<PEARLD URL> \
  -e PEARLD_RPC_USER=<USER> \
  -e PEARLD_RPC_PASSWORD=<PASSWORD>
  -e HF_TOKEN=<your-token> \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --shm-size 8g \
  vllm_miner:latest \
  pearl-ai/Llama-3.3-70B-Instruct-pearl \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager
```

### CUDA graphs (piecewise) — optional, recommended over `--enforce-eager`

The mining GEMM does per-forward-pass host work (reads the current mining job,
schedules CUDA-event callbacks, allocates scratch), which cannot be replayed
inside a CUDA graph. Historically the miner therefore ran fully eager
(`--enforce-eager`). vLLM's v1 engine instead supports **piecewise** CUDA
graphs: the model graph is split at declared "splitting ops", the subgraphs
between splits are captured/replayed, and the splitting ops run eager. The miner
registers its mining GEMM as the custom op `vllm_miner::noisy_gemm_mining`, so
declaring it a splitting op keeps mining eager while capturing the rest of the
model (norms, rotary, attention, small GEMMs, sampler).

Replace `--enforce-eager` with a piecewise compilation config:

```bash
  ... \
  --compilation-config '{"cudagraph_mode": "PIECEWISE", "splitting_ops": ["vllm::unified_attention_with_output", "vllm::unified_mla_attention_with_output", "vllm::mamba_mixer2", "vllm::mamba_mixer", "vllm::short_conv", "vllm::linear_attention", "vllm::plamo2_mamba_mixer", "vllm::gdn_attention_core", "vllm::gdn_attention_core_xpu", "vllm::olmo_hybrid_gdn_full_forward", "vllm::kda_attention", "vllm::sparse_attn_indexer", "vllm::rocm_aiter_sparse_attn_indexer", "vllm::deepseek_v4_attention", "vllm_miner::noisy_gemm_mining"]}'
```

The `splitting_ops` list **must** include vLLM's default attention ops plus
`vllm_miner::noisy_gemm_mining`; if you pass `splitting_ops`, vLLM does not
re-add its defaults. The exact default list can be generated with
`vllm_miner.piecewise.pearl_splitting_ops()`. Mining semantics are unchanged:
the mining op still runs eager between captured subgraphs.

`--enforce-eager` remains a valid fallback (no CUDA graphs at all) if you hit a
capture issue.

> **Status:** the piecewise recipe is validated at the op/compiler-plumbing
> level (the mining op is a single FX-graph node and is preserved in
> `splitting_ops`). End-to-end throughput numbers on H100 are pending.




