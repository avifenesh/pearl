import torch
from miner_base.commitment_hash import CommitmentHasher
from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
from miner_utils import get_logger
from pearl_gemm import (
    commitment_hash_from_merkle_roots,
    gemm,
    get_host_signal_sync_size,
    get_required_scratchpad_bytes,
    make_pow_target_tensor,
    noise_B,
    noise_gen,
    noisy_gemm,
    tensor_hash,
)

from .bpeb_cache import cached_noised_weight
from .callbacks import (
    StatusCheckCallback,
)
from .config import config
from .mining_state import (
    get_async_manager,
    get_pinned_pool,
)
from .weight_hash_cache import cached_weight_hash

_LOGGER = get_logger("vllm.pearl_miner")


def pearl_gemm_vanilla(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Performs standard quantized matrix multiplication without mining operations.

    Computes C = A @ B.T using optimized CUDA kernels for int8 quantized inputs.

    :param A: Input matrix A (int8, quantized)
    :param B: Input matrix B (int8, quantized)
    :param scale_a: Quantization scale factors for matrix A
    :param scale_b: Quantization scale factors for matrix B
    :param out_dtype: Output data type (bfloat16 or float16)
    :return: Result matrix C
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16

    A_scales = scale_a
    B_scales = scale_b

    C = torch.empty((A.shape[0], B.shape[0]), dtype=out_dtype, device=A.device)

    gemm(
        A=A,
        B=B,
        A_scales=A_scales,
        B_scales=B_scales,
        C=C,
        tile_size_m=config.settings.tile_size_m,
        tile_size_n=config.settings.tile_size_n,
        tile_size_k=config.settings.tile_size_k,
    )

    return C


def pearl_gemm_noisy(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    layer: torch.nn.Module | None = None,
    submit_block: bool = True,
) -> torch.Tensor:
    """
    Performs quantized matrix multiplication with cryptographic noise for blockchain mining.

    Computes C = A @ B.T while generating proof-of-work hashes from intermediate computations
    using optimized CUDA kernels for noise generation and matrix operations.

    :param a: Input matrix A (int8, quantized)
    :param b: Input matrix B (int8, quantized)
    :param scale_a: Quantization scale factors for matrix A
    :param scale_b: Quantization scale factors for matrix B
    :param out_dtype: Output data type (bfloat16 or float16)
    :param layer: Layer containing weight tensors
    :param submit_block: Whether to submit mining results
    :return: Result matrix C
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16

    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]
    r = config.settings.noise_rank
    A = a
    B = b
    A_scales = scale_a
    B_scales = scale_b
    C = torch.empty((m, n), dtype=out_dtype, device=a.device)

    matrix_bytes = max(m * k, n * k)
    tensor_hash_scratchpad = torch.empty(
        get_required_scratchpad_bytes(matrix_bytes),
        dtype=torch.uint8,
        device=a.device,
    )
    matmul_config = GPUMatmulConfigFactory.create(k=k, noise_rank=r)

    # Get current mining job from shared state
    mining_job = get_async_manager().get_mining_job()

    # Calculate adjusted pow_target
    adjusted_target = mining_job.adjust_target(mining_config=matmul_config.mining_config)

    hash_key = CommitmentHasher.get_key(
        mining_job.incomplete_header_bytes, matmul_config.mining_config
    )

    key_tensor = torch.frombuffer(bytearray(hash_key), dtype=torch.uint8).to("cuda")

    A_tensor_hash = torch.empty(32, device="cuda", dtype=torch.uint8)
    tensor_hash(
        A,
        key_tensor,
        A_tensor_hash,
        tensor_hash_scratchpad,
    )

    # B is the layer weight (constant for the process); its keyed merkle root is
    # cached across forward passes within a mining job (bit-identical to fresh).
    def _hash_weight(B=B, key_tensor=key_tensor, scratchpad=tensor_hash_scratchpad):
        out = torch.empty(32, device="cuda", dtype=torch.uint8)
        tensor_hash(B, key_tensor, out, scratchpad)
        return out

    B_tensor_hash = cached_weight_hash(B, hash_key, _hash_weight)

    # Generate commitment hash for noise generation
    commitment_hash_A_tensor = torch.empty(32, device="cuda", dtype=torch.uint8)
    commitment_hash_B_tensor = torch.empty(32, device="cuda", dtype=torch.uint8)
    commitment_hash_from_merkle_roots(
        A_tensor_hash,
        B_tensor_hash,
        key_tensor,
        commitment_hash_A_tensor,
        commitment_hash_B_tensor,
    )

    # Generate noise factors from commitment hashes
    (
        EAL,
        EAR_R_major,
        EBL_R_major,
        EAR_K_major,
        EBL_K_major,
        EBR,
        EAL_fp16,
        EBR_fp16,
    ) = generate_noise_factors(
        m,
        n,
        k,
        r,
        commitment_hash_A_tensor,
        commitment_hash_B_tensor,
        a.device,
    )

    # B-side noising. BpEB = B + EBL@EBR is A-INDEPENDENT (seeded by commitment_B
    # = blake3(job_key || merkle_root(B))), so it is byte-identical across every
    # forward pass within a mining job (proven vs miner_base; and on H100 the GEMM
    # output C is bit-exact when a precomputed BpEB is supplied with
    # run_noising_B=False). The mining job rotates ~1/s while a forward is ~ms, so
    # caching BpEB per (weight, job) turns the dominant, m-independent weight-noising
    # (forming EBL@EBR + the n x k int8 HBM write, ~0.17 ms/layer on gate_up) into a
    # once-per-job cost that amortizes to ~0 over a long prefill.
    #
    # The denoise side-product EARxBpEB = BpEB @ EAR is A-DEPENDENT (EAR is seeded by
    # commitment_A) and is recomputed fresh every call from the cached BpEB. Its
    # values are small integers that fit exactly in fp32, so an fp32 matmul with TF32
    # disabled is bit-exact and GPU-native (CUDA has no int matmul). The PoW path is
    # therefore bit-unchanged on a hit.
    BpEB = torch.empty((n, k), dtype=torch.int8, device=a.device)
    EARxBpEB = torch.empty((n, r), dtype=torch.float16, device=a.device)

    use_bpeb_cache = (
        config.settings.cache_noised_weight
        and layer is not None
        and B.is_contiguous()
    )
    bpeb_cached = None
    EARxBpEB_int32 = None
    if use_bpeb_cache:
        commitment_b_bytes = bytes(commitment_hash_B_tensor.cpu().numpy().tobytes())

        def _form_bpeb(B=B, EBR=EBR, EAR_K_major=EAR_K_major, EBL_R_major=EBL_R_major):
            # Form BpEB once per (weight, job) via the standalone noise_B op (the
            # EARxBpEB it also writes is scratch — recomputed fresh below).
            bpeb_out = torch.empty((n, k), dtype=torch.int8, device=a.device)
            scratch = torch.empty((n, r), dtype=torch.float16, device=a.device)
            noise_B(B=B, EBR=EBR, EARxBpEB=scratch, BpEB=bpeb_out,
                    EAR=EAR_K_major, EBL=EBL_R_major)
            return bpeb_out

        bpeb_cached = cached_noised_weight(B, commitment_b_bytes, _form_bpeb)
        BpEB = bpeb_cached
        # EARxBpEB = BpEB(n x k) @ EAR_R_major(k x r) -> (n x r), as int32 (the
        # kernel's denoise converter scales int32 -> fp16). fp32 + TF32 off is
        # bit-exact for these small-integer values; verified on H100.
        EARxBpEB_int32 = torch.empty((n, r), dtype=torch.int32, device=a.device)
        _prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        EARxBpEB_int32.copy_(
            (BpEB.to(torch.float32) @ EAR_R_major.to(torch.float32)).to(torch.int32)
        )
        torch.backends.cuda.matmul.allow_tf32 = _prev_tf32

    # Allocate A noising tensors (input-dependent)
    ApEA = torch.empty((m, k), dtype=torch.int8, device=a.device)
    A_E_BL = torch.empty((m, r), dtype=torch.float16, device=a.device)

    host_signal_sync_size = get_host_signal_sync_size()
    host_signal_sync = torch.zeros((host_signal_sync_size,), dtype=torch.int8, device="cuda")
    host_signal_header_pinned = get_pinned_pool().acquire()

    # Create pow_target tensor from adjusted_target
    pow_target_tensor = make_pow_target_tensor(adjusted_target)

    # Run noisy GEMM with default kernel configurations
    noisy_gemm(
        A=A,  # Input matrix A (m x k)
        B=B,  # Input matrix B (n x k)
        EAL=EAL,  # Noise factor E_AL (m x r)
        EAL_fp16=EAL_fp16,  # fp16 version
        EBR=EBR,  # Noise factor E_BR (n x r)
        EBR_fp16=EBR_fp16,  # fp16 version
        EAR_R_major=EAR_R_major,
        EBL_R_major=EBL_R_major,
        EAR_K_major=EAR_K_major,
        EBL_K_major=EBL_K_major,
        AxEBL_fp16=A_E_BL,  # Intermediate tensor A * E_BL (m x r)
        EARxBpEB_fp16=EARxBpEB,  # fp16 output buffer (denoise-converter target on the cached path)
        EARxBpEB_int32=EARxBpEB_int32,  # set on a cache hit; the GEMM skips B-noising
        ApEA=ApEA,  # Output tensor for A + EA (m x k)
        BpEB=BpEB,  # Output tensor for B + EB (n x k)
        A_scales=A_scales,  # Scale factors for A
        B_scales=B_scales,  # Scale factors for B
        C=C,  # Output matrix C (m x n)
        host_signal_header_pinned=host_signal_header_pinned,
        host_signal_sync=host_signal_sync,
        pow_target=pow_target_tensor,
        pow_key=commitment_hash_A_tensor.view(torch.uint32),
        tile_size_m=config.settings.tile_size_m,
        tile_size_n=config.settings.tile_size_n,
        tile_size_k=config.settings.tile_size_k,
        run_noising_A=True,  # run_noising_A
        run_noising_B=not use_bpeb_cache,  # skip B-noising when BpEB is cached
        skip_reduction=False,  # skip_reduction
        skip_denoising=False,  # skip_denoising
    )

    if submit_block:
        # Record a CUDA event after the kernel launch - will complete when kernel finishes
        cuda_event = torch.cuda.Event()
        cuda_event.record()

        # Create callback for processing the status check
        callback = StatusCheckCallback(
            host_signal_header_pinned=host_signal_header_pinned,
            commitment_hash_A_tensor=commitment_hash_A_tensor,
            commitment_hash_B_tensor=commitment_hash_B_tensor,
            A=A,
            B=B,
            mining_job=mining_job,
        )

        get_async_manager().schedule_status_check(cuda_event, callback)

        # Callback owns these tensors
        host_signal_header_pinned = None
        commitment_hash_A_tensor = None
        commitment_hash_B_tensor = None
    else:
        get_pinned_pool().release(host_signal_header_pinned)
        del host_signal_header_pinned
        del commitment_hash_A_tensor
        del commitment_hash_B_tensor

    del pow_target_tensor
    del ApEA
    # BpEB may be cache-owned (the cache holds a ref); drop only our local binding.
    if bpeb_cached is None:
        del BpEB
    del A_E_BL
    del EAL
    del EBR
    del EAR_R_major
    del EBL_R_major
    del EAR_K_major
    del EBL_K_major
    del EAL_fp16
    del EBR_fp16
    del key_tensor
    del A_tensor_hash
    del B_tensor_hash
    del tensor_hash_scratchpad
    del host_signal_sync
    del EARxBpEB
    if EARxBpEB_int32 is not None:
        del EARxBpEB_int32
    return C


def generate_noise_factors(
    m: int,
    n: int,
    k: int,
    r: int,
    commitment_hash_A: torch.Tensor,
    commitment_hash_B: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Generates cryptographic noise factors for noisy GEMM operations.

    Creates low-rank noise matrices using commitment hashes as seeds for deterministic
    noise generation in blockchain mining operations using optimized CUDA kernels.

    :param m: Number of rows in matrix A
    :param n: Number of rows in matrix B
    :param k: Number of columns in matrices A and B
    :param r: Rank of the noise matrices
    :param commitment_hash_A: Cryptographic hash for matrix A noise generation
    :param commitment_hash_B: Cryptographic hash for matrix B noise generation
    :param device: CUDA device for tensor allocation
    :return: Tuple of noise tensors (EAL, EAR_R_major, EBL_R_major,
             EAR_K_major, EBL_K_major, EBR, EAL_fp16, EBR_fp16)
    """
    EAL = torch.empty((m, r), dtype=torch.int8, device=device)
    EBR = torch.empty((n, r), dtype=torch.int8, device=device)

    EAR_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
    EBL_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
    EAR_K_major = torch.empty((r, k), dtype=torch.int8, device=device)
    EBL_K_major = torch.empty((r, k), dtype=torch.int8, device=device)
    EAL_fp16 = torch.empty((m, r), dtype=torch.float16, device=device)
    EBR_fp16 = torch.empty((n, r), dtype=torch.float16, device=device)

    noise_gen(
        R=r,
        EAL=EAL,
        EAL_fp16=EAL_fp16,
        EAR_R_major=EAR_R_major,
        EAR_K_major=EAR_K_major,
        EBL_R_major=EBL_R_major,
        EBL_K_major=EBL_K_major,
        EBR=EBR,
        EBR_fp16=EBR_fp16,
        key_A=commitment_hash_A,
        key_B=commitment_hash_B,
    )
    return (
        EAL,
        EAR_R_major,
        EBL_R_major,
        EAR_K_major,
        EBL_K_major,
        EBR,
        EAL_fp16,
        EBR_fp16,
    )
