import math
import os

import pytest
import torch
from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
from miner_base.matmul_config import MatmulConfig
from pearl_gemm import (
    bpeb_ear_product,
    denoise_converter,
    gemm,
    noise_A,
    noise_B,
    noise_B_side_product,
    noisy_gemm,
)
from pearl_gemm.testing import GEMMParam, GemmTensorGenerator

DISABLE_R32 = os.getenv("PEARL_GEMM_DISABLE_R32", "TRUE") == "TRUE"
DISABLE_R64 = os.getenv("PEARL_GEMM_DISABLE_R64", "FALSE") == "TRUE"
DISABLE_R128 = os.getenv("PEARL_GEMM_DISABLE_R128", "FALSE") == "TRUE"
Rs = (
    []
    + ([32] if not DISABLE_R32 else [])
    + ([64] if not DISABLE_R64 else [])
    + ([128] if not DISABLE_R128 else [])
)

DISABLE_DEBUG_MODE = os.getenv("PEARL_GEMM_DISABLE_DEBUG_MODE", "FALSE") == "TRUE"

# Run tests against default compiled kernels
from pearl_gemm_build_utils.kernel_configs.default_compiled_kernels import (  # noqa: E402
    KERNEL_CONFIGS,
)

# Get kernel configs and filter by Rs
kernel_grid = KERNEL_CONFIGS
matmul_kernels = [k for k in kernel_grid.matmul_kernels if k.R in Rs]
noise_a_kernels = [k for k in kernel_grid.noising_a_kernels if k.R in Rs]
noise_b_kernels = [k for k in kernel_grid.noising_b_kernels if k.R in Rs]

# For torch.compile tests, allow many recompilations
torch._dynamo.config.cache_size_limit = 256


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Base Pearl GEMM test class with util functions. Because of parameterize, it's difficult to
#  select individual test functions. So the tests are grouped into classes. Tests should
#  extend this class.
class TestPearlGEMMBase:
    # Always reset seed
    def setup_method(self):
        torch.random.manual_seed(1)

    def run_denoise_converter(self, tensor_generator, gemm_params):
        denoise_converter(
            tensor_generator.EARxBpEB_int32,
            tensor_generator.AxEBL_int32,
            tensor_generator.EARxBpEB_fp16,
            tensor_generator.AxEBL_fp16,
        )
        torch.cuda.synchronize()

    #### Wrapping pearl GEMM interface ####
    def run_noise_A(self, tensor_generator, gemm_params):
        noise_A(
            tensor_generator.A,
            tensor_generator.EAL,
            tensor_generator.AxEBL,
            tensor_generator.ApEA,
            EAR=tensor_generator.EAR_R_major,
            EBL=tensor_generator.EBL_K_major,
            tile_size_m=gemm_params.tile_size_m_noising_A,
            tile_size_k=gemm_params.tile_size_k_noising_A,
            pipeline_stages=gemm_params.pipeline_stages_noising_A,
            k_blocks_per_split=gemm_params.k_blocks_per_split_noising_A,
        )
        torch.cuda.synchronize()

    def run_noise_B(self, tensor_generator, gemm_params):
        noise_B(
            tensor_generator.B,
            tensor_generator.EBR,
            tensor_generator.EARxBpEB,
            tensor_generator.BpEB,
            EAR=tensor_generator.EAR_K_major,
            EBL=tensor_generator.EBL_R_major,
            tile_size_n=gemm_params.tile_size_n_noising_B,
            tile_size_k=gemm_params.tile_size_k_noising_B,
            pipeline_stages=gemm_params.pipeline_stages_noising_B,
            k_blocks_per_split=gemm_params.k_blocks_per_split_noising_B,
        )
        torch.cuda.synchronize()

    def run_noise_B_side_product(self, tensor_generator, gemm_params):
        noise_B_side_product(
            tensor_generator.B,
            tensor_generator.EBR,
            tensor_generator.EARxBpEB,
            EAR=tensor_generator.EAR_K_major,
            EBL=tensor_generator.EBL_R_major,
            tile_size_n=gemm_params.tile_size_n_noising_B,
            tile_size_k=gemm_params.tile_size_k_noising_B,
            pipeline_stages=gemm_params.pipeline_stages_noising_B,
            k_blocks_per_split=gemm_params.k_blocks_per_split_noising_B,
        )
        torch.cuda.synchronize()

    def run_noisy_gemm(self, tensor_generator, gemm_params):
        noisy_gemm(
            A=tensor_generator.A,
            B=tensor_generator.B,
            EAL=tensor_generator.EAL,
            EAL_fp16=tensor_generator.EAL_fp16,
            EAR_R_major=tensor_generator.EAR_R_major,
            EBL_R_major=tensor_generator.EBL_R_major,
            EAR_K_major=tensor_generator.EAR_K_major,
            EBL_K_major=tensor_generator.EBL_K_major,
            EBR=tensor_generator.EBR,
            EBR_fp16=tensor_generator.EBR_fp16,
            AxEBL_fp16=tensor_generator.AxEBL_fp16,
            EARxBpEB_fp16=tensor_generator.EARxBpEB_fp16,
            ApEA=tensor_generator.ApEA,
            BpEB=tensor_generator.BpEB,
            A_scales=tensor_generator.A_scales,
            B_scales=tensor_generator.B_scales,
            C=tensor_generator.C,
            host_signal_header_pinned=tensor_generator.host_signal_header_pinned,
            host_signal_sync=tensor_generator.host_signal_sync,
            AxEBL_int32=tensor_generator.AxEBL_int32,
            EARxBpEB_int32=tensor_generator.EARxBpEB_int32,
            tile_size_m=gemm_params.tile_size_m,
            tile_size_n=gemm_params.tile_size_n,
            tile_size_k=gemm_params.tile_size_k,
            pipeline_stages=gemm_params.pipeline_stages,
            cluster_size_m=gemm_params.cluster_size_m,
            cluster_size_n=gemm_params.cluster_size_n,
            swizzle=gemm_params.swizzle,
            swizzle_n_maj=gemm_params.swizzle_n_maj,
            tile_size_m_noising_A=gemm_params.tile_size_m_noising_A,
            tile_size_n_noising_B=gemm_params.tile_size_n_noising_B,
            tile_size_k_noising_A=gemm_params.tile_size_k_noising_A,
            tile_size_k_noising_B=gemm_params.tile_size_k_noising_B,
            k_blocks_per_split_noising_A=gemm_params.k_blocks_per_split_noising_A,
            k_blocks_per_split_noising_B=gemm_params.k_blocks_per_split_noising_B,
            run_noising_A=not gemm_params.skip_noising_a,
            run_noising_B=not gemm_params.skip_noising_b,
            skip_reduction=gemm_params.skip_reduction,
            skip_denoising=False,
            pow_target=tensor_generator.pow_target,
            pow_key=tensor_generator.pow_key,
        )
        torch.cuda.synchronize()

    def run_gemm(self, tensor_generator, gemm_params, noised=False):
        gemm(
            A=tensor_generator.A if not noised else tensor_generator.ApEA,
            B=tensor_generator.B if not noised else tensor_generator.BpEB,
            A_scales=tensor_generator.A_scales,
            B_scales=tensor_generator.B_scales,
            C=tensor_generator.C,
            tile_size_m=gemm_params.tile_size_m,
            tile_size_n=gemm_params.tile_size_n,
            tile_size_k=gemm_params.tile_size_k,
            pipeline_stages=gemm_params.pipeline_stages,
            cluster_size_m=gemm_params.cluster_size_m,
            cluster_size_n=gemm_params.cluster_size_n,
            swizzle=gemm_params.swizzle,
            swizzle_n_maj=gemm_params.swizzle_n_maj,
        )
        torch.cuda.synchronize()

    #### Reference Generators ####
    def compute_ref_tensor(self, tensor_generator):
        A_ref = tensor_generator.A.clone()
        B_ref = tensor_generator.B.clone()
        A_scales_ref = tensor_generator.A_scales.clone()
        B_scales_ref = tensor_generator.B_scales.clone()
        AB_ref = torch._int_mm(A_ref, B_ref.t())
        return torch.einsum(
            "mn,m,n->mn", AB_ref.to(torch.float32), A_scales_ref, B_scales_ref
        ).cpu()

    def compute_ref_noise_B(self, tensor_generator, gemm_params, return_cpu=True):
        B_ref = tensor_generator.B.clone()
        EBR_ref = tensor_generator.EBR.clone()
        EAR_ref = tensor_generator.EAR_R_major.clone()
        EBL_ref = tensor_generator.EBL_R_major.clone()

        EB_ref = torch._int_mm(EBR_ref, EBL_ref.t())  # (n, k)
        BpEB_ref_for_matmul = (B_ref + EB_ref).to(torch.int8)
        EARxBpEB_ref_int32 = torch._int_mm(BpEB_ref_for_matmul, EAR_ref)  # (n, r)
        if gemm_params.EARxBpEB_type_noising == torch.float16:
            EARxBpEB_ref_float = EARxBpEB_ref_int32.to(torch.float32) * (2**-12)
            EARxBpEB_ref = EARxBpEB_ref_float.to(torch.float16)
        else:
            EARxBpEB_ref = EARxBpEB_ref_int32
        BpEB_ref = B_ref + EB_ref
        return (
            (EARxBpEB_ref.cpu(), BpEB_ref.cpu())
            if return_cpu
            else (EARxBpEB_ref, BpEB_ref.to(torch.int8))
        )

    def compute_ref_noise_A(self, tensor_generator, gemm_params, return_cpu=True):
        A_ref = tensor_generator.A.clone()
        EAL_ref = tensor_generator.EAL.clone()
        EAR_ref = tensor_generator.EAR_R_major.clone()
        EBL_ref = tensor_generator.EBL_R_major.clone()

        EA_ref = torch._int_mm(EAL_ref, EAR_ref.t())  # (m, k)
        ApEA_ref = A_ref + EA_ref
        AxEBL_ref_int32 = torch._int_mm(A_ref, EBL_ref)  # (m, r)
        if gemm_params.AxEBL_type_noising == torch.float16:
            AxEBL_ref_float = AxEBL_ref_int32.to(torch.float32) * (2**-14)
            AxEBL_ref = AxEBL_ref_float.to(torch.float16)
        else:
            AxEBL_ref = AxEBL_ref_int32
        return (
            (AxEBL_ref.cpu(), ApEA_ref.cpu())
            if return_cpu
            else (AxEBL_ref, ApEA_ref.to(torch.int8))
        )


class TestDenoiseConverter(TestPearlGEMMBase):
    @pytest.mark.parametrize("R", [64, 128])
    @pytest.mark.parametrize("convert_denoise", ["AxEBL", "EARxBpEB", "both"])
    def test_denoise_converter_opcheck(self, R, convert_denoise):
        EARxBpEB_type_noising = (
            torch.int32
            if (convert_denoise == "EARxBpEB") or (convert_denoise == "both")
            else torch.float16
        )
        AxEBL_type_noising = (
            torch.int32
            if (convert_denoise == "AxEBL") or (convert_denoise == "both")
            else torch.float16
        )
        gemm_params = GEMMParam(
            m=1024,
            n=1024,
            k=1024,
            R=R,
            EARxBpEB_type_noising=EARxBpEB_type_noising,
            AxEBL_type_noising=AxEBL_type_noising,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        kwargs = {
            "EARxBpEB_in": tg.EARxBpEB_int32,
            "AxEBL_in": tg.AxEBL_int32,
            "EARxBpEB_out": tg.EARxBpEB_fp16,
            "AxEBL_out": tg.AxEBL_fp16,
        }
        torch.library.opcheck(
            torch.ops.pearl_gemm.denoise_converter.default, args=(), kwargs=kwargs
        )

    @pytest.mark.parametrize("m", [128, 1024, 1025, 8192])
    @pytest.mark.parametrize("n", [128, 1024, 1025, 8192])
    @pytest.mark.parametrize("k", [1024])
    @pytest.mark.parametrize("R", [64, 128])
    @pytest.mark.parametrize("convert_denoise", ["AxEBL", "EARxBpEB", "both"])
    def test_denoise_converter(self, m, n, k, R, convert_denoise):
        EARxBpEB_type_noising = (
            torch.int32
            if (convert_denoise == "EARxBpEB") or (convert_denoise == "both")
            else torch.float16
        )
        AxEBL_type_noising = (
            torch.int32
            if (convert_denoise == "AxEBL") or (convert_denoise == "both")
            else torch.float16
        )
        gemm_params = GEMMParam(
            m=m,
            n=n,
            k=k,
            R=R,
            EARxBpEB_type_noising=EARxBpEB_type_noising,
            AxEBL_type_noising=AxEBL_type_noising,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        tg._AxEBL[0], _ = self.compute_ref_noise_A(tg, gemm_params, return_cpu=False)
        tg._EARxBpEB[0], _ = self.compute_ref_noise_B(tg, gemm_params, return_cpu=False)
        self.run_denoise_converter(tg, gemm_params)
        if AxEBL_type_noising == torch.int32:
            AxEBL_ours = tg.AxEBL_fp16.cpu()
            AxEBL_ref_int32 = tg.AxEBL.cpu()
            AxEBL_ref_float = AxEBL_ref_int32.to(torch.float32) * (2**-14)
            AxEBL_ref = AxEBL_ref_float.to(torch.float16)
            assert torch.equal(AxEBL_ref, AxEBL_ours)
        if EARxBpEB_type_noising == torch.int32:
            EARxBpEB_ours = tg.EARxBpEB_fp16.cpu()
            EARxBpEB_ref_int32 = tg.EARxBpEB.cpu()
            EARxBpEB_ref_float = EARxBpEB_ref_int32.to(torch.float32) * (2**-12)
            EARxBpEB_ref = EARxBpEB_ref_float.to(torch.float16)
            assert torch.equal(EARxBpEB_ref, EARxBpEB_ours)


class TestNoiseA(TestPearlGEMMBase):
    @pytest.mark.parametrize("k_blocks_per_split_noising_A", [0, 1, None])
    @pytest.mark.parametrize("noising_a_config", noise_a_kernels)
    def test_noise_a_opcheck(
        self,
        k_blocks_per_split_noising_A,
        noising_a_config,
    ):
        gemm_params = GEMMParam(
            m=1024,
            n=1024,
            k=512,
            noising_a_config=noising_a_config,
            k_blocks_per_split_noising_A=k_blocks_per_split_noising_A,
        )
        if k_blocks_per_split_noising_A != 0 and gemm_params.AxEBL_type_noising == torch.float16:
            pytest.skip()
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        kwargs = {
            "A": tg.A,
            "EAL": tg.EAL,
            "AxEBL": tg.AxEBL,
            "ApEA": tg.ApEA,
            "EAR": tg.EAR_R_major,
            "EBL": tg.EBL_K_major,
            "tile_size_m": gemm_params.tile_size_m_noising_A,
            "tile_size_k": gemm_params.tile_size_k_noising_A,
            "pipeline_stages": gemm_params.pipeline_stages_noising_A,
            "k_blocks_per_split": k_blocks_per_split_noising_A,
        }
        torch.library.opcheck(torch.ops.pearl_gemm.noise_A.default, args=(), kwargs=kwargs)

    @pytest.mark.slow
    @pytest.mark.parametrize("k_blocks_per_split_noising_A", [0, 1, None])
    @pytest.mark.parametrize("noising_a_config", noise_a_kernels)
    def test_noise_a_consistency(
        self,
        k_blocks_per_split_noising_A,
        noising_a_config,
    ):
        gemm_params = GEMMParam(
            m=4096,
            n=4096,
            k=4096,
            noising_a_config=noising_a_config,
            k_blocks_per_split_noising_A=k_blocks_per_split_noising_A,
        )
        if k_blocks_per_split_noising_A != 0 and gemm_params.AxEBL_type_noising == torch.float16:
            pytest.skip()
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        self.run_noise_A(tg, gemm_params)

        ApEA_ref = tg.ApEA.clone()
        AxEBL_ref = tg.AxEBL.clone()
        for _ in range(10000):
            tg.AxEBL.zero_()
            self.run_noise_A(tg, gemm_params)
            assert torch.equal(tg.ApEA, ApEA_ref)
            assert torch.equal(tg.AxEBL, AxEBL_ref)

    @pytest.mark.parametrize("m", [128, 1024, 1025, 8192])
    @pytest.mark.parametrize("n", [128])
    @pytest.mark.parametrize("k", [256, 512, 8192])
    @pytest.mark.parametrize("noising_a_config", noise_a_kernels)
    def test_int7_test_noise_a(self, m, n, k, noising_a_config):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            noising_a_config=noising_a_config,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        self.run_noise_A(tg, gemm_params)

        AxEBL_ref, ApEA_ref = self.compute_ref_noise_A(tg, gemm_params)
        AxEBL_ours = tg.AxEBL.cpu()
        ApEA_ours = tg.ApEA.cpu().to(torch.int32)

        assert torch.equal(AxEBL_ref, AxEBL_ours)
        assert torch.equal(ApEA_ref, ApEA_ours)

    @pytest.mark.parametrize("m", [128, 1024])
    @pytest.mark.parametrize("n", [128])
    @pytest.mark.parametrize("k", [256, 512])
    @pytest.mark.parametrize("noising_a_config", noise_a_kernels)
    def test_int7_test_torch_noise_a(
        self,
        m,
        n,
        k,
        noising_a_config,
    ):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            noising_a_config=noising_a_config,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        noise_a_args = {
            "A": tg.A,
            "EAL": tg.EAL,
            "AxEBL": tg.AxEBL,
            "ApEA": tg.ApEA,
            "EAR": tg.EAR_R_major,
            "EBL": tg.EBL_K_major,
            "tile_size_m": gemm_params.tile_size_m_noising_A,
            "tile_size_k": gemm_params.tile_size_k_noising_A,
            "pipeline_stages": gemm_params.pipeline_stages_noising_A,
        }

        @torch.compile(fullgraph=True)
        def op_noise_a_simplified(noise_a_args):
            torch.ops.pearl_gemm.noise_A(**noise_a_args)

        op_noise_a_simplified(noise_a_args)
        torch.cuda.synchronize()

        AxEBL_ref, ApEA_ref = self.compute_ref_noise_A(tg, gemm_params)

        assert torch.equal(AxEBL_ref, tg.AxEBL.cpu())
        assert torch.equal(ApEA_ref, tg.ApEA.to(torch.int32).cpu())

    @pytest.mark.parametrize("m", [128, 1024, 1025, 8192])
    @pytest.mark.parametrize("n", [128])
    @pytest.mark.parametrize("k", [256, 512, 8192])
    @pytest.mark.parametrize("k_blocks_per_split_noising_A", [1, 8, None])
    @pytest.mark.parametrize("noising_a_config", noise_a_kernels)
    def test_int7_test_split_k_noise_a(
        self, m, n, k, k_blocks_per_split_noising_A, noising_a_config
    ):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            k_blocks_per_split_noising_A=k_blocks_per_split_noising_A,
            noising_a_config=noising_a_config,
        )
        if k_blocks_per_split_noising_A != 0 and gemm_params.AxEBL_type_noising == torch.float16:
            pytest.skip()
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        self.run_noise_A(tg, gemm_params)

        AxEBL_ref, ApEA_ref = self.compute_ref_noise_A(tg, gemm_params)
        AxEBL_ours = tg.AxEBL.cpu()
        ApEA_ours = tg.ApEA.cpu().to(torch.int32)

        assert torch.equal(AxEBL_ref, AxEBL_ours)
        assert torch.equal(ApEA_ref, ApEA_ours)


class TestNoiseB(TestPearlGEMMBase):
    @pytest.mark.parametrize("k_blocks_per_split_noising_B", [0, 1, None])
    @pytest.mark.parametrize("noising_b_config", noise_b_kernels)
    def test_noise_b_opcheck(self, k_blocks_per_split_noising_B, noising_b_config):
        gemm_params = GEMMParam(
            m=1024,
            n=1024,
            k=512,
            noising_b_config=noising_b_config,
            k_blocks_per_split_noising_B=k_blocks_per_split_noising_B,
        )
        if (
            k_blocks_per_split_noising_B != 0
            and gemm_params.EARxBpEB_type_noising == torch.float16
        ):
            pytest.skip()
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        kwargs = {
            "B": tg.B,
            "EBR": tg.EBR,
            "EARxBpEB": tg.EARxBpEB,
            "BpEB": tg.BpEB,
            "EAR": tg.EAR_K_major,
            "EBL": tg.EBL_R_major,
            "tile_size_n": gemm_params.tile_size_n_noising_B,
            "tile_size_k": gemm_params.tile_size_k_noising_B,
            "pipeline_stages": gemm_params.pipeline_stages_noising_B,
            "k_blocks_per_split": gemm_params.k_blocks_per_split_noising_B,
        }
        torch.library.opcheck(torch.ops.pearl_gemm.noise_B.default, args=(), kwargs=kwargs)

    @pytest.mark.slow
    @pytest.mark.parametrize("k_blocks_per_split_noising_B", [0, 1, None])
    @pytest.mark.parametrize("noising_b_config", noise_b_kernels)
    def test_noise_b_consistency(self, k_blocks_per_split_noising_B, noising_b_config):
        gemm_params = GEMMParam(
            m=4096,
            n=4096,
            k=4096,
            noising_b_config=noising_b_config,
            k_blocks_per_split_noising_B=k_blocks_per_split_noising_B,
        )
        if (
            k_blocks_per_split_noising_B != 0
            and gemm_params.EARxBpEB_type_noising == torch.float16
        ):
            pytest.skip()
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        self.run_noise_B(tg, gemm_params)
        BpEB_ref = tg.BpEB.clone()
        EARxBpEB_ref = tg.EARxBpEB.clone()
        for _ in range(10000):
            tg.EARxBpEB.zero_()
            self.run_noise_B(tg, gemm_params)
            assert torch.equal(tg.BpEB, BpEB_ref)
            assert torch.equal(tg.EARxBpEB, EARxBpEB_ref)

    @pytest.mark.parametrize("m", [128])
    @pytest.mark.parametrize("n", [128, 1024, 1032, 8192])
    @pytest.mark.parametrize("k", [256, 512, 8192])
    @pytest.mark.parametrize("noising_b_config", noise_b_kernels)
    def test_int7_test_noise_b(self, m, n, k, noising_b_config):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            noising_b_config=noising_b_config,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        self.run_noise_B(tg, gemm_params)

        EARxBpEB_ref, BpEB_ref = self.compute_ref_noise_B(tg, gemm_params)

        assert torch.equal(EARxBpEB_ref, tg.EARxBpEB.cpu())
        assert torch.equal(BpEB_ref, tg.BpEB.to(torch.int32).cpu())

    @pytest.mark.parametrize("m", [128])
    @pytest.mark.parametrize("n", [128, 1024, 1032])
    @pytest.mark.parametrize("k", [256, 512, 8192])
    @pytest.mark.parametrize("k_blocks_per_split_noising_B", [0, 1, 8, None])
    @pytest.mark.parametrize("noising_b_config", noise_b_kernels)
    def test_int7_test_split_k_noise_b(
        self, m, n, k, k_blocks_per_split_noising_B, noising_b_config
    ):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            noising_b_config=noising_b_config,
            k_blocks_per_split_noising_B=k_blocks_per_split_noising_B,
        )
        if k_blocks_per_split_noising_B != 0 and gemm_params.EARxBpEB_type_noising == torch.float16:
            pytest.skip()
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        self.run_noise_B(tg, gemm_params)

        EARxBpEB_ref, BpEB_ref = self.compute_ref_noise_B(tg, gemm_params)

        assert torch.equal(EARxBpEB_ref, tg.EARxBpEB.cpu())
        assert torch.equal(BpEB_ref, tg.BpEB.to(torch.int32).cpu())

    @pytest.mark.parametrize("m", [128])
    @pytest.mark.parametrize(
        ("n", "k"),
        [
            (128, 256),
            (1024, 512),
            (1024, 8192),
            (1032, 80),
            (130, 144),
        ],
    )
    @pytest.mark.parametrize("k_blocks_per_split", [0, 1, None])
    @pytest.mark.parametrize(
        "noising_b_config",
        [cfg for cfg in noise_b_kernels if cfg.EARxBpEB_type == "int32"],
    )
    def test_bpeb_ear_product_matches_noise_b(
        self, m, n, k, k_blocks_per_split, noising_b_config
    ):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            noising_b_config=noising_b_config,
            k_blocks_per_split_noising_B=k_blocks_per_split,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        self.run_noise_B(tg, gemm_params)
        EARxBpEB_ref = tg.EARxBpEB.clone()
        EARxBpEB_from_cached = torch.empty_like(tg.EARxBpEB)

        bpeb_ear_product(
            tg.BpEB,
            tg.EAR_K_major,
            EARxBpEB_from_cached,
            tile_size_n=gemm_params.tile_size_n_noising_B,
            tile_size_k=gemm_params.tile_size_k_noising_B,
            pipeline_stages=gemm_params.pipeline_stages_noising_B,
            k_blocks_per_split=k_blocks_per_split,
        )
        torch.cuda.synchronize()

        assert torch.equal(EARxBpEB_ref, EARxBpEB_from_cached)

    @pytest.mark.parametrize("m", [128])
    @pytest.mark.parametrize("n", [128, 1024, 1032])
    @pytest.mark.parametrize("k", [256, 512, 8192])
    @pytest.mark.parametrize("k_blocks_per_split_noising_B", [0, 1, None])
    @pytest.mark.parametrize("noising_b_config", noise_b_kernels)
    def test_noise_b_side_product_matches_noise_b_without_storing_bpeb(
        self, m, n, k, k_blocks_per_split_noising_B, noising_b_config
    ):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            noising_b_config=noising_b_config,
            k_blocks_per_split_noising_B=k_blocks_per_split_noising_B,
        )
        if k_blocks_per_split_noising_B != 0 and gemm_params.EARxBpEB_type_noising == torch.float16:
            pytest.skip()
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        self.run_noise_B(tg, gemm_params)
        EARxBpEB_ref = tg.EARxBpEB.clone()
        BpEB_ref = tg.BpEB.clone()
        tg.EARxBpEB.zero_()
        tg.BpEB.zero_()

        self.run_noise_B_side_product(tg, gemm_params)

        assert torch.equal(EARxBpEB_ref, tg.EARxBpEB)
        assert not torch.equal(BpEB_ref, tg.BpEB)
        assert torch.count_nonzero(tg.BpEB) == 0

    @pytest.mark.parametrize("m", [512])
    @pytest.mark.parametrize("n", [512, 1024])
    @pytest.mark.parametrize("k", [256, 512])
    @pytest.mark.parametrize("noising_b_config", noise_b_kernels)
    def test_int7_test_torch_noise_b(self, m, n, k, noising_b_config):
        # GEMM parameters
        gemm_params = GEMMParam(
            m,
            n,
            k,
            noising_b_config=noising_b_config,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        noise_b_args = {
            "B": tg.B,
            "EBR": tg.EBR,
            "EARxBpEB": tg.EARxBpEB,
            "BpEB": tg.BpEB,
            "EAR": tg.EAR_K_major,
            "EBL": tg.EBL_R_major,
            "tile_size_n": gemm_params.tile_size_n_noising_B,
            "tile_size_k": gemm_params.tile_size_k_noising_B,
            "pipeline_stages": gemm_params.pipeline_stages_noising_B,
            "k_blocks_per_split": gemm_params.k_blocks_per_split_noising_B,
        }

        @torch.compile(fullgraph=True)
        def op_noise_b_simplified(noise_b_args):
            torch.ops.pearl_gemm.noise_B(**noise_b_args)

        op_noise_b_simplified(noise_b_args)
        torch.cuda.synchronize()

        EARxBpEB_ref, BpEB_ref = self.compute_ref_noise_B(tg, gemm_params)

        assert torch.equal(EARxBpEB_ref, tg.EARxBpEB.cpu())
        assert torch.equal(BpEB_ref, tg.BpEB.to(torch.int32).cpu())


class TestNoisyGEMM(TestPearlGEMMBase):
    @pytest.mark.skip
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    def test_noisy_gemm_opcheck(self, matmul_config):
        gemm_params = GEMMParam(m=1024, n=1024, k=512, matmul_config=matmul_config)
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        gemm_kwargs = {
            "A": tg.A,
            "B": tg.B,
            "EAL": tg.EAL,
            "EBR": tg.EBR,
            "EAR_R_major": tg.EAR_R_major,
            "EBL_R_major": tg.EBL_R_major,
            "EAR_K_major": tg.EAR_K_major,
            "EBL_K_major": tg.EBL_K_major,
            "AxEBL_fp16": tg.AxEBL_fp16,
            "EARxBpEB_fp16": tg.EARxBpEB_fp16,
            "ApEA": tg.ApEA,
            "BpEB": tg.BpEB,
            "A_scales": tg.A_scales,
            "B_scales": tg.B_scales,
            "C": tg.C,
            "host_signal_header_pinned": tg.host_signal_header_pinned,
            "host_signal_sync": tg.host_signal_sync,
            "AxEBL_int32": tg.AxEBL_int32,
            "EARxBpEB_int32": tg.EARxBpEB_int32,
            "tile_size_m": gemm_params.tile_size_m,
            "tile_size_n": gemm_params.tile_size_n,
            "tile_size_k": gemm_params.tile_size_k,
            "pipeline_stages": gemm_params.pipeline_stages,
            "swizzle": gemm_params.swizzle,
            "swizzle_n_maj": gemm_params.swizzle_n_maj,
            "cluster_size_m": gemm_params.cluster_size_m,
            "cluster_size_n": gemm_params.cluster_size_n,
            "tile_size_m_noising_A": gemm_params.tile_size_m_noising_A,
            "tile_size_n_noising_B": gemm_params.tile_size_n_noising_B,
            "tile_size_k_noising_A": gemm_params.tile_size_k_noising_A,
            "tile_size_k_noising_B": gemm_params.tile_size_k_noising_B,
            "pipeline_stages_noising_A": gemm_params.pipeline_stages_noising_A,
            "pipeline_stages_noising_B": gemm_params.pipeline_stages_noising_B,
            "k_blocks_per_split_noising_A": gemm_params.k_blocks_per_split_noising_A,
            "k_blocks_per_split_noising_B": gemm_params.k_blocks_per_split_noising_B,
            "run_noising_A": not gemm_params.skip_noising_a,
            "run_noising_B": not gemm_params.skip_noising_b,
            "skip_reduction": gemm_params.skip_reduction,
            "skip_denoising": gemm_params.skip_denoising,
            "pow_target": tg.pow_target,
            "pow_key": tg.pow_key,
        }
        torch.library.opcheck(torch.ops.pearl_gemm.noisy_gemm.default, args=(), kwargs=gemm_kwargs)

    @pytest.mark.slow
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    @pytest.mark.parametrize("noising_dtype", ["fp16", "int32"])
    def test_noisy_gemm_consistency(self, matmul_config, noising_dtype):
        gemm_params = GEMMParam(
            m=4096,
            n=4096,
            k=4096,
            matmul_config=matmul_config,
            EARxBpEB_type_noising=noising_dtype,
            AxEBL_type_noising=noising_dtype,
            skip_noising_b=False,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        self.run_noisy_gemm(tg, gemm_params)

        ApEA_ref = tg.ApEA.clone()
        AxEBL_ref = tg.AxEBL.clone()
        BpEB_ref = tg.BpEB.clone()
        EARxBpEB_ref = tg.EARxBpEB.clone()
        C_ref = tg.C.clone()
        for _ in range(10000):
            tg.EARxBpEB.zero_()
            tg.AxEBL.zero_()
            self.run_noisy_gemm(tg, gemm_params)
            assert torch.equal(tg.ApEA, ApEA_ref)
            assert torch.equal(tg.AxEBL, AxEBL_ref)
            assert torch.equal(tg.BpEB, BpEB_ref)
            assert torch.equal(tg.EARxBpEB, EARxBpEB_ref)
            assert torch.equal(tg.C, C_ref)

    @pytest.mark.parametrize("m", [128, 1024, 1025, 8192])
    @pytest.mark.parametrize("n", [128, 1024, 1032, 6144])
    @pytest.mark.parametrize("k", [256, 512, 4096])
    @pytest.mark.parametrize("AxEBL_type_noising", ["fp16", "int32"])
    @pytest.mark.parametrize("EARxBpEB_type_noising", ["fp16", "int32"])
    @pytest.mark.parametrize("variable_scales", [True, False])
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    def test_int7_noisy_gemm(
        self, m, n, k, variable_scales, AxEBL_type_noising, EARxBpEB_type_noising, matmul_config
    ):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            skip_noising_a=False,
            skip_noising_b=False,
            EARxBpEB_type_noising=EARxBpEB_type_noising,
            AxEBL_type_noising=AxEBL_type_noising,
            matmul_config=matmul_config,
            use_variable_scales=variable_scales,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        self.run_noisy_gemm(tg, gemm_params)

        C_ref = self.compute_ref_tensor(tg)

        atol = 1e-1  # absolute tolerance
        rtol = 1e-2  # relatve  tolerance

        # Not sure if casting up or down is better. Casting down for now
        torch.testing.assert_close(tg.C.cpu(), C_ref.to(torch.bfloat16), atol=atol, rtol=rtol)

    @pytest.mark.parametrize("m", [128, 1024, 1025, 8192])
    @pytest.mark.parametrize("n", [128, 1024, 1032, 8192])
    @pytest.mark.parametrize("k", [256, 512, 8192])
    @pytest.mark.parametrize("variable_scales", [True, False])
    @pytest.mark.parametrize("noising_dtype", ["fp16", "int32"])
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    def test_int7_noisy_gemm_no_noising_b(
        self, m, n, k, variable_scales, noising_dtype, matmul_config
    ):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            EARxBpEB_type_noising=noising_dtype,
            AxEBL_type_noising=noising_dtype,
            matmul_config=matmul_config,
            skip_noising_b=True,
            use_variable_scales=variable_scales,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        # Run the kernel
        # we first do reference noisingb and then do gemm without noising b
        tg._EARxBpEB[0], tg._BpEB[0] = self.compute_ref_noise_B(tg, gemm_params, return_cpu=False)
        self.run_noisy_gemm(tg, gemm_params)

        C_ref = self.compute_ref_tensor(tg)

        atol = 1e-1  # absolute tolerance
        rtol = 1e-2  # relatve  tolerance

        # Not sure if casting up or down is better. Casting down for now
        torch.testing.assert_close(tg.C.cpu(), C_ref.to(torch.bfloat16), atol=atol, rtol=rtol)

    @pytest.mark.parametrize("m", [512, 1024])
    @pytest.mark.parametrize("n", [512, 1024])
    @pytest.mark.parametrize("k", [256, 512])
    @pytest.mark.parametrize("noising_dtype", ["fp16", "int32"])
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    def test_int7_noisy_torch_gemm(self, m, n, k, noising_dtype, matmul_config):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            EARxBpEB_type_noising=noising_dtype,
            AxEBL_type_noising=noising_dtype,
            matmul_config=matmul_config,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        # torch.compile doesn't like GemmTensorGenerator's __getattribute__ shenanigans
        gemm_kwargs = {
            "A": tg.A,
            "B": tg.B,
            "EAL": tg.EAL,
            "EAL_fp16": tg.EAL_fp16,
            "EBR": tg.EBR,
            "EBR_fp16": tg.EBR_fp16,
            "EAR_R_major": tg.EAR_R_major,
            "EBL_R_major": tg.EBL_R_major,
            "EAR_K_major": tg.EAR_K_major,
            "EBL_K_major": tg.EBL_K_major,
            "AxEBL_fp16": tg.AxEBL_fp16,
            "EARxBpEB_fp16": tg.EARxBpEB_fp16,
            "ApEA": tg.ApEA,
            "BpEB": tg.BpEB,
            "A_scales": tg.A_scales,
            "B_scales": tg.B_scales,
            "C": tg.C,
            "host_signal_header_pinned": tg.host_signal_header_pinned,
            "host_signal_sync": tg.host_signal_sync,
            "AxEBL_int32": tg.AxEBL_int32,
            "EARxBpEB_int32": tg.EARxBpEB_int32,
            "pow_target": tg.pow_target,
            "pow_key": tg.pow_key,
            "pipeline_stages": gemm_params.pipeline_stages,
            "tile_size_m": gemm_params.tile_size_m,
            "tile_size_n": gemm_params.tile_size_n,
            "tile_size_k": gemm_params.tile_size_k,
            "swizzle": gemm_params.swizzle,
            "swizzle_n_maj": gemm_params.swizzle_n_maj,
            "cluster_size_m": gemm_params.cluster_size_m,
            "cluster_size_n": gemm_params.cluster_size_n,
            "tile_size_m_noising_A": gemm_params.tile_size_m_noising_A,
            "tile_size_n_noising_B": gemm_params.tile_size_n_noising_B,
            "tile_size_k_noising_A": gemm_params.tile_size_k_noising_A,
            "tile_size_k_noising_B": gemm_params.tile_size_k_noising_B,
            "pipeline_stages_noising_A": gemm_params.pipeline_stages_noising_A,
            "pipeline_stages_noising_B": gemm_params.pipeline_stages_noising_B,
            "k_blocks_per_split_noising_A": gemm_params.k_blocks_per_split_noising_A,
            "k_blocks_per_split_noising_B": gemm_params.k_blocks_per_split_noising_B,
            "run_noising_A": True,
            "run_noising_B": True,
        }

        @torch.compile(fullgraph=True)
        def op_noisy_gemm_simplified(gemm_kwargs):
            torch.ops.pearl_gemm.noisy_gemm(**gemm_kwargs)

        op_noisy_gemm_simplified(gemm_kwargs)
        torch.cuda.synchronize()

        C_ref = self.compute_ref_tensor(tg)

        atol = 1e-1  # absolute tolerance
        rtol = 1e-2  # relatve  tolerance

        # Not sure if casting up or down is better. Casting down for now
        torch.testing.assert_close(tg.C.cpu(), C_ref.to(torch.bfloat16), atol=atol, rtol=rtol)

    @pytest.mark.parametrize("m", [128, 1024])
    @pytest.mark.parametrize("n", [128, 1024])
    @pytest.mark.parametrize("k", [256, 512])
    @pytest.mark.parametrize("noising_dtype", ["fp16", "int32"])
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    def test_int7_noisy_torch_gemm_separate_functions(self, m, n, k, noising_dtype, matmul_config):
        gemm_params = GEMMParam(
            m,
            n,
            k,
            EARxBpEB_type_noising=noising_dtype,
            AxEBL_type_noising=noising_dtype,
            matmul_config=matmul_config,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        noise_a_kwargs = {
            "A": tg.A,
            "EAL": tg.EAL,
            "AxEBL": tg.AxEBL,
            "ApEA": tg.ApEA,
            "EAR": tg.EAR_R_major,
            "EBL": tg.EBL_K_major,
        }
        noise_b_kwargs = {
            "B": tg.B,
            "EBR": tg.EBR,
            "EARxBpEB": tg.EARxBpEB,
            "BpEB": tg.BpEB,
            "EAR": tg.EAR_K_major,
            "EBL": tg.EBL_R_major,
        }
        gemm_kwargs = {
            "A": tg.A,
            "B": tg.B,
            "EAL": tg.EAL,
            "EAL_fp16": tg.EAL_fp16,
            "EBR": tg.EBR,
            "EBR_fp16": tg.EBR_fp16,
            "EAR_R_major": tg.EAR_R_major,
            "EBL_R_major": tg.EBL_R_major,
            "EAR_K_major": tg.EAR_K_major,
            "EBL_K_major": tg.EBL_K_major,
            "AxEBL_fp16": tg.AxEBL_fp16,
            "EARxBpEB_fp16": tg.EARxBpEB_fp16,
            "ApEA": tg.ApEA,
            "BpEB": tg.BpEB,
            "A_scales": tg.A_scales,
            "B_scales": tg.B_scales,
            "C": tg.C,
            "host_signal_header_pinned": tg.host_signal_header_pinned,
            "host_signal_sync": tg.host_signal_sync,
            "AxEBL_int32": tg.AxEBL_int32,
            "EARxBpEB_int32": tg.EARxBpEB_int32,
            "pow_target": tg.pow_target,
            "pow_key": tg.pow_key,
            "tile_size_m": gemm_params.tile_size_m,
            "tile_size_n": gemm_params.tile_size_n,
            "tile_size_k": gemm_params.tile_size_k,
            "pipeline_stages": gemm_params.pipeline_stages,
            "cluster_size_m": gemm_params.cluster_size_m,
            "cluster_size_n": gemm_params.cluster_size_n,
            "swizzle": gemm_params.swizzle,
            "swizzle_n_maj": gemm_params.swizzle_n_maj,
            "run_noising_A": False,
            "run_noising_B": False,
            "skip_denoising": False,
            "skip_reduction": False,
        }

        @torch.compile(fullgraph=True, dynamic=False)
        def op_gemm_simplified(noise_a_kwargs, noise_b_kwargs, gemm_kwargs):
            torch.ops.pearl_gemm.noise_A(**noise_a_kwargs)
            torch.ops.pearl_gemm.noise_B(**noise_b_kwargs)
            torch.ops.pearl_gemm.noisy_gemm(**gemm_kwargs)

        op_gemm_simplified(noise_a_kwargs, noise_b_kwargs, gemm_kwargs)
        torch.cuda.synchronize()

        C_ref = self.compute_ref_tensor(tg)

        atol = 1e-1  # absolute tolerance
        rtol = 1e-2  # relatve  tolerance

        # Not sure if casting up or down is better. Casting down for now
        torch.testing.assert_close(tg.C.cpu(), C_ref.to(torch.bfloat16), atol=atol, rtol=rtol)


class TestGEMM(TestPearlGEMMBase):
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    def test_gemm_opcheck(self, matmul_config):
        gemm_params = GEMMParam(
            m=1024,
            n=1024,
            k=512,
            matmul_config=matmul_config,
            skip_reduction=True,
            skip_denoising=True,
            skip_noising_a=True,
            skip_noising_b=True,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        gemm_kwargs = {
            "A": tg.A,
            "B": tg.B,
            "A_scales": tg.A_scales,
            "B_scales": tg.B_scales,
            "C": tg.C,
            "tile_size_m": gemm_params.tile_size_m,
            "tile_size_n": gemm_params.tile_size_n,
            "tile_size_k": gemm_params.tile_size_k,
            "pipeline_stages": gemm_params.pipeline_stages,
            "swizzle": gemm_params.swizzle,
            "swizzle_n_maj": gemm_params.swizzle_n_maj,
            "cluster_size_m": gemm_params.cluster_size_m,
            "cluster_size_n": gemm_params.cluster_size_n,
        }
        torch.library.opcheck(torch.ops.pearl_gemm.gemm.default, args=(), kwargs=gemm_kwargs)

    @pytest.mark.slow
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    def test_gemm_consistency(self, matmul_config):
        gemm_params = GEMMParam(
            m=4096,
            n=4096,
            k=4096,
            matmul_config=matmul_config,
            skip_noising_a=True,
            skip_noising_b=True,
            skip_denoising=True,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        self.run_gemm(tg, gemm_params)

        C_ref = tg.C.clone()
        for _ in range(10000):
            self.run_gemm(tg, gemm_params)
            assert torch.equal(tg.C, C_ref)

    @pytest.mark.parametrize("m", [128, 1024, 1025, 8192])
    @pytest.mark.parametrize("n", [128, 1024, 1032, 8192])
    @pytest.mark.parametrize("k", [256, 512, 8192])
    @pytest.mark.parametrize("variable_scales", [True, False])
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    def test_noiseless_int7_gemm(self, m, n, k, variable_scales, matmul_config):
        # GEMM parameters
        gemm_params = GEMMParam(
            m,
            n,
            k,
            matmul_config=matmul_config,
            use_variable_scales=variable_scales,
        )
        # Input parameters
        tg = GemmTensorGenerator(gemm_params)
        # Create the tensors
        tg.generate()

        # Run the kernel
        self.run_gemm(tg, gemm_params)

        C_ref = self.compute_ref_tensor(tg)

        atol = 1e-1  # absolute tolerance
        rtol = 1e-2  # relatve  tolerance

        # Not sure if casting up or down is better. Casting down for now
        torch.testing.assert_close(tg.C.cpu(), C_ref.to(torch.bfloat16), atol=atol, rtol=rtol)

    @pytest.mark.parametrize("m", [128, 1024])
    @pytest.mark.parametrize("n", [128, 1024])
    @pytest.mark.parametrize("k", [256, 512])
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    def test_noiseless_int7_torch_gemm(self, m, n, k, matmul_config):
        gemm_params = GEMMParam(m, n, k, matmul_config=matmul_config)
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()
        gemm_kwargs = {
            "A": tg.A,
            "B": tg.B,
            "A_scales": tg.A_scales,
            "B_scales": tg.B_scales,
            "C": tg.C,
            "tile_size_m": gemm_params.tile_size_m,
            "tile_size_n": gemm_params.tile_size_n,
            "tile_size_k": gemm_params.tile_size_k,
            "pipeline_stages": gemm_params.pipeline_stages,
            "cluster_size_m": gemm_params.cluster_size_m,
            "cluster_size_n": gemm_params.cluster_size_n,
            "swizzle": gemm_params.swizzle,
            "swizzle_n_maj": gemm_params.swizzle_n_maj,
        }

        @torch.compile(fullgraph=True)
        def op_gemm_simplified(gemm_kwargs):
            torch.ops.pearl_gemm.gemm(**gemm_kwargs)

        op_gemm_simplified(gemm_kwargs)
        torch.cuda.synchronize()

        C_ref = self.compute_ref_tensor(tg)

        atol = 1e-1  # absolute tolerance
        rtol = 1e-2  # relatve  tolerance

        # Not sure if casting up or down is better. Casting down for now
        torch.testing.assert_close(tg.C.cpu(), C_ref.to(torch.bfloat16), atol=atol, rtol=rtol)


class TestSwizzleNoisyGEMM(TestPearlGEMMBase):
    @pytest.mark.parametrize("m", [512, 513])
    @pytest.mark.parametrize("n", [8192, 8200, 28672])
    @pytest.mark.parametrize("k", [256, 8192])
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    # None is used for heuristic test
    @pytest.mark.parametrize("swizzle", [1, 2, 4, 8, 16, 32, 64, 128, None])
    @pytest.mark.parametrize("swizzle_n_maj", [True, False])
    def test_int7_swizzle_noisy_gemm(self, m, n, k, matmul_config, swizzle, swizzle_n_maj):
        # GEMM parameters
        gemm_params = GEMMParam(
            m, n, k, matmul_config=matmul_config, swizzle=swizzle, swizzle_n_maj=swizzle_n_maj
        )
        # Input parameters
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        self.run_noisy_gemm(tg, gemm_params)

        C_ref = self.compute_ref_tensor(tg)

        atol = 1e-1  # absolute tolerance
        rtol = 1e-2  # relatve  tolerance

        # Not sure if casting up or down is better. Casting down for now
        torch.testing.assert_close(tg.C.cpu(), C_ref.to(torch.bfloat16), atol=atol, rtol=rtol)


class TestSwizzleGEMM(TestPearlGEMMBase):
    @pytest.mark.parametrize("m", [512, 513])
    @pytest.mark.parametrize("n", [8192, 8200, 28672])
    @pytest.mark.parametrize("k", [256, 8192])
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    # None is used for heuristic test
    @pytest.mark.parametrize("swizzle", [1, 2, 4, 8, 16, 32, 64, 128, None])
    @pytest.mark.parametrize("swizzle_n_maj", [True, False])
    def test_int7_swizzle_gemm(self, m, n, k, matmul_config, swizzle, swizzle_n_maj):
        gemm_params = GEMMParam(
            m, n, k, matmul_config=matmul_config, swizzle=swizzle, swizzle_n_maj=swizzle_n_maj
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        self.run_gemm(tg, gemm_params)

        C_ref = self.compute_ref_tensor(tg)

        atol = 1e-1  # absolute tolerance
        rtol = 1e-2  # relatve  tolerance

        # Not sure if casting up or down is better. Casting down for now
        torch.testing.assert_close(tg.C.cpu(), C_ref.to(torch.bfloat16), atol=atol, rtol=rtol)


class TestSkipReductionNoisyGEMM(TestPearlGEMMBase):
    @pytest.mark.parametrize("m", [128, 1024, 1025, 8192])
    @pytest.mark.parametrize("n", [128, 1024, 1032, 8192])
    @pytest.mark.parametrize("k", [256, 512, 768, 8192])
    @pytest.mark.parametrize("matmul_config", matmul_kernels)
    def test_int7_skip_reduction_noisy_gemm(self, m, n, k, matmul_config):
        print(matmul_config)
        gemm_params = GEMMParam(
            m,
            n,
            k,
            matmul_config=matmul_config,
            skip_reduction=True,
        )
        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        # Run the kernel
        self.run_noisy_gemm(tg, gemm_params)

        C_ref = self.compute_ref_tensor(tg)

        atol = 1e-1  # absolute tolerance
        rtol = 1e-2  # relatve  tolerance

        # Not sure if casting up or down is better. Casting down for now
        torch.testing.assert_close(tg.C.cpu(), C_ref.to(torch.bfloat16), atol=atol, rtol=rtol)


def calculate_expected_inner_hash_count(m: int, n: int, k: int, matmul_config: MatmulConfig) -> int:
    """
    Calculate total inner_hash calls during GEMM for validation.

    Formula: ceil(M/tile_m) x ceil(N/tile_n) x (K // R) x num_threads_per_cta
      - ceil(M/tile_m) x ceil(N/tile_n) = number of CTAs (output tiles)
      - K // R = number of reductions per CTA
      - num_threads_per_cta = threads calling inner_hash (one hash per thread per reduction)

    Each thread holds (tile_m * tile_n / num_threads) registers.
    Each thread calls inner_hash once per reduction step.
    """
    num_tiles = math.ceil(m / matmul_config.matmul_tile_h) * math.ceil(
        n / matmul_config.matmul_tile_w
    )
    reductions_per_tile = k // matmul_config.noise_rank
    num_threads_per_cta = (matmul_config.matmul_tile_h // matmul_config.hash_tile_h) * (
        matmul_config.matmul_tile_w // matmul_config.hash_tile_w
    )
    return num_tiles * reductions_per_tile * num_threads_per_cta


class TestInnerHashCounting(TestPearlGEMMBase):
    """Test that inner_hash is called the expected number of times during GEMM."""

    SHAPES_MNK = [
        (512, 512, 256),
        (4096, 4096, 4096),
        # Edge cases: small m values
        (1, 512, 256),
        (128, 128, 128),
        # Edge cases: non-divisible M/N by tile sizes
        (257, 512, 256),
        (128, 256, 512),
        (150, 504, 256),
    ]

    @pytest.mark.parametrize("m, n, k", SHAPES_MNK)
    @pytest.mark.parametrize("matmul_kernel_config", matmul_kernels)
    @pytest.mark.skipif(DISABLE_DEBUG_MODE, reason="Debug mode is disabled")
    def test_inner_hash_count(self, m, n, k, matmul_kernel_config):
        """Verify inner_hash call count matches expected formula."""

        gemm_params = GEMMParam(
            m,
            n,
            k,
            matmul_config=matmul_kernel_config,
            skip_reduction=False,
        )

        matmul_config = GPUMatmulConfigFactory.create(k=k, noise_rank=gemm_params.R)

        tg = GemmTensorGenerator(gemm_params)
        tg.generate()

        counter = torch.zeros(1, dtype=torch.int64, device="cuda")
        torch.ops.pearl_gemm.noisy_gemm(
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
            tile_size_m=gemm_params.tile_size_m,
            tile_size_n=gemm_params.tile_size_n,
            tile_size_k=gemm_params.tile_size_k,
            pipeline_stages=gemm_params.pipeline_stages,
            swizzle=gemm_params.swizzle,
            tile_size_m_noising_A=gemm_params.tile_size_m_noising_A,
            tile_size_n_noising_B=gemm_params.tile_size_n_noising_B,
            tile_size_k_noising_A=gemm_params.tile_size_k_noising_A,
            tile_size_k_noising_B=gemm_params.tile_size_k_noising_B,
            pipeline_stages_noising_A=gemm_params.pipeline_stages_noising_A,
            pipeline_stages_noising_B=gemm_params.pipeline_stages_noising_B,
            k_blocks_per_split_noising_A=gemm_params.k_blocks_per_split_noising_A,
            k_blocks_per_split_noising_B=gemm_params.k_blocks_per_split_noising_B,
            run_noising_A=False,
            run_noising_B=False,
            skip_reduction=False,
            skip_denoising=False,
            pow_target=tg.pow_target,
            pow_key=tg.pow_key,
            inner_hash_counter=counter,
            enable_debug=True,
        )
        torch.cuda.synchronize()

        expected = calculate_expected_inner_hash_count(
            m,
            n,
            k,
            matmul_config,
        )
        assert counter.item() == expected, f"Count mismatch: {counter.item()} != {expected}"
