#pragma once

#include "cute/tensor.hpp"

#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cutlass/epilogue/collective/builders/sm90_common.inl"
#include "cutlass/gemm/collective/builders/sm90_common.inl"

#include "named_barrier.hpp"
#include "pearl_gemm_constants.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

template <class TileShape_NRK_, int kNumThreads, class Element,
          class ElementDenoise, int kStages, bool UseReduction>
class BpEBEARProductKernel {
 public:
  using ElementAccum = int32_t;
  static_assert(cute::is_same_v<Element, int8_t>);
  static_assert(cute::is_same_v<ElementDenoise, int>,
                "BpEB x EAR product currently writes int32 only");

  using TileShape_NRK = TileShape_NRK_;
  using ArchTag = cutlass::arch::Sm90;
  static constexpr int kBlockN = get<0>(TileShape_NRK{});
  static constexpr int R = get<1>(TileShape_NRK{});
  static constexpr int kBlockK = get<2>(TileShape_NRK{});

  static constexpr uint32_t kNumThreadsPerWarpGroup = 128;
  static constexpr uint32_t kActiveWarpGroups = 2;
  static constexpr uint32_t MaxThreadsPerBlock =
      kActiveWarpGroups * kNumThreadsPerWarpGroup;
  static constexpr uint32_t MinBlocksPerMultiprocessor = R == 64 ? 2 : 1;

  static constexpr uint32_t kNumLoadRegisters = 32;
  static constexpr uint32_t kNumComputeRegisters = 104;

  static_assert(kBlockN == 64);
  static_assert(R == 64 || R == 128);
  static_assert(kBlockK == 64);

  static constexpr int kClusterM = 1;
  static constexpr int kClusterN = 1;
  using ClusterShape = Shape<Int<kClusterM>, Int<kClusterN>, _1>;

  static constexpr int kNumMmaWarpgroups = 1;
  static constexpr int kNumMmaThreads =
      kNumMmaWarpgroups * kNumThreadsPerWarpGroup;
  using AtomLayoutMma = Layout<Shape<Int<kNumMmaWarpgroups>, _1, _1>>;

  using TiledMmaNRK = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<Element, Element, ElementAccum,
                                 TileShape_NRK>(),
      AtomLayoutMma{}));

  using SmemLayoutAtomBpEB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<kBlockN>, Int<kBlockK>>());
  using SmemLayoutBpEB = decltype(tile_to_shape(
      SmemLayoutAtomBpEB{}, Shape<Int<kBlockN>, Int<kBlockK>, Int<kStages>>{}));

  using SmemLayoutAtomEAR =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, Element, Int<R>, Int<kBlockK>>());
  using SmemLayoutEAR = decltype(tile_to_shape(
      SmemLayoutAtomEAR{}, Shape<Int<R>, Int<kBlockK>, Int<kStages>>{}));

  using SmemLayoutAtomEARxBpEB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<kBlockN>, Int<R>>());
  using SmemLayoutEARxBpEB = decltype(tile_to_shape(
      SmemLayoutAtomEARxBpEB{}, Shape<Int<kBlockN>, Int<R>>{}));

  using ShapeT = cute::Shape<int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using TMA_BpEB = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),
      take<0, 2>(SmemLayoutBpEB{}), select<0, 2>(TileShape_NRK{}), _1{}));

  using TMA_EAR = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeT{},
                  StrideT{}),
      take<0, 2>(SmemLayoutEAR{}), select<1, 2>(TileShape_NRK{}), _1{}));

  using GmemTiledCopyEARxBpEB =
      cute::conditional_t<UseReduction, cute::SM90_TMA_REDUCE_ADD,
                          cute::SM90_TMA_STORE>;

  using TMA_EARxBpEB = decltype(make_tma_copy(
      GmemTiledCopyEARxBpEB{},
      make_tensor(make_gmem_ptr(static_cast<ElementDenoise const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      SmemLayoutEARxBpEB{}, select<0, 1>(TileShape_NRK{}), _1{}));

  static constexpr uint32_t TmaTransactionBytesBpEB = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutBpEB{})) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesEAR = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutEAR{})) * cutlass::sizeof_bits_v<Element> / 8);

  using CopyOpR2S = AutoVectorizingCopyWithAssumedAlignment<128>;
  using SmemCopyAtomEARxBpEB = Copy_Atom<CopyOpR2S, ElementDenoise>;

  using MainloopLoadPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using LoadPipelineParams = typename MainloopLoadPipeline::Params;
  using LoadPipelineState = typename MainloopLoadPipeline::PipelineState;
  using LoadBarrierType = typename MainloopLoadPipeline::ProducerBarrierType;

  static constexpr size_t Alignment = 128;

  struct SharedStorage : cute::aligned_struct<Alignment> {
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutBpEB>, Alignment>
        smem_BpEB;
    cute::array_aligned<Element, cute::cosize_v<SmemLayoutEAR>, Alignment>
        smem_EAR;
    cute::array_aligned<ElementDenoise,
                        cute::cosize_v<SmemLayoutEARxBpEB>, Alignment>
        smem_EARxBpEB;

    struct {
      typename MainloopLoadPipeline::SharedStorage pipeline_BpEB;
      typename MainloopLoadPipeline::SharedStorage pipeline_EAR;
    };
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  struct Arguments {
    Element const* const ptr_BpEB;
    Element const* const ptr_EAR;
    ElementDenoise* const ptr_EARxBpEB;
    int n;
    int k;
    int num_k_blocks;
    int total_k_blocks;
  };

  struct Params {
    Element const* const ptr_BpEB;
    Element const* const ptr_EAR;
    ElementDenoise* const ptr_EARxBpEB;
    int n;
    int k;
    int num_k_blocks;
    int total_k_blocks;
    LayoutT layout_BpEB;
    LayoutT layout_EAR;
    LayoutT layout_EARxBpEB;
    TMA_BpEB tma_load_BpEB;
    TMA_EAR tma_load_EAR;
    TMA_EARxBpEB tma_store_EARxBpEB;
  };

  enum struct NamedBarriers {
    R2SCopyEARxBpEBDone,
  };

  static Params to_underlying_arguments(Arguments const& args) {
    LayoutT layout_BpEB =
        make_layout(make_shape(args.n, args.k), make_stride(args.k, _1{}));
    LayoutT layout_EAR =
        make_layout(make_shape(R, args.k), make_stride(args.k, _1{}));
    LayoutT layout_EARxBpEB =
        make_layout(make_shape(args.n, R), make_stride(R, _1{}));

    Tensor mBpEB = make_tensor(make_gmem_ptr(args.ptr_BpEB), layout_BpEB);
    TMA_BpEB tma_load_BpEB = make_tma_copy(
        cute::SM90_TMA_LOAD{}, mBpEB, take<0, 2>(SmemLayoutBpEB{}),
        select<0, 2>(TileShape_NRK{}), _1{});

    Tensor mEAR = make_tensor(make_gmem_ptr(args.ptr_EAR), layout_EAR);
    TMA_EAR tma_load_EAR =
        make_tma_copy(cute::SM90_TMA_LOAD{}, mEAR, take<0, 2>(SmemLayoutEAR{}),
                      select<1, 2>(TileShape_NRK{}), _1{});

    Tensor mEARxBpEB =
        make_tensor(make_gmem_ptr(args.ptr_EARxBpEB), layout_EARxBpEB);
    TMA_EARxBpEB tma_store_EARxBpEB =
        make_tma_copy(GmemTiledCopyEARxBpEB{}, mEARxBpEB,
                      SmemLayoutEARxBpEB{}, select<0, 1>(TileShape_NRK{}),
                      _1{});

    return {.ptr_BpEB = args.ptr_BpEB,
            .ptr_EAR = args.ptr_EAR,
            .ptr_EARxBpEB = args.ptr_EARxBpEB,
            .n = args.n,
            .k = args.k,
            .num_k_blocks = args.num_k_blocks,
            .total_k_blocks = args.total_k_blocks,
            .layout_BpEB = layout_BpEB,
            .layout_EAR = layout_EAR,
            .layout_EARxBpEB = layout_EARxBpEB,
            .tma_load_BpEB = tma_load_BpEB,
            .tma_load_EAR = tma_load_EAR,
            .tma_store_EARxBpEB = tma_store_EARxBpEB};
  }

  static dim3 get_grid_shape(Params const& params) {
    if constexpr (UseReduction) {
      return dim3(ceil_div(params.n, kBlockN),
                  ceil_div(params.k, params.num_k_blocks * kBlockK), 1);
    } else {
      return dim3(ceil_div(params.n, kBlockN), 1, 1);
    }
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

 private:
  CUTLASS_DEVICE void load_tensors(
      Params const& params, SharedStorage& shared_storage, const int n_block,
      const int k_block_min, const int k_block_max,
      MainloopLoadPipeline& pipeline_BpEB, MainloopLoadPipeline& pipeline_EAR,
      LoadPipelineState& smem_pipe_write_BpEB,
      LoadPipelineState& smem_pipe_write_EAR) {
    Tensor sBpEB = make_tensor(make_smem_ptr(shared_storage.smem_BpEB.data()),
                               SmemLayoutBpEB{});
    Tensor sEAR = make_tensor(make_smem_ptr(shared_storage.smem_EAR.data()),
                              SmemLayoutEAR{});

    Tensor mBpEB = params.tma_load_BpEB.get_tma_tensor(params.layout_BpEB.shape());
    Tensor mEAR = params.tma_load_EAR.get_tma_tensor(params.layout_EAR.shape());

    Tensor gBpEB =
        local_tile(mBpEB, select<0, 2>(TileShape_NRK{}), make_coord(n_block, _));
    Tensor gEAR =
        local_tile(mEAR, select<1, 2>(TileShape_NRK{}), make_coord(_0{}, _));

    if (cute::elect_one_sync()) {
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_block = k_block_min; k_block < k_block_max; ++k_block) {
        auto [tBpEBgBpEB, tBpEBsBpEB] =
            tma_partition(params.tma_load_BpEB, Int<0>{}, Layout<_1>{},
                          group_modes<0, 2>(sBpEB), group_modes<0, 2>(gBpEB));
        pipeline_BpEB.producer_acquire(smem_pipe_write_BpEB);
        LoadBarrierType* tmaBarBpEB =
            pipeline_BpEB.producer_get_barrier(smem_pipe_write_BpEB);
        copy(params.tma_load_BpEB.with(*tmaBarBpEB, 0),
             tBpEBgBpEB(_, k_block), tBpEBsBpEB(_, smem_pipe_write_BpEB.index()));
        pipeline_BpEB.producer_commit(smem_pipe_write_BpEB,
                                      TmaTransactionBytesBpEB);
        ++smem_pipe_write_BpEB;

        auto [tEARgEAR, tEARsEAR] =
            tma_partition(params.tma_load_EAR, Int<0>{}, Layout<_1>{},
                          group_modes<0, 2>(sEAR), group_modes<0, 2>(gEAR));
        pipeline_EAR.producer_acquire(smem_pipe_write_EAR);
        LoadBarrierType* tmaBarEAR =
            pipeline_EAR.producer_get_barrier(smem_pipe_write_EAR);
        copy(params.tma_load_EAR.with(*tmaBarEAR, 0), tEARgEAR(_, k_block),
             tEARsEAR(_, smem_pipe_write_EAR.index()));
        pipeline_EAR.producer_commit(smem_pipe_write_EAR,
                                     TmaTransactionBytesEAR);
        ++smem_pipe_write_EAR;
      }
    }
  }

 public:
  template <typename FrgTensorC>
  CUTLASS_DEVICE void compute_product(
      Params const& params, MainloopLoadPipeline& pipeline_BpEB,
      MainloopLoadPipeline& pipeline_EAR, SharedStorage& shared_storage,
      FrgTensorC& tCrEARxBpEB, const int k_block_min, const int k_block_max,
      const int tid) {
    Tensor sBpEB = make_tensor(make_smem_ptr(shared_storage.smem_BpEB.data()),
                               SmemLayoutBpEB{});
    Tensor sEAR = make_tensor(make_smem_ptr(shared_storage.smem_EAR.data()),
                              SmemLayoutEAR{});

    TiledMmaNRK tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    Tensor tCsBpEB = thr_mma.partition_A(sBpEB);
    Tensor tCsEAR = thr_mma.partition_B(sEAR);

    Tensor tCrBpEB = thr_mma.make_fragment_A(tCsBpEB);
    Tensor tCrEAR = thr_mma.make_fragment_B(tCsEAR);

    LoadPipelineState smem_pipe_read_BpEB;
    LoadPipelineState smem_pipe_read_EAR;

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_block = k_block_min; k_block < k_block_max; ++k_block) {
      pipeline_BpEB.consumer_wait(smem_pipe_read_BpEB);
      pipeline_EAR.consumer_wait(smem_pipe_read_EAR);
      cutlass::arch::fence_view_async_shared();
      warpgroup_fence_operand(tCrEARxBpEB);
      warpgroup_arrive();
      gemm(tiled_mma, tCrBpEB(_, _, _, smem_pipe_read_BpEB.index()),
           tCrEAR(_, _, _, smem_pipe_read_EAR.index()), tCrEARxBpEB);
      warpgroup_commit_batch();
      warpgroup_wait<0>();
      warpgroup_fence_operand(tCrEARxBpEB);

      cutlass::arch::fence_view_async_shared();
      pipeline_BpEB.consumer_release(smem_pipe_read_BpEB);
      pipeline_EAR.consumer_release(smem_pipe_read_EAR);
      ++smem_pipe_read_BpEB;
      ++smem_pipe_read_EAR;
    }
  }

  template <typename FrgTensorC>
  CUTLASS_DEVICE void store_EARxBpEB(Params const& params,
                                     FrgTensorC& tCrEARxBpEB,
                                     SharedStorage& shared_storage,
                                     const int n_block, const int tid) {
    Tensor sEARxBpEB =
        make_tensor(make_smem_ptr(shared_storage.smem_EARxBpEB.data()),
                    SmemLayoutEARxBpEB{});
    Tensor sEARxBpEB_pi = as_position_independent_swizzle_tensor(sEARxBpEB);

    TiledMmaNRK tiled_mma;
    auto r2s_tiled_copy_O =
        make_tiled_copy_C(SmemCopyAtomEARxBpEB{}, tiled_mma);
    auto r2s_thr_copy_O = r2s_tiled_copy_O.get_thread_slice(tid);
    Tensor tR2SsEARxBpEB = r2s_thr_copy_O.partition_D(sEARxBpEB_pi);

    Tensor mEARxBpEB = params.tma_store_EARxBpEB.get_tma_tensor(
        params.layout_EARxBpEB.shape());
    Tensor gEARxBpEB = local_tile(mEARxBpEB, select<0, 1>(TileShape_NRK{}),
                                  make_coord(n_block, _0{}));
    auto tma_thr_EARxBpEB = params.tma_store_EARxBpEB.get_slice(_0{});
    Tensor tOgEARxBpEB = tma_thr_EARxBpEB.partition_D(gEARxBpEB);
    Tensor tOsEARxBpEB = tma_thr_EARxBpEB.partition_S(sEARxBpEB);

    Tensor tR2SrEARxBpEB = r2s_thr_copy_O.retile_S(tCrEARxBpEB);
    cute::copy(r2s_tiled_copy_O, tR2SrEARxBpEB, tR2SsEARxBpEB);

    cutlass::arch::NamedBarrier::sync(
        kNumMmaThreads,
        static_cast<uint32_t>(NamedBarriers::R2SCopyEARxBpEBDone));
    cutlass::arch::fence_view_async_shared();

    int warp_idx_in_warpgroup = pearl::warp_idx_in_warpgroup_sync();
    if (warp_idx_in_warpgroup == 0 && cute::elect_one_sync()) {
      cute::copy(params.tma_store_EARxBpEB, tOsEARxBpEB, tOgEARxBpEB);
      tma_store_arrive();
      tma_store_wait<0>();
    }
  }

  CUTLASS_DEVICE void run_compute_consumer(
      Params const& params, MainloopLoadPipeline& pipeline_BpEB,
      MainloopLoadPipeline& pipeline_EAR, SharedStorage& shared_storage,
      const int n_block, const int k_block_min, const int k_block_max,
      const int tid) {
    TiledMmaNRK tiled_mma;
    Tensor tCrEARxBpEB =
        partition_fragment_C(tiled_mma, select<0, 1>(TileShape_NRK{}));
    clear(tCrEARxBpEB);

    compute_product(params, pipeline_BpEB, pipeline_EAR, shared_storage,
                    tCrEARxBpEB, k_block_min, k_block_max, tid);
    store_EARxBpEB(params, tCrEARxBpEB, shared_storage, n_block, tid);
  }

  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const tid = threadIdx.x;
    int const warp_idx = cutlass::canonical_warp_idx_sync();
    int const warp_group_idx = cutlass::canonical_warp_group_idx();
    int const warp_group_thread_idx = tid % cutlass::NumThreadsPerWarpGroup;
    int const lane_predicate = cute::elect_one_sync();

    if (warp_idx == 0 && lane_predicate) {
      cute::prefetch_tma_descriptor(params.tma_load_BpEB.get_tma_descriptor());
      cute::prefetch_tma_descriptor(params.tma_load_EAR.get_tma_descriptor());
      cute::prefetch_tma_descriptor(
          params.tma_store_EARxBpEB.get_tma_descriptor());
    }

    int const n_block = blockIdx.x;
    int const k_block_min =
        UseReduction ? blockIdx.y * params.num_k_blocks : 0;
    int const num_k_blocks_cta =
        cute::min(params.num_k_blocks, params.total_k_blocks - k_block_min);
    int const k_block_max = k_block_min + num_k_blocks_cta;

    LoadPipelineParams pipeline_params_BpEB;
    pipeline_params_BpEB.transaction_bytes = TmaTransactionBytesBpEB;
    pipeline_params_BpEB.role =
        warp_group_idx == 0 ? MainloopLoadPipeline::ThreadCategory::Producer
        : warp_group_idx == 1
            ? MainloopLoadPipeline::ThreadCategory::Consumer
            : MainloopLoadPipeline::ThreadCategory::NonParticipant;
    pipeline_params_BpEB.is_leader = warp_group_thread_idx == 0;
    pipeline_params_BpEB.num_consumers = kNumMmaThreads;
    MainloopLoadPipeline pipeline_BpEB(shared_storage.pipeline_BpEB,
                                       pipeline_params_BpEB, ClusterShape{});

    LoadPipelineParams pipeline_params_EAR;
    pipeline_params_EAR.transaction_bytes = TmaTransactionBytesEAR;
    pipeline_params_EAR.role =
        warp_group_idx == 0 ? MainloopLoadPipeline::ThreadCategory::Producer
        : warp_group_idx == 1
            ? MainloopLoadPipeline::ThreadCategory::Consumer
            : MainloopLoadPipeline::ThreadCategory::NonParticipant;
    pipeline_params_EAR.is_leader = warp_group_thread_idx == 0;
    pipeline_params_EAR.num_consumers = kNumMmaThreads;
    MainloopLoadPipeline pipeline_EAR(shared_storage.pipeline_EAR,
                                      pipeline_params_EAR, ClusterShape{});

    __syncthreads();

    if (warp_group_idx == 0) {
      cutlass::arch::warpgroup_reg_dealloc<kNumLoadRegisters>();
      if (pearl::warp_idx_in_warpgroup_sync() == 0) {
        LoadPipelineState smem_pipe_write_BpEB =
            cutlass::make_producer_start_state<MainloopLoadPipeline>();
        LoadPipelineState smem_pipe_write_EAR =
            cutlass::make_producer_start_state<MainloopLoadPipeline>();

        load_tensors(params, shared_storage, n_block, k_block_min, k_block_max,
                     pipeline_BpEB, pipeline_EAR, smem_pipe_write_BpEB,
                     smem_pipe_write_EAR);
      }
    } else if (warp_group_idx == 1) {
      constexpr int ThreadOffset = kNumThreadsPerWarpGroup;
      cutlass::arch::warpgroup_reg_alloc<kNumComputeRegisters>();
      run_compute_consumer(params, pipeline_BpEB, pipeline_EAR, shared_storage,
                           n_block, k_block_min, k_block_max,
                           tid - ThreadOffset);
    }
  }
};

}  // namespace pearl
