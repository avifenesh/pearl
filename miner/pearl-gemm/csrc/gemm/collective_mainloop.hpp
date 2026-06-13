#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"

#include "blake3/blake3_constants.hpp"
#include "host_signal_header.hpp"
#include "pow_utils.hpp"
#include "utils.h"

namespace pearl {

using namespace cute;

template <typename KTraits>
struct CollectiveMainloop {

  using ElementIn = typename KTraits::ElementIn;
  using TileShape_MNK = typename KTraits::TileShape_MNK;
  using TileShape_MNR = typename KTraits::TileShape_MNR;

  using ProblemShape = typename KTraits::ProblemShape;
  using ClusterShape_MNK = typename KTraits::ClusterShape_MNK;

  static constexpr int kStages = KTraits::kStages;
  static constexpr int SkipReduction = KTraits::SkipReduction;
  static constexpr int kClusterSizeM = KTraits::kClusterSizeM;
  static constexpr int kClusterSizeN = KTraits::kClusterSizeN;
  static constexpr int srcLane = KTraits::srcLane;

  using MMAAtom_K = typename KTraits::MMAAtom_K;

  using SmemLayoutA = typename KTraits::SmemLayoutA;
  using SmemLayoutB = typename KTraits::SmemLayoutB;

  using ShapeT = cute::Shape<int32_t, int32_t>;
  using StrideT = cute::Shape<int32_t, _1>;
  using LayoutT = cute::Layout<ShapeT, StrideT>;
  using TMAOpA = KTraits::TMAOpA;
  using TMAOpB = KTraits::TMAOpB;

  // mcast in n direction of cluster
  using TMA_A = decltype(make_tma_copy(
      TMAOpA{},
      make_tensor(make_gmem_ptr(static_cast<ElementIn const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      take<0, 2>(SmemLayoutA{}), select<0, 2>(TileShape_MNK{}), kClusterSizeN));

  // mcast in m direction of cluster
  using TMA_B = decltype(make_tma_copy(
      TMAOpB{},
      make_tensor(make_gmem_ptr(static_cast<ElementIn const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      take<0, 2>(SmemLayoutB{}), select<1, 2>(TileShape_MNK{}), kClusterSizeM));

  static constexpr int kNumMmaThreads = KTraits::kNumMmaThreads;
  static constexpr bool FuseNoiseB = KTraits::FuseNoiseB;
  using MainloopPipeline = typename KTraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using BarrierType = typename MainloopPipeline::ProducerBarrierType;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesA = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutA{})) * cutlass::sizeof_bits_v<ElementIn> / 8);
  static constexpr uint32_t TmaTransactionBytesB = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutB{})) * cutlass::sizeof_bits_v<ElementIn> / 8);
  static constexpr uint32_t TmaTransactionBytes =
      TmaTransactionBytesA + TmaTransactionBytesB;

  // ===========================================================================
  // B' fusion types (only used when KTraits::FuseNoiseB). Mirror the noising_B
  // kernel so the on-chip BpEB = B + int8(EBL*EBR) is formed with the SAME WGMMA
  // path, but the result is written back into the swizzled SmemLayoutB (the GEMM
  // operand layout) instead of a store-layout buffer destined for HBM.
  // ===========================================================================
  static constexpr int R = KTraits::R;
  using ElementAccum = typename KTraits::ElementAccum;  // int32
  // B + EBR*EBL accumulator tile: (bN, bK, R) as in noising_B (TiledMmaNKR).
  using TileShape_NKR =
      cute::Shape<cute::Int<KTraits::bN>, cute::Int<KTraits::bK>, cute::Int<R>>;
  using TiledMmaNoiseB = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<ElementIn, ElementIn, ElementAccum,
                                 TileShape_NKR>(),
      typename KTraits::AtomLayoutMNK{}));
  // EBR: (bN, R) R-major, no pipeline (constant over k). EBL: (bK, R) R-major.
  using SmemLayoutAtomEBR =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementIn, cute::Int<KTraits::bN>,
               cute::Int<R>>());
  using SmemLayoutEBR = decltype(tile_to_shape(
      SmemLayoutAtomEBR{}, cute::Shape<cute::Int<KTraits::bN>, cute::Int<R>>{}));
  using SmemLayoutAtomEBL =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementIn, cute::Int<KTraits::bK>,
               cute::Int<R>>());
  using SmemLayoutEBL = decltype(tile_to_shape(
      SmemLayoutAtomEBL{},
      cute::Shape<cute::Int<KTraits::bK>, cute::Int<R>, cute::Int<kStages>>{}));
  // Half TiledMma + S2R/R2S atoms to read B from / write BpEB into swizzled sB,
  // matching noising_B's compute_BpEB (16-bit STSM/LDSM over int8 pairs).
  using TileShape_NKR_half =
      cute::Shape<cute::Int<KTraits::bN>, cute::Int<KTraits::bK / 2>,
                  cute::Int<R>>;
  using TiledMmaNoiseB_half = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<ElementIn, ElementIn, ElementAccum,
                                 TileShape_NKR_half>(),
      typename KTraits::AtomLayoutMNK{}));
  using S2RCopyAtomB = Copy_Atom<SM75_U32x4_LDSM_N, uint16_t>;
  using R2SCopyAtomB = Copy_Atom<SM90_U32x4_STSM_N, uint16_t>;

  // TMA loaders for the int8 noise factors (only used when FuseNoiseB).
  using TMA_EBL = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementIn const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      take<0, 2>(SmemLayoutEBL{}),
      cute::Shape<cute::Int<KTraits::bK>, cute::Int<R>>{}, _1{}));
  using TMA_EBR = decltype(make_tma_copy(
      cute::SM90_TMA_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementIn const*>(nullptr)),
                  ShapeT{}, StrideT{}),
      SmemLayoutEBR{},
      cute::Shape<cute::Int<KTraits::bN>, cute::Int<R>>{}, _1{}));
  static constexpr uint32_t TmaTransactionBytesEBL = static_cast<uint32_t>(
      size(take<0, 2>(SmemLayoutEBL{})) * cutlass::sizeof_bits_v<ElementIn> / 8);
  static constexpr uint32_t TmaTransactionBytesEBR = static_cast<uint32_t>(
      size(SmemLayoutEBR{}) * cutlass::sizeof_bits_v<ElementIn> / 8);

  struct Arguments {
    ElementIn const* ptr_A;
    ElementIn const* ptr_B;
    void* host_signal_header_pinned;
    void* host_signal_sync;
    ProblemShape const problem_shape;
    uint64_t* inner_hash_counter;
    uint32_t const* ptr_pow_target;
    uint32_t const* ptr_pow_key;
    // B-side fusion (B'): raw (un-noised) weights + skinny noise factors. When
    // kFuseNoiseB, ptr_B points at raw B and BpEB is formed on-chip from these.
    // Null/ignored in the stock (kFuseNoiseB==false) path.
    ElementIn const* ptr_EBL_R_major = nullptr;  // (k, R)
    ElementIn const* ptr_EBR = nullptr;          // (R, n)
  };

  struct Params {
    ElementIn const* ptr_A;  // needed for host signal
    ElementIn const* ptr_B;  // needed for host signal
    LayoutT layout_A;
    LayoutT layout_B;
    TMA_A tma_load_A;
    TMA_B tma_load_B;
    HostSignalHeader* host_signal_header_pinned;
    HostSignalSync* host_signal_sync;
    ProblemShape const problem_shape;
    uint64_t* inner_hash_counter;
    uint32_t const* ptr_pow_target;
    uint32_t const* ptr_pow_key;
    ElementIn const* ptr_EBL_R_major = nullptr;  // (k, R), only when kFuseNoiseB
    ElementIn const* ptr_EBR = nullptr;          // (R, n), only when kFuseNoiseB
    LayoutT layout_EBL;
    LayoutT layout_EBR;
    TMA_EBL tma_load_EBL;
    TMA_EBR tma_load_EBR;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    auto [M, N, K, R] = args.problem_shape;
    LayoutT layout_A = make_layout(make_shape(M, K), make_stride(K, _1{}));
    LayoutT layout_B = make_layout(make_shape(N, K), make_stride(K, _1{}));
    Tensor mA = make_tensor(make_gmem_ptr(args.ptr_A), layout_A);
    Tensor mB = make_tensor(make_gmem_ptr(args.ptr_B), layout_B);
    // tile is divided into kClusterSizeN or kClusterSizeM many pieces to be multicasted
    // mcast in n direction of cluster
    TMA_A tma_load_A =
        make_tma_copy(TMAOpA{}, mA, SmemLayoutA{}(_, _, _0{}),
                      select<0, 2>(TileShape_MNK{}), kClusterSizeN);
    // mcast in m direction of cluster
    TMA_B tma_load_B =
        make_tma_copy(TMAOpB{}, mB, SmemLayoutB{}(_, _, _0{}),
                      select<1, 2>(TileShape_MNK{}), kClusterSizeM);

    // B' fusion noise-factor TMA loaders. Only construct real descriptors when
    // fusing (else TMA init validates bogus EBL/EBR layouts over ptr_B and fails).
    LayoutT layout_EBL =
        make_layout(make_shape(K, (int32_t)R), make_stride((int32_t)R, _1{}));
    LayoutT layout_EBR =
        make_layout(make_shape(N, (int32_t)R), make_stride((int32_t)R, _1{}));
    TMA_EBL tma_load_EBL{};
    TMA_EBR tma_load_EBR{};
    if constexpr (FuseNoiseB) {
      // Only when the noise factors are actually present at runtime (mining on).
      if (args.ptr_EBL_R_major != nullptr && args.ptr_EBR != nullptr) {
        Tensor mEBL =
            make_tensor(make_gmem_ptr(args.ptr_EBL_R_major), layout_EBL);
        tma_load_EBL = make_tma_copy(
            cute::SM90_TMA_LOAD{}, mEBL, take<0, 2>(SmemLayoutEBL{}),
            cute::Shape<cute::Int<KTraits::bK>, cute::Int<KTraits::R>>{}, _1{});
        Tensor mEBR = make_tensor(make_gmem_ptr(args.ptr_EBR), layout_EBR);
        tma_load_EBR = make_tma_copy(
            cute::SM90_TMA_LOAD{}, mEBR, SmemLayoutEBR{},
            cute::Shape<cute::Int<KTraits::bN>, cute::Int<KTraits::R>>{}, _1{});
      }
    }

    return {.ptr_A = args.ptr_A,
            .ptr_B = args.ptr_B,
            .layout_A = layout_A,
            .layout_B = layout_B,
            .tma_load_A = tma_load_A,
            .tma_load_B = tma_load_B,
            .host_signal_header_pinned = reinterpret_cast<HostSignalHeader*>(
                args.host_signal_header_pinned),
            .host_signal_sync =
                reinterpret_cast<HostSignalSync*>(args.host_signal_sync),
            .problem_shape = args.problem_shape,
            .inner_hash_counter = args.inner_hash_counter,
            .ptr_pow_target = args.ptr_pow_target,
            .ptr_pow_key = args.ptr_pow_key,
            .ptr_EBL_R_major = args.ptr_EBL_R_major,
            .ptr_EBR = args.ptr_EBR,
            .layout_EBL = layout_EBL,
            .layout_EBR = layout_EBR,
            .tma_load_EBL = tma_load_EBL,
            .tma_load_EBR = tma_load_EBR};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_A.get_tma_descriptor());
    cute::prefetch_tma_descriptor(
        mainloop_params.tma_load_B.get_tma_descriptor());
  }

  template <typename SharedStorage>
  CUTLASS_DEVICE void load(Params const& mainloop_params,
                           MainloopPipeline pipeline,
                           PipelineState& smem_pipe_write,
                           SharedStorage& shared_storage,
                           cute::tuple<int32_t, int32_t, int32_t> block_coord,
                           int k_tile_count, uint16_t const tma_mcast_mask_a,
                           uint16_t const tma_mcast_mask_b) {

    // Fetch logical block coordinates
    auto [m_block, n_block, bidb] = block_coord;

    // Define SMEM tensors
    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.data()),
                            SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()),
                            SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

    // Define GMEM tensors as TMA tensors
    Tensor mA = mainloop_params.tma_load_A.get_tma_tensor(
        mainloop_params.layout_A.shape());
    Tensor mB = mainloop_params.tma_load_B.get_tma_tensor(
        mainloop_params.layout_B.shape());

    // Get CTA views of GMEM
    Tensor gA = local_tile(mA, select<0, 2>(TileShape_MNK{}),
                           make_coord(m_block, _));  // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB, select<1, 2>(TileShape_MNK{}),
                           make_coord(n_block, _));  // (BLK_N,BLK_K,k)

    // Partition the copying of A and B tiles, including which part of the tile this
    //  CTA is responsible for when participating in multicast
    auto [tAgA, tAsA] =
        tma_partition(mainloop_params.tma_load_A, get<1>(block_id_in_cluster()),
                      make_layout(kClusterSizeN), group_modes<0, 2>(sA),
                      group_modes<0, 2>(gA));  // (TMA,k) and (TMA,PIPE)
    auto [tBgB, tBsB] =
        tma_partition(mainloop_params.tma_load_B, get<0>(block_id_in_cluster()),
                      make_layout(kClusterSizeM), group_modes<0, 2>(sB),
                      group_modes<0, 2>(gB));  // (TMA,k) and (TMA,PIPE)
    // DO TMA LOAD from a single thread
    int lane_predicate = cute::elect_one_sync();

    if constexpr (!KTraits::SkipDenoising) {
      // Wait for EAxBpEB matmul to finish on previous tile before loading current tile A, B
      cutlass::arch::NamedBarrier::sync(
          kNumMmaThreads + cutlass::NumThreadsPerWarp,
          static_cast<cutlass::arch::ReservedNamedBarriers>(
              pearl::NamedBarriers::DenoiseComplete));
    }

    if (lane_predicate) {
      // MAINLOOP LOADS
      CUTLASS_PRAGMA_NO_UNROLL
      for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        pipeline.producer_acquire(smem_pipe_write);
        BarrierType* tmaBar = pipeline.producer_get_barrier(smem_pipe_write);
        auto stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_A.with(*tmaBar, tma_mcast_mask_a),
             tAgA(_, k_tile), tAsA(_, stage));
        copy(mainloop_params.tma_load_B.with(*tmaBar, tma_mcast_mask_b),
             tBgB(_, k_tile), tBsB(_, stage));
        if constexpr (FuseNoiseB) {
          // Load this k-tile's EBL (bK,R) into the gated SMEM ring on the SAME
          // mbarrier/stage as A,B so it is ready when the consumer transforms sB.
          Tensor sEBL = make_tensor(
              make_smem_ptr(shared_storage.smem_EBLi8.data()),
              typename KTraits::SmemLayoutEBLi8{});
          Tensor mEBL = mainloop_params.tma_load_EBL.get_tma_tensor(
              mainloop_params.layout_EBL.shape());
          Tensor gEBL = local_tile(
              mEBL, cute::Shape<cute::Int<KTraits::bK>, cute::Int<R>>{},
              make_coord(_, _0{}));  // (bK,R,k)
          auto [tEgE, tEsE] = tma_partition(
              mainloop_params.tma_load_EBL, _0{}, make_layout(_1{}),
              group_modes<0, 2>(sEBL), group_modes<0, 2>(gEBL));
          copy(mainloop_params.tma_load_EBL.with(*tmaBar, 0),
               tEgE(_, k_tile), tEsE(_, stage));
          // EBR (bN,R) is constant over k; load it once (k_tile==0) on the same
          // mbarrier into the single-buffer gated SMEM.
          if (k_tile == 0) {
            Tensor sEBR = make_tensor(
                make_smem_ptr(shared_storage.smem_EBRi8.data()),
                typename KTraits::SmemLayoutEBRi8{});
            Tensor mEBR = mainloop_params.tma_load_EBR.get_tma_tensor(
                mainloop_params.layout_EBR.shape());
            Tensor gEBR = local_tile(
                mEBR, cute::Shape<cute::Int<KTraits::bN>, cute::Int<R>>{},
                make_coord(n_block, _0{}));  // (bN,R)
            auto [tRgR, tRsR] = tma_partition(
                mainloop_params.tma_load_EBR, _0{}, make_layout(_1{}),
                group_modes<0, 2>(sEBR), group_modes<0, 2>(gEBR));
            copy(mainloop_params.tma_load_EBR.with(*tmaBar, 0), tRgR, tRsR);
          }
        }
        uint32_t commit_bytes = TmaTransactionBytes;
        if constexpr (FuseNoiseB) {
          commit_bytes += TmaTransactionBytesEBL;
          if (k_tile == 0) commit_bytes += TmaTransactionBytesEBR;
        }
        pipeline.producer_commit(smem_pipe_write, commit_bytes);
        ++smem_pipe_write;
      }
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline,
                                PipelineState& smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup =
        __shfl_sync(0xffffffff,
                    (threadIdx.x / cutlass::NumThreadsPerWarp) %
                        cutlass::NumWarpsPerWarpGroup,
                    srcLane);
    // Issue the epilogue waits
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
          * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
          * then would just be acquired since the phase was still inverted from make_producer_start_state
          */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  CUTLASS_DEVICE void mma_init() {
    if constexpr (!KTraits::SkipDenoising) {
      // Allow producer warp to issue initial loads of A and B
      cutlass::arch::NamedBarrier::arrive(
          kNumMmaThreads + cutlass::NumThreadsPerWarp,
          static_cast<cutlass::arch::ReservedNamedBarriers>(
              pearl::NamedBarriers::DenoiseComplete));
    }
  }

  // B' fusion: transform sB[stage] in place from raw B to BpEB = B + int8(EBL*EBR).
  // Runs on the consumer MMA warpgroup(s) using the existing TiledMma partitioning,
  // mirroring noising_B::compute_BpEB but writing back into the swizzled sB instead
  // of a store-layout buffer. Must be called AFTER pipeline.consumer_wait(stage)
  // (B + EBL present) and BEFORE the main GEMM consumes sB for this stage.
  template <typename SharedStorage>
  CUTLASS_DEVICE void fuse_noise_b_inplace(SharedStorage& shared_storage,
                                           int stage, int thread_idx) {
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.data()),
                            SmemLayoutB{});
    Tensor sB_pi = as_position_independent_swizzle_tensor(sB);
    Tensor sEBR = make_tensor(make_smem_ptr(shared_storage.smem_EBRi8.data()),
                              typename KTraits::SmemLayoutEBRi8{});
    Tensor sEBL = make_tensor(make_smem_ptr(shared_storage.smem_EBLi8.data()),
                              typename KTraits::SmemLayoutEBLi8{});

    // WGMMA E_B = EBR * EBL -> int32 accumulator (bN, bK), as in compute_BpEB.
    TiledMmaNoiseB tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);
    Tensor tCrEB = partition_fragment_C(tiled_mma,
                                        cute::Shape<cute::Int<KTraits::bN>,
                                                    cute::Int<KTraits::bK>>{});
    Tensor tCrEB_int8 = make_fragment_like<ElementIn>(tCrEB);
    Tensor tCsEBR = thr_mma.partition_A(sEBR);
    Tensor tCrEBR = thr_mma.make_fragment_A(tCsEBR);
    Tensor tCsEBL = thr_mma.partition_B(sEBL);
    Tensor tCrEBL = thr_mma.make_fragment_B(tCsEBL);

    // S2R / R2S for reading B from and writing BpEB into the swizzled sB.
    TiledMmaNoiseB_half tiled_mma_half;
    auto s2r_tiled_copy_B = make_tiled_copy_C(S2RCopyAtomB{}, tiled_mma_half);
    auto s2r_thr_copy_B = s2r_tiled_copy_B.get_slice(thread_idx);
    auto sB_u16 = recast<uint16_t>(sB_pi);
    auto taccCsB = s2r_thr_copy_B.partition_S(sB_u16);
    auto tCrB = partition_fragment_C(
        tiled_mma_half,
        cute::Shape<cute::Int<KTraits::bN>, cute::Int<KTraits::bK / 2>>{});
    auto tCrB_u16 = make_tensor_like<uint16_t>(tCrB);
    auto taccCrB = s2r_thr_copy_B.retile_D(tCrB_u16);
    auto taccCrB_int8 = recast<ElementIn>(taccCrB);
    auto r2s_tiled_copy_B = make_tiled_copy_C(R2SCopyAtomB{}, tiled_mma_half);
    auto r2s_thr_copy_B = r2s_tiled_copy_B.get_slice(thread_idx);

    // E_B = EBR * EBL
    clear(tCrEB);
    warpgroup_fence_operand(tCrEB);
    warpgroup_arrive();
    gemm(tiled_mma, tCrEBR, tCrEBL(_, _, _, stage), tCrEB);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrEB);

    // Load current (raw) B tile from swizzled sB into registers.
    cute::copy(s2r_tiled_copy_B, taccCsB(_, _, _, stage), taccCrB);
    cutlass::arch::NamedBarrier::sync(
        kNumMmaThreads,
        static_cast<uint32_t>(pearl::NamedBarriers::S2RCopyBDone));
    cutlass::arch::fence_view_async_shared();

    // Narrow E_B int32 -> int8 (exact; noise is in-range), shuffle, add to B.
    pearl::convert_type_out(tCrEB, tCrEB_int8);
    permute_Aregs_fp8(tCrEB_int8);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(taccCrB_int8); ++i) {
      taccCrB_int8[i] += tCrEB_int8[i];
    }
    // Write BpEB back into the swizzled sB[stage] (R2S), in place. One barrier
    // ensures the write is complete + visible before the main GEMM reads sB.
    auto taccCsB_w = r2s_thr_copy_B.partition_D(sB_u16);
    cute::copy(r2s_tiled_copy_B, taccCrB, taccCsB_w(_, _, _, stage));
    cutlass::arch::fence_view_async_shared();
    cutlass::arch::NamedBarrier::sync(
        kNumMmaThreads,
        static_cast<uint32_t>(pearl::NamedBarriers::S2RCopyBDone));
  }

  template <typename SharedStorage, typename FrgTensorC,
            typename TranscriptTensor>
  CUTLASS_DEVICE void mma(Params const& mainloop_params,
                          MainloopPipeline pipeline,
                          PipelineState& smem_pipe_read, FrgTensorC& tCrC,
                          TranscriptTensor& transcript_extraction_tensor,
                          bool& block_found, int& block_found_k_tile,
                          int thread_idx, SharedStorage& shared_storage,
                          int k_tile_count) {

    Tensor sA =
        make_tensor(make_smem_ptr(shared_storage.smem_A.data()), SmemLayoutA{});
    Tensor sB =
        make_tensor(make_smem_ptr(shared_storage.smem_B.data()), SmemLayoutB{});

    typename KTraits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);

    Tensor tCsA = thr_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments" -- these are WGMMA matrix descriptors
    Tensor tCrA = thr_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)

    const uint32_t last_full_k_block =
        shape<1>(mainloop_params.layout_A) / MMAAtom_K{};

    // Compile-time constants for tile hash accumulation
    constexpr int k_blocks_per_tile = size<2>(tCrA);
    // R/32
    constexpr int reduce_every_k = get<2>(TileShape_MNR{}) / MMAAtom_K{};

    using HashAccumulator =
        TileHashAccumulator<k_blocks_per_tile, reduce_every_k,
                            KTraits::EnableDebug>;
    HashAccumulator hash_accumulator(last_full_k_block,
                                     mainloop_params.inner_hash_counter);

    CUTLASS_PRAGMA_NO_UNROLL
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      if constexpr (!SkipReduction) {
        hash_accumulator.preload(transcript_extraction_tensor);
      }

      // Wait for TMA to load this stage of the pipeline
      pipeline.consumer_wait(smem_pipe_read);
      auto stage = smem_pipe_read.index();

      if constexpr (FuseNoiseB) {
        // Transform raw B in sB[stage] -> BpEB = B + int8(EBL*EBR), in place,
        // before the main GEMM consumes it. Only when noise factors are actually
        // present (mining on); otherwise sB already holds the operand to use.
        if (mainloop_params.ptr_EBR != nullptr &&
            mainloop_params.ptr_EBL_R_major != nullptr) {
          fuse_noise_b_inplace(shared_storage, stage, thread_idx);
        }
      }

      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < k_blocks_per_tile; ++k_block) {
        warpgroup_fence_operand(tCrC);
        warpgroup_arrive();
        // WGMMA with dispatch mode (V,M,K) x (V,N,K) => (V,M,N)
        gemm(tiled_mma, tCrA(_, _, k_block, stage), tCrB(_, _, k_block, stage),
             tCrC);
        warpgroup_commit_batch();

        if constexpr (!SkipReduction) {
          hash_accumulator.accumulate(tCrC, k_block);
        }
      }

      // Write back transcript elements after tile completes
      if constexpr (!SkipReduction) {
        hash_accumulator.writeback(transcript_extraction_tensor);
      }

      warpgroup_wait<0>();
      // Release the stage of the pipeline for TMA
      pipeline.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
    }

    // Notify producer that main gemm is complete
    cutlass::arch::NamedBarrier::arrive(
        kNumMmaThreads + cutlass::NumThreadsPerWarp,
        static_cast<cutlass::arch::ReservedNamedBarriers>(
            pearl::NamedBarriers::MmaComplete));
  }
};

}  // namespace pearl
