#pragma once

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include <cutlass/arch/arch.h>
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "pearl_gemm_constants.hpp"

namespace pearl {
using namespace cute;

template <typename ElementIn_, typename ElementOut_, typename ElementDenoise_,
          typename ElementScale_, typename TileShape_MNKR_, bool Is_Even_M_,
          bool Is_Even_N_, int cM_, int cN_, bool SkipReduction_,
          bool SkipDenoising_, int kStages_, bool EnableDebug_>
struct KernelTraits {

  using ElementIn = ElementIn_;
  using ElementScale = ElementScale_;
  using ElementAccum = int32_t;  // accum dtype for main gemm
  using ElementOut = ElementOut_;
  using ElementDenoise =
      ElementDenoise_;                // dtype of denoise matrices before gemm
  using ElementDenoiseAccum = float;  // accum dtype for denoise gemm
  using index_t = int64_t;

  using TileShape_MNKR = TileShape_MNKR_;
  static constexpr bool Is_Even_M = Is_Even_M_;
  static constexpr bool Is_Even_N = Is_Even_N_;
  static constexpr bool SkipReduction = SkipReduction_;
  static constexpr bool SkipDenoising = SkipDenoising_;
  // B-side noise fusion (B'): compile-time, default false (see pearl_gemm_constants.hpp).
  static constexpr bool FuseNoiseB = pearl::kFuseNoiseB;
  static constexpr int kStages = kStages_;
  static constexpr bool EnableDebug = EnableDebug_;
  static constexpr int srcLane = 0;

  using ProblemShape = Shape<int, int, int, int>;
  static_assert(is_same_v<ElementDenoise, half_t> ||
                is_same_v<ElementDenoise, int32_t>);
  static_assert(is_same_v<ElementDenoise, half_t> ||
                is_same_v<ElementDenoise, int32_t>);

  static constexpr int bM = get<0>(TileShape_MNKR{});
  static constexpr int bN = get<1>(TileShape_MNKR{});
  static constexpr int bK = get<2>(TileShape_MNKR{});
  static constexpr int R = get<3>(TileShape_MNKR{});

  // Use a 64 x bN tile per warpgroup; so thread count controlled by tile_size_m parameter
  static constexpr int kNumMmaWarpgroups = bM / 64;
  static constexpr int kNumMmaThreads = kNumMmaWarpgroups * 128;
  // Use one warp in producer warpgroup for TMA
  static constexpr int kNumProducerThreads = cutlass::NumThreadsPerWarp;
  static constexpr int kNumThreads = kNumMmaThreads + 128;
  static constexpr int kNumWarps = kNumThreads / cutlass::NumThreadsPerWarp;

  using TileShape_MNK = Shape<Int<bM>, Int<bN>, Int<bK>>;
  // used for denoising
  using TileShape_MNR = Shape<Int<bM>, Int<bN>, Int<R>>;

  static constexpr int kClusterSizeM = cM_;
  static constexpr int kClusterSizeN = cN_;
  using ClusterShape_MNK = Shape<Int<kClusterSizeM>, Int<kClusterSizeN>, _1>;
  // Multicasting to a single CTA has been known to be worse perf than non-multicast
  using TMAOpA =
      std::conditional_t<(kClusterSizeN > 1), cute::SM90_TMA_LOAD_MULTICAST,
                         cute::SM90_TMA_LOAD>;
  using TMAOpB =
      std::conditional_t<(kClusterSizeM > 1), cute::SM90_TMA_LOAD_MULTICAST,
                         cute::SM90_TMA_LOAD>;
  // GEMM traits
  using AtomLayoutMNK = Layout<Shape<Int<kNumMmaWarpgroups>, _1, _1>>;
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<ElementIn, ElementIn, ElementAccum,
                                 TileShape_MNK>(),
      AtomLayoutMNK{}));
  using MMATraits = typename TiledMma::Atom::Traits;
  using MMAAtomShape_MNK = typename TiledMma::AtomShape_MNK;
  using MMAAtom_K = decltype(get<2>(MMAAtomShape_MNK{}));

  // Fake warp layout and warp tile shape for reduce_buffer permutation
  // We assume <= 2 MMA warpgroups which are always tiled in the M direction.
  static constexpr int kWarpRows = 4 * kNumMmaWarpgroups;
  static constexpr int kWarpCols = 1;
  using MMAWarpLayout = Layout<Shape<Int<kWarpRows>, Int<kWarpCols>, _1>>;
  using MMAWarpTileShape = Shape<_16, _8, _32>;

  using SmemLayoutAtomA =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementIn,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}),
                 Int<kStages>{})));

  using SmemLayoutAtomB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementIn,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}),
                 Int<kStages>{})));

  using SmemLayoutAtomC =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementOut,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<1>(TileShape_MNK{}))>());
  using SmemLayoutC =
      decltype(tile_to_shape(SmemLayoutAtomC{}, select<0, 1>(TileShape_MNK{})));

  using SmemCopyAtomC = Copy_Atom<cute::SM90_U32x4_STSM_N, ElementOut>;

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;

  // Probably don't need more than 1 stage here because denoise load latency
  // is well-hidden under the rest of the matmul
  static constexpr int kDenoiseStages = 1;
  using DenoisePipeline = typename cutlass::PipelineTmaAsync<kDenoiseStages>;

  // Scales traits
  using G2SScales_copy_op = SM80_CP_ASYNC_CACHEALWAYS<ElementScale>;
  using G2SScales_copy_traits = Copy_Traits<G2SScales_copy_op>;
  using G2SScales_copy_atom = Copy_Atom<G2SScales_copy_traits, ElementScale>;
  using SmemLayoutScaleA = Layout<Shape<Int<bM>>, Stride<_1>>;
  using SmemLayoutScaleB = Layout<Shape<Int<bN>>, Stride<_1>>;

  using G2SScalesCopyA = decltype(make_tiled_copy(
      G2SScales_copy_atom{}, Layout<Shape<Int<bM>>, Stride<_1>>{},
      Layout<Shape<_1>, Stride<_1>>{}));
  using G2SScalesCopyB = decltype(make_tiled_copy(
      G2SScales_copy_atom{}, Layout<Shape<Int<bN>>, Stride<_1>>{},
      Layout<Shape<_1>, Stride<_1>>{}));

  // Denoising
  // MMA
  using AtomLayoutMNR = Layout<Shape<Int<kNumMmaWarpgroups>, _1, _1>>;
  using TiledMmaDenoise = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<ElementDenoise, ElementDenoise,
                                 ElementDenoiseAccum, TileShape_MNR>(),
      AtomLayoutMNR{}));
  // SMEM layouts
  using SmemLayoutEAL_Atom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<bM>, Int<R>>());
  using SmemLayoutEAL = decltype(tile_to_shape(
      SmemLayoutEAL_Atom{}, Shape<Int<bM>, Int<R>, Int<kDenoiseStages>>{}));

  using SmemLayoutEBR_Atom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<bN>, Int<R>>());
  using SmemLayoutEBR = decltype(tile_to_shape(
      SmemLayoutEBR_Atom{}, Shape<Int<bN>, Int<R>, Int<kDenoiseStages>>{}));

  // Other factors have 1 layout if fp16, or 2 layouts if int32
  using SmemLayoutAxEBL_Atom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<bM>, Int<R>>());
  using SmemLayoutAxEBL = decltype(tile_to_shape(
      SmemLayoutAxEBL_Atom{}, Shape<Int<bM>, Int<R>, Int<kDenoiseStages>>{}));

  using SmemLayoutEARxBpEB_Atom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementDenoise, Int<bN>, Int<R>>());
  using SmemLayoutEARxBpEB =
      decltype(tile_to_shape(SmemLayoutEARxBpEB_Atom{},
                             Shape<Int<bN>, Int<R>, Int<kDenoiseStages>>{}));

  static_assert(R % 16 == 0);  // needed for this MMA op

  // --- B' fusion: int8 EBL/EBR SMEM layouts (separate from the fp16 denoise
  // factors above) for forming BpEB = B + int8(EBL*EBR) on-chip. ---
  using SmemLayoutEBLi8_Atom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementIn, Int<bK>, Int<R>>());
  using SmemLayoutEBLi8 = decltype(tile_to_shape(
      SmemLayoutEBLi8_Atom{}, Shape<Int<bK>, Int<R>, Int<kStages>>{}));
  using SmemLayoutEBRi8_Atom =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K, ElementIn, Int<bN>, Int<R>>());
  using SmemLayoutEBRi8 = decltype(tile_to_shape(
      SmemLayoutEBRi8_Atom{}, Shape<Int<bN>, Int<R>>{}));

  // NOTE if you change these, also change the pipeline stages heuristic in
  // heuristics.hpp!
  struct SharedStorageDenoise : cute::aligned_struct<128> {
    // Overlapping to allow larger tile sizes.
    // Denoise factors are all fp16 and used as inputs to SS WGMMA. Currently
    //  all denoise factors' smem storage are disjoint with each other while
    //  overlapped with mainloop smem, so we wait for mainloop gemms to finish
    //  before starting loads for denoise factors.
    union {
      struct {
        cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutA>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutA{})>
            smem_A;
        cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutB>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutB{})>
            smem_B;
      };

      struct {
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutAxEBL>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutAxEBL{})>
            smem_AxEBL;
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEBR>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutEBR{})>
            smem_EBR;
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEAL>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutEAL{})>
            smem_EAL;
        cute::array_aligned<ElementDenoise, cute::cosize_v<SmemLayoutEARxBpEB>,
                            cutlass::detail::alignment_for_swizzle(
                                SmemLayoutEARxBpEB{})>
            smem_EARxBpEB;
      };

      cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutC>,
                          cutlass::detail::alignment_for_swizzle(SmemLayoutC{})>
          smem_C;
    };

    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleA>,
                        cutlass::detail::alignment_for_swizzle(
                            SmemLayoutScaleA{})>
        smem_scale_a;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleB>,
                        cutlass::detail::alignment_for_swizzle(
                            SmemLayoutScaleB{})>
        smem_scale_b;

#ifdef PEARL_FUSE_NOISE_B
    // B' fusion: int8 EBL/EBR staged for on-chip BpEB = B + int8(EBL*EBR).
    cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutEBLi8>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutEBLi8{})>
        smem_EBLi8;
    cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutEBRi8>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutEBRi8{})>
        smem_EBRi8;
#ifdef PEARL_FUSE_NOISE_B_V3
    cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutB>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutB{})>
        smem_BpEB;  // V3 decoupled BpEB buffer
#endif
#endif

    struct {
      typename MainloopPipeline::SharedStorage pipeline;
      typename DenoisePipeline::SharedStorage AxEB_pipeline;
      typename DenoisePipeline::SharedStorage EAxBpEB_pipeline;
    };
  };

  struct SharedStorageNoDenoise : cute::aligned_struct<128> {
    struct {
      cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutA>,
                          cutlass::detail::alignment_for_swizzle(SmemLayoutA{})>
          smem_A;
      cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutB>,
                          cutlass::detail::alignment_for_swizzle(SmemLayoutB{})>
          smem_B;
    };

    cute::array_aligned<ElementOut, cute::cosize_v<SmemLayoutC>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutC{})>
        smem_C;

    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleA>,
                        cutlass::detail::alignment_for_swizzle(
                            SmemLayoutScaleA{})>
        smem_scale_a;
    cute::array_aligned<ElementScale, cute::cosize_v<SmemLayoutScaleB>,
                        cutlass::detail::alignment_for_swizzle(
                            SmemLayoutScaleB{})>
        smem_scale_b;

#ifdef PEARL_FUSE_NOISE_B
    // B' fusion: int8 EBL/EBR staged for on-chip BpEB = B + int8(EBL*EBR).
    cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutEBLi8>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutEBLi8{})>
        smem_EBLi8;
    cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutEBRi8>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutEBRi8{})>
        smem_EBRi8;
#ifdef PEARL_FUSE_NOISE_B_V3
    cute::array_aligned<ElementIn, cute::cosize_v<SmemLayoutB>,
                        cutlass::detail::alignment_for_swizzle(SmemLayoutB{})>
        smem_BpEB;  // V3 decoupled BpEB buffer
#endif
#endif

    struct {
      typename MainloopPipeline::SharedStorage pipeline;
      typename DenoisePipeline::SharedStorage AxEB_pipeline;
      typename DenoisePipeline::SharedStorage EAxBpEB_pipeline;
    };
  };

  using SharedStorage =
      cute::conditional_t<SkipDenoising, SharedStorageNoDenoise,
                          SharedStorageDenoise>;
};

}  // namespace pearl
