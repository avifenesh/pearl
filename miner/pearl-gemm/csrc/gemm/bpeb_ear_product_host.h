#pragma once

#include "bpeb_ear_product_kernel.h"
#include "error_check.hpp"
#include "pearl_api_params.h"

#include "cute/tensor.hpp"

#include <cutlass/device_kernel.h>

template <typename ElementDenoise_EARxBpEB, class TileShape_NRK, int kStages,
          bool UseReduction>
void run_bpeb_ear_product(PearlAPIParams const& params,
                          cudaStream_t stream = 0) {
  using namespace cute;
  using Element = int8_t;
  using NoisingProductKernel =
      pearl::BpEBEARProductKernel<TileShape_NRK, 128, Element,
                                  ElementDenoise_EARxBpEB, kStages,
                                  UseReduction>;

  int total_k_blocks = ceil_div(params.k, get<2>(TileShape_NRK{}));
  typename NoisingProductKernel::Arguments args{
      .ptr_BpEB = static_cast<Element const*>(params.ptr_BpEB),
      .ptr_EAR = static_cast<Element const*>(params.ptr_EAR_K_major),
      .ptr_EARxBpEB =
          static_cast<ElementDenoise_EARxBpEB*>(params.ptr_EARxBpEB),
      .n = params.n,
      .k = params.k,
      .num_k_blocks = UseReduction ? params.k_blocks_per_split_noising_B
                                   : total_k_blocks,
      .total_k_blocks = total_k_blocks};

  typename NoisingProductKernel::Params kernel_params =
      NoisingProductKernel::to_underlying_arguments(args);

  dim3 grid_dims = NoisingProductKernel::get_grid_shape(kernel_params);
  dim3 block_dims = NoisingProductKernel::get_block_shape();
  constexpr static int smem_size = NoisingProductKernel::SharedStorageSize;

  auto kernel = cutlass::device_kernel<NoisingProductKernel>;
  if (smem_size >= 48 * 1024) {
    gpuErrchk(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
  CHECK_CUDA_KERNEL_LAUNCH();
}
