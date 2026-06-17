#include <cutlass/numeric_types.h>

#include "gemm/pearl_gemm_host.h"
#include "gemm/bpeb_ear_product_host.h"
#include "gemm/pearl_api_params.h"

template <class ElementDenoise_EARxBpEB, int R, int bN_noising, int bK_noising,
          int kStages, bool UseReduction>
void run_bpeb_ear_product_(PearlAPIParams& params, cudaStream_t stream = 0) {
  using namespace cute;
  using TileShape_NRK = Shape<Int<bN_noising>, Int<R>, Int<bK_noising>>;

  run_bpeb_ear_product<ElementDenoise_EARxBpEB, TileShape_NRK, kStages,
                       UseReduction>(params, stream);
}

template void run_bpeb_ear_product_<int, 64, 64, 64, 2, false>(
    PearlAPIParams&, cudaStream_t);
template void run_bpeb_ear_product_<int, 64, 64, 64, 2, true>(
    PearlAPIParams&, cudaStream_t);
template void run_bpeb_ear_product_<int, 128, 64, 64, 2, false>(
    PearlAPIParams&, cudaStream_t);
template void run_bpeb_ear_product_<int, 128, 64, 64, 2, true>(
    PearlAPIParams&, cudaStream_t);
