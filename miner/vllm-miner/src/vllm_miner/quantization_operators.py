# Importing the package registers the `pearl_gemm::*` torch custom ops (and their
# fakes) used below. We call the ops via ``torch.ops.pearl_gemm.*`` rather than the
# raw pybind functions so the quantization step is traceable by torch.compile /
# Dynamo (required for piecewise CUDA-graph capture; a raw pybind call is "marked
# as skipped" and breaks full-graph capture of the model).
import pearl_gemm  # noqa: F401  (registers torch.ops.pearl_gemm.*)
import torch

from .mining_state import get_async_manager

MAX_VAL_7BIT = 63
MAX_VAL_8BIT = 127


def quantize_kernel(x: torch.Tensor, max_val: int = 63, smooth_scale: torch.Tensor | None = None):
    """
    Symmetric per-token quantization with optional smooth scaling (CUDA kernel version)

    Args:
        x: Input tensor (any dtype)
        max_val: Maximum quantization value (63 for 7-bit, 127 for 8-bit)
        smooth_scale: Optional smooth_quant_scale to divide input by

    Returns:
        xq: Quantized int8 tensor
        xq_scales: Per-token scales (fp32)
        None: Zero point (not used for symmetric quant)
    """
    fast_math = get_async_manager()._conf.quantization_fast_math
    num_tokens, _ = x.shape
    x_q = torch.empty_like(x, dtype=torch.int8)
    x_s = torch.empty((num_tokens, 1), dtype=torch.float32, device=x.device)
    # Use the registered custom op (traceable) instead of the raw pybind call.
    torch.ops.pearl_gemm.quantize(x, x_q, x_s, max_val, smooth_scale, fast_math)
    return x_q, x_s, None


# Convenience wrappers
def quant_7bit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
    """7-bit symmetric quantization (range: [-63, 63])."""
    return quantize_kernel(x, max_val=MAX_VAL_7BIT, smooth_scale=None)


def quant_8bit(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
    """8-bit symmetric quantization (range: [-127, 127])."""

    return quantize_kernel(x, max_val=MAX_VAL_8BIT, smooth_scale=None)


def quant_7bit_smooth(
    x: torch.Tensor, smooth_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """7-bit symmetric quantization (range: [-63, 63]) with smooth scale."""
    return quantize_kernel(x, max_val=MAX_VAL_7BIT, smooth_scale=smooth_scale)


def quant_8bit_smooth(
    x: torch.Tensor, smooth_scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """8-bit symmetric quantization (range: [-127, 127]) with smooth scale."""
    return quantize_kernel(x, max_val=MAX_VAL_8BIT, smooth_scale=smooth_scale)
