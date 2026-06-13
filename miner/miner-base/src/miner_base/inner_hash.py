from dataclasses import dataclass

import numpy as np
import torch


def xor_reduction(inputs: torch.Tensor) -> np.uint32:
    """
    XOR all uint32 elements in the input tensor.
    Returns a single uint32 value.
    """
    assert inputs.dtype == torch.int32, f"Expected int32, got {inputs.dtype}"
    arr = inputs.flatten().numpy().view(np.uint32)
    return np.bitwise_xor.reduce(arr)


@dataclass
class InnerHashResult:
    hash: np.uint32
    index: tuple[int, int]


def hash_tile(
    tensor: torch.Tensor,
    index: tuple[int, int] | None = None,
) -> InnerHashResult:
    if tensor.dtype != torch.int32:
        raise ValueError(f"Tensor dtype must be int32, got {tensor.dtype}")

    final_hash = xor_reduction(tensor)

    return InnerHashResult(hash=final_hash, index=index if index is not None else (0, 0))


class InnerHasher:
    def __init__(self, tile_h: int, tile_w: int):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.num_tiles_hashed = 0

    def hash_tensor(self, tensor: torch.Tensor) -> list[InnerHashResult]:
        if tensor.dtype != torch.int32:
            raise ValueError(f"Tensor dtype must be int32, got {tensor.dtype}")

        if tensor.shape[0] < self.tile_h or tensor.shape[1] < self.tile_w:
            raise ValueError(
                f"Tensor must have shape of at least ({self.tile_h}, {self.tile_w}), got {tensor.shape}"
            )

        num_tiles_h = tensor.shape[0] // self.tile_h
        num_tiles_w = tensor.shape[1] // self.tile_w

        self.num_tiles_hashed += num_tiles_h * num_tiles_w

        # Vectorized per-tile XOR reduction: reshape into a (nh, nw, tile_h*tile_w)
        # grid and XOR-reduce the flattened tile axis in one numpy op, instead of
        # a Python double loop with a slice + reduce + object per tile. This is
        # bit-identical to the per-tile xor_reduction loop (the tiles cover the
        # full-tile region [: nh*tile_h, : nw*tile_w] in the same row/col order).
        cropped = tensor[: num_tiles_h * self.tile_h, : num_tiles_w * self.tile_w]
        grid = (
            cropped.reshape(num_tiles_h, self.tile_h, num_tiles_w, self.tile_w)
            .permute(0, 2, 1, 3)
            .reshape(num_tiles_h, num_tiles_w, self.tile_h * self.tile_w)
        )
        tile_hashes = np.bitwise_xor.reduce(grid.numpy().view(np.uint32), axis=2)

        return [
            InnerHashResult(hash=tile_hashes[i, j], index=(i, j))
            for i in range(num_tiles_h)
            for j in range(num_tiles_w)
        ]
