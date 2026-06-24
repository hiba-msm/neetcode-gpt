import torch
from torchtyping import TensorType
from typing import Tuple


class Solution:
    def create_batches(
        self,
        data: TensorType[int],
        context_length: int,
        batch_size: int
    ) -> Tuple[TensorType[int], TensorType[int]]:
        torch.manual_seed(0)

        # valid starts must allow Y to access start + context_length
        max_start = len(data) - context_length

        starts = torch.randint(0, max_start, (batch_size,))

        offsets = torch.arange(context_length)

        indices = starts[:, None] + offsets[None, :]

        X = data[indices]
        Y = data[indices + 1]

        return X, Y