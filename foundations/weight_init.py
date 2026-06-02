import torch
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = math.sqrt(2.0 / (fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std

        return torch.round(weights * 10000).div(10000).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = math.sqrt(2.0 / fan_in)
        weights = torch.randn(fan_out, fan_in) * std

        return torch.round(weights * 10000).div(10000).tolist()

    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str
    ) -> List[float]:

        torch.manual_seed(0)

        weights = []
        current_dim = input_dim

        for _ in range(num_layers):
            if init_type == "xavier":
                std = math.sqrt(2.0 / (current_dim + hidden_dim))
                W = torch.randn(hidden_dim, current_dim) * std

            elif init_type == "kaiming":
                std = math.sqrt(2.0 / current_dim)
                W = torch.randn(hidden_dim, current_dim) * std

            else:  # random
                W = torch.randn(hidden_dim, current_dim)

            weights.append(W)
            current_dim = hidden_dim

        x = torch.randn(input_dim)

        stds = []

        for W in weights:
            x = W @ x
            x = torch.relu(x)
            stds.append(round(float(x.std()), 2))

        return stds