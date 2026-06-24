import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)

        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # x shape: (B, T)
        embedded = self.embedding(x)          # (B, T, 16)

        averaged = embedded.mean(dim=1)       # (B, 16)

        out = self.linear(averaged)           # (B, 1)
        out = self.sigmoid(out)               # (B, 1)

        return torch.round(out * 10000) / 10000