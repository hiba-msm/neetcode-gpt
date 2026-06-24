import torch
import torch.nn as nn
from torchtyping import TensorType
import math


class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        # Instantiation order matters: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

        self.attention_dim = attention_dim

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # embedded shape: (B, T, embedding_dim)

        K = self.key(embedded)      # (B, T, attention_dim)
        Q = self.query(embedded)    # (B, T, attention_dim)
        V = self.value(embedded)    # (B, T, attention_dim)

        scores = Q @ K.transpose(1, 2)
        scores = scores / math.sqrt(self.attention_dim)

        T = embedded.shape[1]
        mask = torch.tril(torch.ones(T, T, device=embedded.device))

        scores = scores.masked_fill(mask == 0, float("-inf"))

        scores = torch.softmax(scores, dim=2)

        out = scores @ V

        return torch.round(out * 10000) / 10000