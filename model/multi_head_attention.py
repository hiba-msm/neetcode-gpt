import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class MultiHeadedSelfAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        self.num_heads = num_heads
        self.head_size = attention_dim // num_heads
        self.attention_dim = attention_dim

        # Required by the prompt
        self.heads = nn.ModuleList([
            self.SingleHeadAttention(embedding_dim, self.head_size)
            for _ in range(num_heads)
        ])

        self.output_projection = nn.Linear(attention_dim, attention_dim, bias=False)

        # Cache all head weights together for faster vectorized forward
        self.register_buffer(
            "k_weight",
            torch.cat([h.key_gen.weight.detach() for h in self.heads], dim=0)
        )
        self.register_buffer(
            "q_weight",
            torch.cat([h.query_gen.weight.detach() for h in self.heads], dim=0)
        )
        self.register_buffer(
            "v_weight",
            torch.cat([h.value_gen.weight.detach() for h in self.heads], dim=0)
        )

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        with torch.inference_mode():
            B, T, _ = embedded.shape
            H = self.num_heads
            D = self.head_size

            k = F.linear(embedded, self.k_weight).view(B, T, H, D).transpose(1, 2)
            q = F.linear(embedded, self.q_weight).view(B, T, H, D).transpose(1, 2)
            v = F.linear(embedded, self.v_weight).view(B, T, H, D).transpose(1, 2)

            scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)

            mask = torch.triu(
                torch.ones(T, T, device=embedded.device, dtype=torch.bool),
                diagonal=1
            )

            scores = scores.masked_fill(mask, float("-inf"))
            scores = F.softmax(scores, dim=-1)

            out = scores @ v
            out = out.transpose(1, 2).reshape(B, T, self.attention_dim)

            out = self.output_projection(out)

            return torch.round(out * 10000) / 10000

    class SingleHeadAttention(nn.Module):
        def __init__(self, embedding_dim: int, attention_dim: int):
            super().__init__()
            torch.manual_seed(0)

            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2)
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(
                torch.ones(context_length, context_length, device=embedded.device)
            )

            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float("-inf"))
            scores = F.softmax(scores, dim=2)

            return scores @ v