import torch
import torch.nn as nn
from torchtyping import TensorType


class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        self.repeat_factor = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, _ = x.shape
        H = self.num_heads
        G = self.num_kv_heads
        R = self.repeat_factor
        Hd = self.head_dim

        q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, G, Hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, G, Hd).transpose(1, 2)

        # Group Q heads: (B, H, T, Hd) -> (B, G, R, T, Hd)
        q = q.view(B, G, R, T, Hd)

        # K/V become broadcastable: (B, G, 1, T, Hd)
        k = k.unsqueeze(2)
        v = v.unsqueeze(2)

        scores = q @ k.transpose(-2, -1)
        scores.mul_(self.scale)

        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        scores.masked_fill_(mask, float("-inf"))

        att = torch.softmax(scores, dim=-1)
        out = att @ v

        # Back to (B, T, H * Hd)
        out = out.reshape(B, H, T, Hd).transpose(1, 2).reshape(B, T, H * Hd)

        out = self.output_proj(out)

        return torch.round(out * 10000) / 10000