import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType


class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads
        self.repeat_factor = num_heads // num_kv_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, _ = x.shape
        H = self.num_heads
        G = self.num_kv_heads
        Hd = self.head_dim

        q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, G, Hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, G, Hd).transpose(1, 2)

        # Expand KV heads to match Q heads
        if G != H:
            k = k.repeat_interleave(self.repeat_factor, dim=1)
            v = v.repeat_interleave(self.repeat_factor, dim=1)

        # Faster built-in causal attention
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )

        out = out.transpose(1, 2).contiguous().view(B, T, H * Hd)
        out = self.output_proj(out)

        return torch.round(out * 10000) / 10000