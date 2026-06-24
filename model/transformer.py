import torch
import torch.nn as nn
from torchtyping import TensorType


class TransformerBlock(nn.Module):

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)

        self.attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
        self.feed_forward = self.VanillaNeuralNetwork(model_dim)
        self.layer_norm_1 = nn.LayerNorm(model_dim)
        self.layer_norm_2 = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)

        x = embedded + self.attention(self.layer_norm_1(embedded))
        x = x + self.feed_forward(self.layer_norm_2(x))

        return torch.round(x * 10000) / 10000

    class MultiHeadedSelfAttention(nn.Module):

        class SingleHeadAttention(nn.Module):
            def __init__(self, model_dim: int, head_size: int):
                super().__init__()
                torch.manual_seed(0)

                self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                self.value_gen = nn.Linear(model_dim, head_size, bias=False)

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                k = self.key_gen(embedded)
                q = self.query_gen(embedded)
                v = self.value_gen(embedded)

                scores = q @ torch.transpose(k, 1, 2)

                context_length = k.shape[1]
                attention_dim = k.shape[2]
                scores = scores / (attention_dim ** 0.5)

                lower_triangular = torch.tril(
                    torch.ones(context_length, context_length, device=embedded.device)
                )

                mask = lower_triangular == 0
                scores = scores.masked_fill(mask, float("-inf"))
                scores = nn.functional.softmax(scores, dim=2)

                return scores @ v

        def __init__(self, model_dim: int, num_heads: int):
            super().__init__()
            torch.manual_seed(0)

            self.att_heads = nn.ModuleList()

            for _ in range(num_heads):
                self.att_heads.append(
                    self.SingleHeadAttention(model_dim, model_dim // num_heads)
                )

            self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            head_outputs = []

            for head in self.att_heads:
                head_outputs.append(head(embedded))

            concatenated = torch.cat(head_outputs, dim=2)

            return self.output_proj(concatenated)

    class VanillaNeuralNetwork(nn.Module):

        def __init__(self, model_dim: int):
            super().__init__()
            torch.manual_seed(0)

            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x: TensorType[float]) -> TensorType[float]:
            torch.manual_seed(0)

            x = self.up_projection(x)
            x = self.relu(x)
            x = self.down_projection(x)
            x = self.dropout(x)

            return x