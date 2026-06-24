import torch
import torch.nn as nn
import torch.nn.functional as F


class Solution:
    def train(
        self,
        model: nn.Module,
        data: torch.Tensor,
        epochs: int,
        context_length: int,
        batch_size: int,
        lr: float
    ) -> float:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        model.train()
        final_loss = None

        for epoch in range(epochs):
            torch.manual_seed(epoch)

            # Random start indices
            max_start = len(data) - context_length
            starts = torch.randint(0, max_start, (batch_size,))

            # Build batch indices
            offsets = torch.arange(context_length)
            indices = starts[:, None] + offsets[None, :]

            # X and shifted Y
            X = data[indices]
            Y = data[indices + 1]

            # Forward pass
            logits = model(X)  # (B, T, vocab_size)

            B, T, C = logits.shape

            logits_flat = logits.reshape(B * T, C)
            targets_flat = Y.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            final_loss = loss

        return round(final_loss.item(), 4)