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
        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            foreach=False
        )

        device = data.device
        max_start = data.shape[0] - context_length

        offsets = torch.arange(context_length, device=device).unsqueeze(0)

        final_loss = None

        for epoch in range(epochs):
            torch.manual_seed(epoch)

            starts = torch.randint(
                0,
                max_start,
                (batch_size, 1),
                device=device
            )

            idx = starts + offsets

            X = data[idx]
            Y = data[idx + 1]

            logits = model(X)

            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                Y.reshape(-1)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            final_loss = loss

        return round(final_loss.item(), 4)