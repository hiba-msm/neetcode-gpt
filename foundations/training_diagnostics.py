import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(
        self,
        model: nn.Module,
        x: torch.Tensor
    ) -> List[Dict[str, float]]:

        stats = []
        hooks = []

        def hook(module, inp, out):
            out = out.detach()

            # Flatten batch dimensions, keep neuron/features dimension
            out_2d = out.reshape(-1, out.shape[-1])

            # A neuron is dead if it is <= 0 for every sample
            dead = (out_2d <= 0).all(dim=0).float().mean()

            stats.append({
                "mean": round(float(out.mean().item()), 4),
                "std": round(float(out.std().item()), 4),
                "dead_fraction": round(float(dead.item()), 4)
            })

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(hook))

        with torch.no_grad():
            model(x)

        for h in hooks:
            h.remove()

        return stats

    def compute_gradient_stats(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> List[Dict[str, float]]:

        model.zero_grad()

        pred = model(x)
        loss = nn.MSELoss()(pred, y)
        loss.backward()

        stats = []

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad.detach()

                stats.append({
                    "mean": round(float(grad.mean().item()), 4),
                    "std": round(float(grad.std().item()), 4),
                    "norm": round(float(grad.norm().item()), 4)
                })

        return stats

    def diagnose(
        self,
        activation_stats: List[Dict[str, float]],
        gradient_stats: List[Dict[str, float]]
    ) -> str:

        for s in activation_stats:
            if s["dead_fraction"] > 0.5:
                return "dead_neurons"

        for s in gradient_stats:
            if s["norm"] > 1000:
                return "exploding_gradients"

        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        for s in activation_stats:
            if s["std"] < 0.1:
                return "vanishing_gradients"

        for s in activation_stats:
            if s["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"