import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fractions = []
        hooks = []

        def hook(module, inp, out):
            out = out.detach()

            # Shape: (batch_size, neurons)
            out_2d = out.reshape(out.shape[0], -1)

            # Dead neuron = outputs 0 for ALL samples in batch
            dead = (out_2d == 0).all(dim=0).float().mean()

            dead_fractions.append(round(float(dead.item()), 4))

        for layer in model.modules():
            if isinstance(layer, nn.ReLU):
                hooks.append(layer.register_forward_hook(hook))

        try:
            with torch.no_grad():
                model(x)
        finally:
            for h in hooks:
                h.remove()

        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        if not dead_fractions:
            return "healthy"

        if any(f > 0.5 for f in dead_fractions):
            return "use_leaky_relu"

        if dead_fractions[0] > 0.3:
            return "reinitialize"

        strictly_increasing = all(
            dead_fractions[i] < dead_fractions[i + 1]
            for i in range(len(dead_fractions) - 1)
        )

        if strictly_increasing and dead_fractions[-1] > 0.1:
            return "reduce_learning_rate"

        if max(dead_fractions) < 0.1:
            return "healthy"

        return "healthy"