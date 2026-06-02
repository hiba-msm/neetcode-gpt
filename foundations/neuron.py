import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # Pre-activation
        z = np.dot(x, w) + b

        # Activation
        if activation == "sigmoid":
            output = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            output = max(0, z)
        else:
            raise ValueError("activation must be 'sigmoid' or 'relu'")

        return round(float(output), 5)
        pass
