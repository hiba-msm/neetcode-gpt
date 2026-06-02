import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = float(np.dot(x, w)) + b

        if activation == "relu":
            return round(z if z > 0.0 else 0.0, 5)

        return round(1.0 / (1.0 + math.exp(-z)), 5)
        pass
