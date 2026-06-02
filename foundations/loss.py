import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        loss = -np.sum(
            y_true * np.log(y_pred + epsilon) +
            (1 - y_true) * np.log(1 - y_pred + epsilon)
        ) / y_true.size

        return round(float(loss), 4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:

        epsilon = 1e-7

        loss = -np.sum(
            y_true * np.log(y_pred + epsilon)
        ) / y_true.shape[0]

        return round(float(loss), 4)
        pass
