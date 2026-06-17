import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        epochs: int,
        lr: float
    ) -> Tuple[NDArray[np.float64], float]:

        n_samples, n_features = X.shape

        w = np.zeros(n_features, dtype=np.float64)
        b = 0.0

        step = (2.0 * lr) / n_samples

        # Precompute reusable values
        XtX = X.T @ X
        Xty = X.T @ y
        sum_X = X.sum(axis=0)
        sum_y = float(y.sum())

        for _ in range(epochs):
            # Gradients using precomputed statistics
            dw = XtX @ w + b * sum_X - Xty
            db = sum_X @ w + n_samples * b - sum_y

            w -= step * dw
            b -= step * db

        return np.round(w, 5), round(float(b), 5)