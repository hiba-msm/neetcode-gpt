import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        correct_probs = np.where(y_true == 1, y_pred, 1 - y_pred)

        loss = -np.mean(np.log(correct_probs))

        return round(float(loss), 4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:

        epsilon = 1e-7

        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        true_class_indices = np.argmax(y_true, axis=1)

        correct_probs = y_pred[np.arange(y_true.shape[0]), true_class_indices]

        loss = -np.mean(np.log(correct_probs))
        return round(float(loss), 4)

        pass
