import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:

        z = float(x.dot(w) + b)

        y_hat = 1.0 / (1.0 + math.exp(-z))

        dL_dz = (y_hat - y_true) * y_hat * (1.0 - y_hat)

        dL_dw = np.empty_like(x)
        np.multiply(x, dL_dz, out=dL_dw)
        np.round(dL_dw, 5, out=dL_dw)

        dL_db = round(dL_dz, 5)

        return dL_dw, dL_db
        pass
