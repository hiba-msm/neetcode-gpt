import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x = np.asarray(x, dtype=np.float64)
        W1 = np.asarray(W1, dtype=np.float64)
        b1 = np.asarray(b1, dtype=np.float64)
        W2 = np.asarray(W2, dtype=np.float64)
        b2 = np.asarray(b2, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        # Forward pass
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)

        predictions = W2 @ a1 + b2

        # Loss
        error = predictions - y_true
        loss = np.mean(error * error)

        # Backward pass
        output_size = y_true.size

        d_pred = (2 / output_size) * error

        dW2 = np.outer(d_pred, a1)
        db2 = d_pred

        da1 = W2.T @ d_pred
        dz1 = da1 * (z1 > 0)

        dW1 = np.outer(dz1, x)
        db1 = dz1

        # Round and remove negative zeros
        dW1 = np.round(dW1, 4)
        db1 = np.round(db1, 4)
        dW2 = np.round(dW2, 4)
        db2 = np.round(db2, 4)

        dW1[dW1 == 0] = 0.0
        db1[db1 == 0] = 0.0
        dW2[dW2 == 0] = 0.0
        db2[db2 == 0] = 0.0

        return {
            "loss": round(float(loss), 4),
            "dW1": dW1.tolist(),
            "db1": db1.tolist(),
            "dW2": dW2.tolist(),
            "db2": db2.tolist()
        }
        pass
