from typing import Tuple, List
from math import sqrt



class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        rows = len(x)
        cols = len(x[0])

        def r4(v: float) -> float:
            v = round(v, 4)
            return 0.0 if v == -0.0 else v

        if training:
            batch_mean = [0.0] * cols

            for row in x:
                for j in range(cols):
                    batch_mean[j] += row[j]

            for j in range(cols):
                batch_mean[j] /= rows

            batch_var = [0.0] * cols

            for row in x:
                for j in range(cols):
                    diff = row[j] - batch_mean[j]
                    batch_var[j] += diff * diff

            for j in range(cols):
                batch_var[j] /= rows

            new_running_mean = [
                (1 - momentum) * running_mean[j] + momentum * batch_mean[j]
                for j in range(cols)
            ]

            new_running_var = [
                (1 - momentum) * running_var[j] + momentum * batch_var[j]
                for j in range(cols)
            ]

            denom = [sqrt(batch_var[j] + eps) for j in range(cols)]

            y = []
            for row in x:
                out_row = []
                for j in range(cols):
                    x_hat = (row[j] - batch_mean[j]) / denom[j]
                    out_row.append(r4(gamma[j] * x_hat + beta[j]))
                y.append(out_row)

        else:
            denom = [sqrt(running_var[j] + eps) for j in range(cols)]

            y = []
            for row in x:
                out_row = []
                for j in range(cols):
                    x_hat = (row[j] - running_mean[j]) / denom[j]
                    out_row.append(r4(gamma[j] * x_hat + beta[j]))
                y.append(out_row)

            new_running_mean = running_mean
            new_running_var = running_var

        return (
            y,
            [r4(v) for v in new_running_mean],
            [r4(v) for v in new_running_var]
        )
        pass
