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
        n = len(x)
        d = len(x[0])

        sqrt_ = sqrt
        round_ = round
        rng = range(d)

        if training:
            sums = [0.0] * d
            sums_sq = [0.0] * d

            for row in x:
                for j in rng:
                    v = row[j]
                    sums[j] += v
                    sums_sq[j] += v * v

            mean = [0.0] * d
            var = [0.0] * d

            inv_n = 1.0 / n
            for j in rng:
                m = sums[j] * inv_n
                mean[j] = m
                var[j] = sums_sq[j] * inv_n - m * m

            new_mean = [0.0] * d
            new_var = [0.0] * d

            one_minus_momentum = 1.0 - momentum

            for j in rng:
                new_mean[j] = one_minus_momentum * running_mean[j] + momentum * mean[j]
                new_var[j] = one_minus_momentum * running_var[j] + momentum * var[j]

            denom = [sqrt_(var[j] + eps) for j in rng]

            y = []
            for row in x:
                out = [0.0] * d
                for j in rng:
                    out[j] = round_(gamma[j] * ((row[j] - mean[j]) / denom[j]) + beta[j], 4)
                y.append(out)

            return (
                y,
                [round_(v, 4) for v in new_mean],
                [round_(v, 4) for v in new_var]
            )

        else:
            denom = [sqrt_(running_var[j] + eps) for j in rng]

            y = []
            for row in x:
                out = [0.0] * d
                for j in rng:
                    out[j] = round_(gamma[j] * ((row[j] - running_mean[j]) / denom[j]) + beta[j], 4)
                y.append(out)

            return (
                y,
                [round_(v, 4) for v in running_mean],
                [round_(v, 4) for v in running_var]
            )
        pass
