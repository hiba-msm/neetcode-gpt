import numpy as np
from typing import List
from math import sqrt


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        n = len(x)

        mean_square = 0.0
        for v in x:
            mean_square += v * v

        rms = sqrt(mean_square / n + eps)

        res = []
        for i in range(n):
            val = round((x[i] / rms) * gamma[i], 4)
            res.append(0.0 if val == -0.0 else val)

        return res