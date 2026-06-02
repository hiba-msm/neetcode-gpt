import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        a = x

        last_layer = len(weights) - 1

        for i, (W, b) in enumerate(zip(weights, biases)):
            # Linear layer
            a = a @ W + b

            # ReLU for hidden layers only
            if i != last_layer:
                a = np.maximum(0, a)

        return np.round(a, 5)
        pass
