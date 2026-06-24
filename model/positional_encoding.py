import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        positions = np.arange(seq_len)[:, np.newaxis]      # shape: (seq_len, 1)
        dims = np.arange(0, d_model, 2)                    # even dimensions: 0, 2, 4, ...

        div_term = np.power(10000, dims / d_model)         # shape: (ceil(d_model/2),)
        angles = positions / div_term                      # broadcasting

        pe = np.zeros((seq_len, d_model), dtype=np.float64)

        pe[:, 0::2] = np.sin(angles)
        pe[:, 1::2] = np.cos(angles[:, :pe[:, 1::2].shape[1]])

        return np.round(pe, 5)