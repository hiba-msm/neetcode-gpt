import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List


class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        sentences = positive + negative

        # 1. Build sorted vocabulary
        words = set()
        for sentence in sentences:
            words.update(sentence.split())

        vocab = {word: i + 1 for i, word in enumerate(sorted(words))}

        # 2. Encode each sentence as tensor of word IDs
        encoded = []
        for sentence in sentences:
            ids = [vocab[word] for word in sentence.split()]
            encoded.append(torch.tensor(ids, dtype=torch.float32))

        # 3. Pad shorter sequences with 0s
        return nn.utils.rnn.pad_sequence(encoded, batch_first=True)