import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)

        self.fc1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)

        x = self.fc1(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return torch.round(x * 10000) / 10000