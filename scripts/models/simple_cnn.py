from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import RotDetModel


class SimpleCNN(RotDetModel):
    """Minimal convolutional classifier for rotation detection."""

    def __init__(self, in_ch: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, 1, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 16 * 16, 32)
        self.fc2 = torch.nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
