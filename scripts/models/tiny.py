from __future__ import annotations

import torch

from .base import RotDetModel


class RotDetTiny(RotDetModel):
    """Compact convolutional network for rotation detection."""

    def __init__(self, in_ch: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 16, 3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(16),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(16, 32, 3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 32, 3, stride=1, padding=1),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2, 2),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(32, num_classes, 1, bias=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.stem(x)
        return self.head(x)


class RotDetTinyBN(RotDetModel):
    """Variant of RotDetTiny with stride-2 blocks."""

    def __init__(self, in_ch: int = 1, num_classes: int = 2, act: torch.nn.Module | None = None) -> None:
        super().__init__()
        activation = act if act is not None else torch.nn.ReLU(inplace=True)

        def block(cin: int, cout: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(cout),
                activation,
            )

        self.stem = torch.nn.Sequential(
            block(in_ch, 16),
            block(16, 32),
            block(32, 32),
        )
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(32, num_classes, 1, bias=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.head(self.stem(x))
