from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import RotDetModel


def rot90_k(w: torch.Tensor, k: int) -> torch.Tensor:
    """Rotate a 2D kernel by ``k`` quarter turns."""

    k %= 4
    return w if k == 0 else torch.rot90(w, k, dims=(-2, -1))


def split_orient(x: torch.Tensor) -> torch.Tensor:
    """(B, C*4, H, W) -> (B, C, 4, H, W)."""

    b, c4, h, w = x.shape
    if c4 % 4 != 0:
        raise ValueError("Number of channels must be divisible by 4.")
    c = c4 // 4
    return x.view(b, c, 4, h, w)


def merge_orient(x: torch.Tensor) -> torch.Tensor:
    """(B, C, 4, H, W) -> (B, C*4, H, W)."""

    b, c, g, h, w = x.shape
    if g != 4:
        raise ValueError("Expected 4 orientations (C4).")
    return x.view(b, c * 4, h, w)


def group_pool_c4(x: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    """Pool across orientations: (B, C*4, H, W) -> (B, C, H, W)."""

    x = split_orient(x)
    if reduce == "mean":
        x = x.mean(dim=2)
    elif reduce == "max":
        x, _ = x.max(dim=2)
    else:
        raise ValueError("reduce must be 'mean' or 'max'")
    return x


class LiftingConvC4(nn.Module):
    """Lift planar features to the C4 group by applying rotated kernels."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 7, stride: int = 2, bias: bool = True):
        super().__init__()
        pad = k // 2
        self.k = k
        self.stride = stride
        self.pad = pad
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, k, k) * (2.0 / (in_ch * k * k)) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h != w:
            raise ValueError("Input must be square (H == W).")
        ys = []
        for r in range(4):
            w_r = rot90_k(self.weight, r)
            y_r = F.conv2d(x, w_r, bias=self.bias, stride=self.stride, padding=self.pad)
            ys.append(y_r)
        return torch.cat(ys, dim=1)


class GroupConvC4(nn.Module):
    """C4 group convolution with shared, rotated kernels."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, bias: bool = True):
        super().__init__()
        pad = k // 2
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = k
        self.stride = stride
        self.pad = pad

        self.weight_base = nn.Parameter(
            torch.randn(out_ch, in_ch, 4, k, k) * (2.0 / (in_ch * k * k)) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None

    def build_full_weight(self, device: torch.device | None = None) -> torch.Tensor:
        cout, cin, k = self.out_ch, self.in_ch, self.k
        blocks = []
        for r in range(4):
            row = []
            for s in range(4):
                t = (s - r) % 4
                w_t = self.weight_base[:, :, t, :, :]
                w_rt = rot90_k(w_t, r)
                row.append(w_rt)
            row = torch.cat(row, dim=1)
            blocks.append(row)
        w_full = torch.cat(blocks, dim=0)
        return w_full.to(device) if device is not None else w_full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c4, h, w = x.shape
        if h != w:
            raise ValueError("Input must be square (H == W).")
        if c4 != self.in_ch * 4:
            raise ValueError("Channel dimension incompatible with group structure.")
        w_full = self.build_full_weight(device=x.device)
        bias = None if self.bias is None else self.bias.repeat(4)
        return F.conv2d(x, w_full, bias=bias, stride=self.stride, padding=self.pad)


class C4Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1):
        super().__init__()
        self.conv = GroupConvC4(in_ch, out_ch, k=k, stride=stride)
        self.bn = nn.BatchNorm2d(out_ch * 4)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class InvariantHeadC4(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = group_pool_c4(x, reduce="mean")
        x = self.gap(x).flatten(1)
        return self.fc(x)


class EquivariantHeadC4(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.conv1x1 = GroupConvC4(in_ch, out_ch, k=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1x1(x)


class C4Net(RotDetModel):
    """Rotation detection network based on C4 group convolutions."""

    def __init__(
        self,
        in_ch: int = 3,
        stem_ch: int = 32,
        widths: tuple[int, ...] = (64, 128, 256),
        num_classes: int = 4,
        head_type: str = "invariant",
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            LiftingConvC4(in_ch, stem_ch, k=7, stride=2),
            nn.BatchNorm2d(stem_ch * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        c = stem_ch

        stages = []
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            stages.append(C4Block(c, w, k=3, stride=stride))
            c = w
        self.backbone = nn.Sequential(*stages)

        if head_type == "invariant":
            self.head: nn.Module = InvariantHeadC4(c, num_classes)
        elif head_type == "equivariant":
            self.head = EquivariantHeadC4(c, out_ch=num_classes)
        else:
            raise ValueError("head_type must be 'invariant' or 'equivariant'")
        self.head_type = head_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b, c, h, w = x.shape
        if h != w:
            raise ValueError(f"Expected square input, got {h}x{w}.")
        x = self.stem(x)
        x = self.backbone(x)
        return self.head(x)

    def compute_loss(self, y_pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.head_type == "invariant":
            return F.cross_entropy(y_pred, labels)

        if y_pred.dim() != 4:
            raise ValueError("Equivariant head expects 4D predictions.")
        b, c4, h, w = y_pred.shape
        if c4 % 4 != 0:
            raise ValueError(f"Channel dimension {c4} is not divisible by 4.")
        c = c4 // 4

        if c == 1:
            logits = F.adaptive_avg_pool2d(y_pred, 1).reshape(b, 4)
            return F.cross_entropy(logits, labels)

        y = y_pred.view(b, c, 4, h, w).mean(dim=2)
        logits = F.adaptive_avg_pool2d(y, 1).reshape(b, c)
        return F.cross_entropy(logits, labels)

    def compute_logits(self, y_pred: torch.Tensor) -> torch.Tensor:
        b, c4, h, w = y_pred.shape
        c = c4 // 4
        y = y_pred.view(b, c, 4, h, w).mean(dim=2)
        logits = F.adaptive_avg_pool2d(y, 1).reshape(b, c)
        return logits


__all__ = [
    "C4Net",
    "group_pool_c4",
    "merge_orient",
    "split_orient",
    "LiftingConvC4",
    "GroupConvC4",
    "C4Block",
    "InvariantHeadC4",
    "EquivariantHeadC4",
]
