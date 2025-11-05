from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotDetModel(nn.Module):
    """Common interface for rotation detection models."""

    def compute_logits(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Return classification logits from the raw forward output."""
        return y_pred

    def compute_loss(self, y_pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return the loss value for the given predictions and labels."""
        logits = self.compute_logits(y_pred)
        return F.cross_entropy(logits, labels)
