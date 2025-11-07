from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotDetModel(nn.Module):
    """Common interface for rotation detection models.

    Subclasses must implement :meth:`forward` and return classification logits
    of shape ``(batch, 4)``. The default :meth:`compute_loss` consumes those
    logits directly, ensuring training and evaluation code can treat every
    implementation uniformly.
    """

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute the cross-entropy loss for the provided logits and labels."""

        return F.cross_entropy(logits, labels)
