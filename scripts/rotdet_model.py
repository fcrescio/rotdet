# rotdet_model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

__all__ = ["SimpleCNN", "load_rotdet"]

class SimpleCNN(nn.Module):
    """
    Minimal classifier used by fcrescio/rotdet.
    Input:  (B, 1, 128, 128)
    Output: (B, 2) -> [Normal, Rotated]
    """
    def __init__(self, in_ch: int = 1, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def load_rotdet(
    repo_id: str | None = "fcrescio/rotdet",
    filename: str | None = "model.safetensors",
    device: str | torch.device = "cpu",
) -> nn.Module:
    """
    Load SimpleCNN weights either from the Hub or from a local .safetensors path.
    If `repo_id` is a local directory or a file path ending with .safetensors,
    load it directly from disk.
    """
    model = SimpleCNN()
    # direct local path
    if repo_id and str(repo_id).endswith(".safetensors"):
        path = repo_id
    elif Path(str(filename)).exists():
        path = filename
    else:
        # hub download
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    state = load_file(str(path))
    model.load_state_dict(state)
    model.to(device)
    return model

