# rotdet_model.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from rotdet_c4net import C4Net

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

class RotDetTiny(nn.Module):
    # Input: (B, 1, 128, 128)  -> change to 64 if you like
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, stride=1, padding=1), nn.LeakyReLU(inplace=True),  # 64x64
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),   nn.LeakyReLU(inplace=True),  # 32x32
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),   nn.LeakyReLU(inplace=True),  # 16x16
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2,2),
        )
        #self.head = nn.Sequential(
        #    nn.AdaptiveAvgPool2d(1),  # -> (B, 32, 1, 1)
        #    nn.Flatten(),
        #    nn.Linear(32, num_classes),
        #)
        # fully-conv classifier head
        self.head = nn.Sequential(
            nn.Conv2d(32, num_classes, 1, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.stem(x)
        return self.head(x)

class RotDetTinyBN(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, act=nn.ReLU(inplace=True)):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                act,
            )
        self.stem = nn.Sequential(
            block(in_ch, 16),   # 64x64
            block(16, 32),      # 32x32
            block(32, 32),      # 16x16
        )
        # fully-conv classifier head
        self.head = nn.Sequential(
            nn.Conv2d(32, num_classes, 1, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
    def forward(self, x):
        return self.head(self.stem(x))


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
    #model = RotDetTiny(num_classes=4)
    model = C4Net(num_classes=4,in_ch=1,stem_ch=16, widths=(32, 64, 128),head_type="equivariant")
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

