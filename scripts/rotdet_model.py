# rotdet_model.py
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from models import SimpleCNN, RotDetTiny, RotDetTinyBN
from rotdet_c4net import C4Net

__all__ = ["SimpleCNN", "RotDetTiny", "RotDetTinyBN", "load_rotdet"]


def load_rotdet(
    repo_id: str | None = "fcrescio/rotdet",
    filename: str | None = "model.safetensors",
    device: str | torch.device = "cpu",
) -> nn.Module:
    """
    Load rotation detector weights either from the Hub or from a local .safetensors path.
    If `repo_id` is a local directory or a file path ending with .safetensors,
    load it directly from disk.
    """
    #model = RotDetTiny(num_classes=4)
    model = C4Net(
        num_classes=4,
        in_ch=1,
        stem_ch=16,
        widths=(32, 64, 128),
        head_type="equivariant",
    )
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

