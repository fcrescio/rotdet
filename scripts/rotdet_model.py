# rotdet_model.py
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from typing import Any, Dict

from models import (
    C4Net,
    SimpleCNN,
    RotDetTiny,
    RotDetTinyBN,
    available_model_names,
    create_model,
    default_model_kwargs,
)

__all__ = [
    "SimpleCNN",
    "RotDetTiny",
    "RotDetTinyBN",
    "C4Net",
    "available_model_names",
    "create_model",
    "default_model_kwargs",
    "load_rotdet",
]


def load_rotdet(
    repo_id: str | None = "fcrescio/rotdet",
    filename: str | None = "model.safetensors",
    device: str | torch.device = "cpu",
    *,
    model: str = "c4net",
    model_kwargs: Dict[str, Any] | None = None,
) -> nn.Module:
    """
    Load rotation detector weights either from the Hub or from a local .safetensors path.
    If `repo_id` is a local directory or a file path ending with .safetensors,
    load it directly from disk.
    """
    overrides = model_kwargs or {}
    model_obj = create_model(model, **overrides)
    # direct local path
    if repo_id and str(repo_id).endswith(".safetensors"):
        path = repo_id
    elif Path(str(filename)).exists():
        path = filename
    else:
        # hub download
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    state = load_file(str(path))
    model_obj.load_state_dict(state)
    model_obj.to(device)
    return model_obj

