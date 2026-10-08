"""CPU inference for the frozen RotDet v2 C4Net checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from .c4net import C4Net


class Detector:
    """Load a local variant directory containing config.json and model.safetensors."""

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        meta = json.loads((directory / "config.json").read_text())
        self.resolution = meta["resolution"]
        if self.resolution not in (256, 384):
            raise ValueError("Supported resolutions: 256, 384")
        weights = directory / "model.safetensors"
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != meta["sha256"]:
            raise ValueError("Checkpoint SHA-256 mismatch")
        cfg = dict(meta["architecture"])
        cfg["widths"] = tuple(cfg["widths"])
        self.model = C4Net(**cfg)
        self.model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
        self.model.eval()

    @classmethod
    def from_pretrained(cls, resolution: int = 256, *,
                        repo_id: str = "fcrescio/rotdet-v2",
                        revision: str):
        """Download one variant at an explicit Hub revision, then verify its checksum."""
        if resolution not in (256, 384):
            raise ValueError("Supported resolutions: 256, 384")
        from huggingface_hub import snapshot_download
        root = snapshot_download(repo_id, revision=revision,
                                 allow_patterns=[f"{resolution}/*"])
        return cls(Path(root) / str(resolution))

    def tensor(self, data: bytes, k: int = 0) -> torch.Tensor:
        with Image.open(io.BytesIO(data)) as page:
            gray = page.convert("RGB").convert("L").resize(
                (self.resolution, self.resolution), Image.Resampling.BICUBIC)
            pixels = np.asarray(gray, dtype=np.uint8)
        pixels = np.rot90(pixels, k % 4).copy(order="C")
        return torch.from_numpy(pixels).float()[None, None] / 255.0

    def predict(self, data: bytes, k: int = 0) -> dict:
        """Class k means k*90 degrees CCW; apply k*90 degrees CW to correct."""
        with torch.inference_mode():
            logits = self.model(self.tensor(data, k))
            probabilities = logits.softmax(1)[0]
        prediction = int(probabilities.argmax())
        return {"class": prediction,
                "rotation_ccw_degrees": prediction * 90,
                "correction_cw_degrees": prediction * 90,
                "confidence": float(probabilities[prediction]),
                "probabilities": probabilities.tolist()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--model", required=True, help="Local variant directory")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    detector = Detector(args.model)
    failed = False
    for filename in args.files:
        try:
            result = {"file": filename, **detector.predict(Path(filename).read_bytes())}
        except (OSError, ValueError) as error:
            result = {"file": filename, "error": str(error)}
            failed = True
        print(json.dumps(result), flush=True)
    return int(failed)
