# evaluate_rotdet.py
# Minimal evaluation for fcrescio/rotdet on a Hugging Face document dataset
# Creates a synthetic eval set: half pages rotated by 90°/270° ("Rotated"=1), half unchanged ("Normal"=0)

import argparse
import random
from pathlib import Path

import itertools

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def _prep_for_model(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("L").resize((128, 128))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # 1xHxW

def _maybe_rotate(pil: Image.Image, rotate_prob: float) -> tuple[Image.Image, int]:
    if random.random() < rotate_prob:
        pil = pil.rotate(random.choice([90, 270]), expand=True)
        return pil, 1
    return pil, 0

# ---- Model (from discussion: define SimpleCNN and load weights) ----
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ---- Dataset wrapper: builds a synthetic Rotated/Normal evaluation set ----

def _get_pages_from_example(
    ex,
    single_image_key="image",
    multi_image_key="images",
    allow_paths=True,
    pages_per_doc=None,
):
    """
    Returns a list[ PIL.Image ] from a dataset example.
    Handles:
      - ex["image"] (single PIL.Image)
      - ex["images"] (list of PIL.Images)
      - ex["image_path"] (string path) if allow_paths=True
    """
    pages = []

    # Multi-page first (Docmatix)
    if multi_image_key in ex and ex[multi_image_key] is not None:
        imgs = ex[multi_image_key]
        # imgs can be: list[PIL.Image] or list[dict] depending on dataset processing
        if isinstance(imgs, (list, tuple)):
            for im in imgs:
                if isinstance(im, Image.Image):
                    pages.append(im)
                elif isinstance(im, dict) and "image" in im and isinstance(im["image"], Image.Image):
                    pages.append(im["image"])
        # cap pages per doc if requested
        if pages_per_doc is not None and len(pages) > pages_per_doc:
            pages = pages[:pages_per_doc]

    # Fallback to single image
    elif single_image_key in ex and ex[single_image_key] is not None:
        if isinstance(ex[single_image_key], Image.Image):
            pages = [ex[single_image_key]]
        elif isinstance(ex[single_image_key], dict) and "image" in ex[single_image_key]:
            maybe_img = ex[single_image_key]["image"]
            if isinstance(maybe_img, Image.Image):
                pages = [maybe_img]

    # Or an image path (rare in HF hub datasets but we keep support)
    elif allow_paths and "image_path" in ex and ex["image_path"]:
        pages = [Image.open(ex["image_path"]).convert("RGB")]

    return pages

class RotDetMap(Dataset):
    """Map-style dataset wrapper that flattens multi-image rows into per-page samples."""
    def __init__(self, hf_ds, rotate_prob=0.5, seed=42,
                 single_image_key="image", multi_image_key="images", pages_per_doc=None):
        self.ds = hf_ds
        self.rotate_prob = rotate_prob
        self.single_image_key = single_image_key
        self.multi_image_key = multi_image_key
        self.pages_per_doc = pages_per_doc
        random.seed(seed)

        # Build an index mapping: flat_idx -> (row_idx, page_idx)
        self._index = []
        for row_idx in range(len(self.ds)):
            ex = self.ds[row_idx]
            pages = _get_pages_from_example(
                ex,
                single_image_key=self.single_image_key,
                multi_image_key=self.multi_image_key,
                allow_paths=True,
                pages_per_doc=self.pages_per_doc,
            )
            for page_idx in range(len(pages)):
                self._index.append((row_idx, page_idx))

        if not self._index:
            raise ValueError("No images found. Check your keys: 'images', 'image', or 'image_path'.")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, flat_idx):
        row_idx, page_idx = self._index[flat_idx]
        ex = self.ds[row_idx]
        pages = _get_pages_from_example(
            ex,
            single_image_key=self.single_image_key,
            multi_image_key=self.multi_image_key,
            allow_paths=True,
            pages_per_doc=self.pages_per_doc,
        )
        pil = pages[page_idx]
        pil, label = _maybe_rotate(pil, self.rotate_prob)
        return _prep_for_model(pil), label

class RotDetIterable(IterableDataset):
    """Iterable wrapper for HF streaming datasets."""
    def __init__(self, hf_stream, single_image_key="image", multi_image_key="images", rotate_prob=0.5, limit=None, seed=42):
        self.ds = hf_stream
        self.rotate_prob = rotate_prob
        self.single_image_key = single_image_key
        self.multi_image_key = multi_image_key
        self.pages_per_doc = 1
        self.limit = limit
        random.seed(seed)

    def __iter__(self):
        yielded = 0
        for ex in self.ds:
            pages = _get_pages_from_example(
                ex,
                single_image_key=self.single_image_key,
                multi_image_key=self.multi_image_key,
                allow_paths=True,
                pages_per_doc=self.pages_per_doc,
            )
            for pil in pages:
                if self.limit is not None and yielded >= self.limit:
                    return
                pil, label = _maybe_rotate(pil, self.rotate_prob)
                yield _prep_for_model(pil), label
                yielded += 1


def collate(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    # Confusion matrix: rows=true, cols=pred, order=[Normal(0), Rotated(1)]
    cm = np.zeros((2, 2), dtype=int)
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", ncols=80):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for t, p in zip(y.tolist(), pred.tolist()):
                cm[t, p] += 1
    acc = correct / total if total else 0.0
    return acc, cm


def main():
    parser = argparse.ArgumentParser(description="Evaluate fcrescio/rotdet on a document dataset")
    parser.add_argument("--repo_id", default="fcrescio/rotdet", help="Model repo on the Hub")
    parser.add_argument("--filename", default="model.safetensors", help="Weights filename in the repo")
    parser.add_argument("--dataset", default="nielsr/funsd", help="Document dataset repo id")
    parser.add_argument("--split", default="test", help="Dataset split (e.g., train/test/validation)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--rotate_prob", type=float, default=0.5, help="Probability to rotate an image")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional cap on number of samples (0 = all)")
    parser.add_argument("--dataset_config", type=str, help="Dataset config selection")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load weights from the Hub
    weights_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename)
    model = SimpleCNN()
    state = load_file(weights_path)
    model.load_state_dict(state)
    model.to(device)

    # 2) Load dataset split
    if args.dataset_config:
        streamed_ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
        eval_set = RotDetIterable(streamed_ds, limit=args.max_samples, rotate_prob=args.rotate_prob, seed=42)
    else:
        ds = load_dataset(args.dataset, split=args.split)
        if args.max_samples and args.max_samples > 0:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        eval_set = RotDetMap(ds, rotate_prob=args.rotate_prob, seed=42)
    # 3) Wrap in synthetic Rotated/Normal set + DataLoader
    loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=(device == "cuda"),
                        collate_fn=collate)

    # 4) Evaluate
    acc, cm = evaluate(model, loader, device)

    print("\n=== Results ===")
    #print(f"Samples: {len(eval_set)}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred) [Normal, Rotated]:")
    print(cm)


if __name__ == "__main__":
    main()

