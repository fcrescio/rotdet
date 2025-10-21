# rotdet_data.py
from __future__ import annotations
import random
from typing import Iterable, Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

# ---------- transforms / helpers ----------

def prep_for_model(pil_img: Image.Image, size: Tuple[int, int] = (128, 128)) -> torch.Tensor:
    img = pil_img.convert("L").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # 1xH*W

def maybe_rotate(pil: Image.Image, rotate_prob: float) -> Tuple[Image.Image, int]:
    if random.random() < rotate_prob:
        pil = pil.rotate(random.choice([90, 270]), expand=True)
        return pil, 1
    return pil, 0

def get_pages_from_example(
    ex: dict,
    single_image_key: str = "image",
    multi_image_key: str = "images",
    allow_paths: bool = True,
    pages_per_doc: Optional[int] = None,
) -> List[Image.Image]:
    """Extract list of PIL pages from a dataset row (supports Docmatix 'images')."""
    pages: List[Image.Image] = []

    if multi_image_key in ex and ex[multi_image_key] is not None:
        imgs = ex[multi_image_key]
        if isinstance(imgs, (list, tuple)):
            for im in imgs:
                if isinstance(im, Image.Image):
                    pages.append(im)
                elif isinstance(im, dict) and "image" in im and isinstance(im["image"], Image.Image):
                    pages.append(im["image"])
        if pages_per_doc is not None and len(pages) > pages_per_doc:
            pages = pages[:pages_per_doc]

    elif single_image_key in ex and ex[single_image_key] is not None:
        if isinstance(ex[single_image_key], Image.Image):
            pages = [ex[single_image_key]]
        elif isinstance(ex[single_image_key], dict) and "image" in ex[single_image_key]:
            maybe_img = ex[single_image_key]["image"]
            if isinstance(maybe_img, Image.Image):
                pages = [maybe_img]

    elif allow_paths and "image_path" in ex and ex["image_path"]:
        pages = [Image.open(ex["image_path"]).convert("RGB")]

    return pages

# ---------- dataset wrappers ----------

class RotDetMap(Dataset):
    """Indexable dataset; flattens multi-page rows into per-page samples."""
    def __init__(
        self,
        hf_ds,
        rotate_prob: float = 0.5,
        seed: int = 42,
        single_image_key: str = "image",
        multi_image_key: str = "images",
        pages_per_doc: Optional[int] = None,
        out_size: Tuple[int, int] = (128, 128),
    ):
        self.ds = hf_ds
        self.rotate_prob = rotate_prob
        self.single_image_key = single_image_key
        self.multi_image_key = multi_image_key
        self.pages_per_doc = pages_per_doc
        self.out_size = out_size
        random.seed(seed)

        # Build flat index: (row_idx, page_idx)
        self._index: List[Tuple[int, int]] = []
        for row_idx in range(len(self.ds)):
            ex = self.ds[row_idx]
            pages = get_pages_from_example(
                ex,
                single_image_key=self.single_image_key,
                multi_image_key=self.multi_image_key,
                allow_paths=True,
                pages_per_doc=self.pages_per_doc,
            )
            for page_idx in range(len(pages)):
                self._index.append((row_idx, page_idx))

        if not self._index:
            raise ValueError("No images found; check keys 'images', 'image', or 'image_path'.")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, flat_idx: int):
        row_idx, page_idx = self._index[flat_idx]
        ex = self.ds[row_idx]
        pages = get_pages_from_example(
            ex,
            single_image_key=self.single_image_key,
            multi_image_key=self.multi_image_key,
            allow_paths=True,
            pages_per_doc=self.pages_per_doc,
        )
        pil = pages[page_idx]
        pil, label = maybe_rotate(pil, self.rotate_prob)
        return prep_for_model(pil, self.out_size), label

class RotDetIterable(IterableDataset):
    """Streaming/iterable dataset; flattens multi-page rows to per-page samples."""
    def __init__(
        self,
        hf_stream: Iterable[dict],
        rotate_prob: float = 0.5,
        limit: Optional[int] = None,
        seed: int = 42,
        single_image_key: str = "image",
        multi_image_key: str = "images",
        pages_per_doc: Optional[int] = None,
        out_size: Tuple[int, int] = (128, 128),
    ):
        self.ds = hf_stream
        self.rotate_prob = rotate_prob
        self.limit = limit
        self.single_image_key = single_image_key
        self.multi_image_key = multi_image_key
        self.pages_per_doc = pages_per_doc
        self.out_size = out_size
        random.seed(seed)

    def __iter__(self):
        yielded = 0
        for ex in self.ds:
            pages = get_pages_from_example(
                ex,
                single_image_key=self.single_image_key,
                multi_image_key=self.multi_image_key,
                allow_paths=True,
                pages_per_doc=self.pages_per_doc,
            )
            for pil in pages:
                if self.limit is not None and yielded >= self.limit:
                    return
                pil, label = maybe_rotate(pil, self.rotate_prob)
                yield prep_for_model(pil, self.out_size), label
                yielded += 1

# ---------- public factory + dataloader ----------

def build_rotdet_dataset(
    hf_obj,
    *,
    streaming: bool = False,
    rotate_prob: float = 0.5,
    single_image_key: str = "image",
    multi_image_key: str = "images",
    pages_per_doc: Optional[int] = None,
    max_samples: Optional[int] = None,   # for map: cap rows before flatten; for iterable: per-page cap
    out_size: Tuple[int, int] = (128, 128),
    seed: int = 42,
):
    if streaming:
        return RotDetIterable(
            hf_obj,
            rotate_prob=rotate_prob,
            limit=max_samples,
            seed=seed,
            single_image_key=single_image_key,
            multi_image_key=multi_image_key,
            pages_per_doc=pages_per_doc,
            out_size=out_size,
        )
    # map-style
    ds = hf_obj
    if max_samples is not None and max_samples > 0 and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    return RotDetMap(
        ds,
        rotate_prob=rotate_prob,
        seed=seed,
        single_image_key=single_image_key,
        multi_image_key=multi_image_key,
        pages_per_doc=pages_per_doc,
        out_size=out_size,
    )

def build_rotdet_loader(
    dataset: Dataset | IterableDataset,
    *,
    batch_size: int = 64,
    num_workers: int = 2,
    device: str = "cuda",
    streaming: bool = False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not streaming,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch], dim=0),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
        ),
    )

