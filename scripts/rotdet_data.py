# rotdet_data.py
from __future__ import annotations
import random
from typing import Iterable, Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tqdm.auto import tqdm

# ---------- transforms / helpers ----------

def prep_for_model(pil_img: Image.Image, size: Tuple[int, int] = (128, 128)) -> torch.Tensor:
    img = pil_img.convert("L").resize(size)
    arr = np.array(img, dtype=np.int8)
    return torch.from_numpy(arr).unsqueeze(0)  # 1xH*W

def rotate_on_device(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float().div_(255)
    for k in (1,2,3):
        mask = (y == k)
        if mask.any():
            x[mask] = torch.rot90(x[mask], k=k, dims=(2,3))
    return x

def maybe_rotate(pil: Image.Image, rotate_prob: float) -> Tuple[Image.Image, int]:
    if random.random() < rotate_prob:
        rotation = random.choice([1,2,3])
        #pil = pil.rotate(90*rotation, expand=True)
        return pil, rotation
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

def meta_from_example(ex, row_idx=None, page_idx=None):
    # Try common id/url/path fields if they exist
    meta = {
        "row_idx": row_idx,
        "page_idx": page_idx,
    }
    # Common keys we might see on HF datasets
    for k in ("id", "doc_id", "document_id", "file_name", "image_path", "url", "image_url", "pdf_url"):
        if k in ex and ex[k] is not None:
            meta[k] = ex[k]
    return meta

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
        debug_pil: Optional[bool] = False,
    ):
        self.ds = hf_ds
        self.rotate_prob = rotate_prob
        self.single_image_key = single_image_key
        self.multi_image_key = multi_image_key
        self.pages_per_doc = pages_per_doc
        self.out_size = out_size
        self.debug_pil = debug_pil
        random.seed(seed)

        bar = tqdm(total=len(self.ds), desc="Building rotdet dataset")

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
            bar.update(1)
        bar.close()

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
        meta = meta_from_example(ex, row_idx=row_idx, page_idx=page_idx)
        return prep_for_model(pil, self.out_size), label, meta, (pil if self.debug_pil else None)

class RotDetIterable(IterableDataset):
    """Streaming/iterable dataset; flattens multi-image rows to per-page samples."""
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
        skip_pages: int = 0,               # NEW: skip initial N page-samples
    ):
        self.ds = hf_stream
        self.rotate_prob = rotate_prob
        self.limit = limit
        self.single_image_key = single_image_key
        self.multi_image_key = multi_image_key
        self.pages_per_doc = pages_per_doc
        self.out_size = out_size
        self.skip_pages = skip_pages
        random.seed(seed)

    def __iter__(self):
        yielded = 0
        skipped = 0
        row_idx = -1
        for ex in self.ds:
            row_idx += 1
            pages = get_pages_from_example(
                ex,
                single_image_key=self.single_image_key,
                multi_image_key=self.multi_image_key,
                allow_paths=True,
                pages_per_doc=self.pages_per_doc,
            )
            for page_idx, pil in enumerate(pages):
                # Skip the first N page-samples across the stream
                if skipped < self.skip_pages:
                    skipped += 1
                    continue
                if self.limit is not None and yielded >= self.limit:
                    return
                pil2, label = maybe_rotate(pil, self.rotate_prob)
                meta = meta_from_example(ex, row_idx=row_idx, page_idx=page_idx)
                # IMPORTANT: always yield 4-tuple (x, y, meta, pil) to match collate()
                yield prep_for_model(pil2, self.out_size), label, meta, pil2
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
    def collate(batch):
        xs   = torch.stack([b[0] for b in batch], dim=0)
        ys   = torch.tensor([b[1] for b in batch], dtype=torch.long)
        metas= [b[2] for b in batch]
        pils = [b[3] for b in batch]  # PILs (already rotated) if you want to save
        return xs, ys, metas, pils
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not streaming,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=True,        
        prefetch_factor=4,           
        collate_fn=collate,
    )


# --- Splitting helpers (map + streaming) ---

import itertools
from typing import Tuple, Optional

def build_rotdet_dataset_pair(
    hf_obj,
    hf_obj_train,
    *,
    streaming: bool,
    rotate_prob: float = 0.5,
    single_image_key: str = "image",
    multi_image_key: str = "images",
    pages_per_doc: Optional[int] = None,
    out_size: Tuple[int, int] = (128, 128),
    seed: int = 42,
    # choose one of the following to size validation:
    val_fraction: Optional[float] = None,  # e.g. 0.1
    val_pages: Optional[int] = None,       # exact number of page-samples for val (streaming-friendly)
    max_train_pages: Optional[int] = None, # optional cap on training pages
) -> Tuple[Dataset | IterableDataset, Dataset | IterableDataset]:
    """
    Returns (train_set, val_set) with per-page samples.

    Map-style:
      - If val_fraction set -> split HF rows with train_test_split then wrap.
      - If val_pages set    -> select first val_pages rows before flatten (quick approximation).
    Streaming:
      - Take first `val_pages` page-samples for val, then continue stream for train.
      - If only val_fraction given -> raise (not supported without knowing length).
    """

    if not streaming:
        ds = hf_obj
        if val_fraction is not None:
            # split at row level, then flatten pages in the wrappers
            split = ds.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
            train_rows, val_rows = split["train"], split["test"]
        elif val_pages is not None:
            # approximate: pick that many rows for val (before flatten)
            n_val_rows = min(len(ds), val_pages)
            val_rows = ds.select(range(n_val_rows))
            train_rows = ds.select(range(n_val_rows, len(ds)))
        else:
            raise ValueError("Provide either val_fraction (map-style) or val_pages.")

        train_set = RotDetMap(
            train_rows, rotate_prob=rotate_prob, seed=seed,
            single_image_key=single_image_key, multi_image_key=multi_image_key,
            pages_per_doc=pages_per_doc, out_size=out_size
        )
        val_set = RotDetMap(
            val_rows, rotate_prob=rotate_prob, seed=seed,
            single_image_key=single_image_key, multi_image_key=multi_image_key,
            pages_per_doc=pages_per_doc, out_size=out_size
        )

        # Optional: cap training pages after flattening
        if max_train_pages is not None and hasattr(train_set, "_index") and len(train_set) > max_train_pages:
            # Thin the flat index to first N
            train_set._index = train_set._index[:max_train_pages]
        return train_set, val_set

    # --- streaming case ---
    if val_pages is None and val_fraction is not None:
        raise ValueError("For streaming datasets, specify val_pages (exact number).")

    # validation stream (exact first val_pages)
    val_set = RotDetIterable(
        hf_obj,                              # <- this stream feeds validation
        rotate_prob=rotate_prob,
        limit=val_pages,
        seed=seed,
        single_image_key=single_image_key,
        multi_image_key=multi_image_key,
        pages_per_doc=pages_per_doc,
        out_size=out_size,
        skip_pages=0,                        # take from the start
    )

    # training needs a FRESH, independent stream:
    # caller should pass another load_dataset(..., streaming=True) for train
    train_set = RotDetIterable(
        hf_obj_train,                        # <- pass a fresh stream here (see note below)
        rotate_prob=rotate_prob,
        limit=max_train_pages,
        seed=seed,
        single_image_key=single_image_key,
        multi_image_key=multi_image_key,
        pages_per_doc=pages_per_doc,
        out_size=out_size,
        skip_pages=(val_pages or 0), 
    )

    return train_set, val_set

