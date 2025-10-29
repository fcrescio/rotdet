#!/usr/bin/env python
"""
Stream → deterministic local snapshot (map-style) for reuse across runs.

Examples:
  uv run scripts/snapshot_stream.py \
    --hf-path docmatix/docmatix \
    --split train \
    --out data/snapshots/rotdet_docmatix_s0_v5 \
    --seed 0 --val-fraction 0.05 \
    --max-train-pages 20000 --max-val-pages 1000 \
    --pages-per-doc 4 \
    --doc-id-keys doc_id document_id id \
    --image-key images \
    --format jpeg --jpeg-quality 90
"""

from __future__ import annotations
import argparse, hashlib, json, random, shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image as PILImage

from datasets import (
    load_dataset, Features, Value, Image, Dataset, DatasetDict
)
from datasets.arrow_writer import ArrowWriter


def _hash01(text: str, seed: int = 0) -> float:
    h = int(hashlib.sha1(f"{text}:{seed}".encode()).hexdigest(), 16)
    return h / float(1 << 160)


def _choose_doc_id(row: dict, keys: List[str]) -> str:
    for k in keys:
        v = row.get(k)
        if v is not None:
            return str(v)
    # fallback: stable hash of (sorted) row metadata (not images)
    meta = {k: v for k, v in row.items() if k not in ("image", "images")}
    return hashlib.sha1(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]


def _select_pages(n: int, k: int | None, rng: random.Random) -> List[int]:
    if k is None or k >= n:
        return list(range(n))
    idxs = list(range(n))
    rng.shuffle(idxs)
    return idxs[:k]


def snapshot_stream(
    hf_path: str,
    split: str,
    out_dir: Path,
    seed: int,
    val_fraction: float,
    max_train_pages: int | None,
    max_val_pages: int | None,
    pages_per_doc: int | None,
    doc_id_keys: List[str],
    image_key: str,
    image_format: str,
    jpeg_quality: int,
    overwrite: bool,
    dry_run: bool,
):
    out_dir = Path(out_dir)

    if out_dir.exists():
        if not overwrite:
            raise SystemExit(f"[error] {out_dir} already exists. Use --overwrite to replace.")
        shutil.rmtree(out_dir)

    if dry_run:
        print("[dry-run] Will not write anything to disk.")

    img_train = out_dir / "train_imgs"
    img_val   = out_dir / "val_imgs"
    if not dry_run:
        img_train.mkdir(parents=True, exist_ok=True)
        img_val.mkdir(parents=True, exist_ok=True)

    # map-style Features for the snapshot
    feats = Features({
        "doc_id": Value("string"),
        "page_index": Value("int32"),
        "image": Image(),   # stores file path in Arrow; decodes on read
        # Add task-specific labels here if you wish (angle, cls, etc.)
    })

    def _writer(path: Path):
        return ArrowWriter(features=feats, path=str(path))

    train_writer = _writer(out_dir / "train.arrow") if not dry_run else None
    val_writer   = _writer(out_dir / "val.arrow")   if not dry_run else None

    rng = random.Random(seed)
    train_pages = val_pages = 0

    # streaming iterator (no full download)
    stream = load_dataset(hf_path, split=split, streaming=True)

    fmt = image_format.lower()
    if fmt not in ("png", "jpeg", "jpg"):
        raise SystemExit("--format must be 'png' or 'jpeg'/'jpg'.")

    for row in stream:
        doc_id = _choose_doc_id(row, doc_id_keys)

        # normalize pages
        if image_key == "images":
            pages: Iterable = row["images"]
            n_pages = len(pages)
            # NOTE: HF Image feature yields PIL.Image in streaming mode
            getter = lambda i: pages[i]
        else:
            n_pages = 1
            getter = lambda i: row[image_key]

        chosen = _select_pages(n_pages, pages_per_doc, rng)
        to_val = _hash01(doc_id, seed) < val_fraction

        for pidx in chosen:
            pil = getter(pidx)
            if not isinstance(pil, PILImage.Image):
                pil = PILImage.fromarray(np.asarray(pil))  # robust to array/bytes

            subdir = img_val if to_val else img_train
            ext = "jpg" if fmt in ("jpeg", "jpg") else "png"
            fname = f"{doc_id}_{pidx}.{ext}"
            dest = subdir / fname

            if not dry_run:
                if fmt in ("jpeg", "jpg"):
                    pil.convert("RGB").save(dest, format="JPEG", quality=jpeg_quality, optimize=True)
                else:
                    pil.save(dest, format="PNG", optimize=True)

            example = {
                "doc_id": doc_id,
                "page_index": int(pidx),
                "image": str(dest),
            }

            if to_val:
                if (max_val_pages is None) or (val_pages < max_val_pages):
                    if not dry_run:
                        val_writer.write(example)
                    val_pages += 1
            else:
                if (max_train_pages is None) or (train_pages < max_train_pages):
                    if not dry_run:
                        train_writer.write(example)
                    train_pages += 1

        # stop when both budgets are satisfied
        if (max_train_pages is not None and train_pages >= max_train_pages) and \
           (max_val_pages   is not None and val_pages   >= max_val_pages):
            break

    if dry_run:
        print(f"[dry-run] train pages≈{train_pages}  val pages≈{val_pages}")
        return

    train_writer.finalize()
    val_writer.finalize()

    # build map-style datasets and save bundle to disk
    train_ds = Dataset.from_file(str(out_dir / "train.arrow"))
    val_ds   = Dataset.from_file(str(out_dir / "val.arrow"))
    DatasetDict({"train": train_ds, "validation": val_ds}).save_to_disk(str(out_dir))

    print(f"[ok] Snapshot saved to {out_dir}")
    print(f"     train pages={len(train_ds)} | val pages={len(val_ds)}")
    print(f"     images in: {img_train} / {img_val}")


def main():
    p = argparse.ArgumentParser(description="Create a reusable local snapshot from a streaming HF dataset.")
    p.add_argument("--hf-path", required=True, help="HF dataset path or 'repo/config'.")
    p.add_argument("--split", default="train", help="Source split to stream (default: train).")
    p.add_argument("--out", dest="out_dir", required=True, help="Output directory for the snapshot bundle.")
    p.add_argument("--seed", type=int, default=0, help="Deterministic seed (affects doc routing & page selection).")
    p.add_argument("--val-fraction", type=float, default=0.05, help="Validation fraction at document level.")
    p.add_argument("--max-train-pages", type=int, default=None, help="Stop after N train pages (optional).")
    p.add_argument("--max-val-pages", type=int, default=None, help="Stop after N val pages (optional).")
    p.add_argument("--pages-per-doc", type=int, default=None, help="Sample up to K pages per doc (None = all).")
    p.add_argument("--doc-id-keys", nargs="+", default=["doc_id", "document_id", "id"],
                   help="Candidate keys to read a stable document id from.")
    p.add_argument("--image-key", default="image", choices=["image", "images"],
                   help="Column to read images from (single 'image' or list 'images').")
    p.add_argument("--format", dest="image_format", default="png", help="png | jpeg")
    p.add_argument("--jpeg-quality", type=int, default=90, help="JPEG quality if --format=jpeg.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory.")
    p.add_argument("--dry-run", action="store_true", help="Iterate & count without writing files.")
    args = p.parse_args()

    snapshot_stream(
        hf_path=args.hf_path,
        split=args.split,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        val_fraction=args.val_fraction,
        max_train_pages=args.max_train_pages,
        max_val_pages=args.max_val_pages,
        pages_per_doc=args.pages_per_doc,
        doc_id_keys=args.doc_id_keys,
        image_key=args.image_key,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
