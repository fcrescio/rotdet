#!/usr/bin/env python3
"""
Filter a local (map-style) snapshot by keeping only images PaddleOCR can orient.

Example:
  uv run scripts/paddleocr_prefilter.py \
    --input data/snapshots/rotdet_docmatix_s0_v5 \
    --output data/snapshots/rotdet_docmatix_s0_v5_paddleclean \
    --splits train validation \
    --angles 0 1 2 3 \
    --min-correct 2 \
    --fail-log paddle_failures.jsonl
"""
from __future__ import annotations
import argparse
import json
from typing import Iterable, List, Sequence

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk
from PIL import Image as PILImage
from tqdm.auto import tqdm


# ----------------------------- Core filtering -----------------------------

def _predict_angle(model, pil: PILImage.Image) -> int:
    """Run PaddleOCR orientation classifier and return predicted class id."""
    arr = np.asarray(pil.convert("RGB"))
    output = model.predict(arr)
    for res in output:
        return int(res.json["res"]["class_ids"][0])
    raise RuntimeError("PaddleOCR returned an empty result list.")


def _passes_filter(model, pil: PILImage.Image, angles: Sequence[int], min_correct: int) -> tuple[bool, list[dict]]:
    """Return whether ``pil`` meets the filter threshold and the raw predictions."""
    correct = 0
    preds: list[dict] = []
    for k in angles:
        rotated = pil.rotate(90 * k, expand=False)
        pred = _predict_angle(model, rotated)
        preds.append({"angle": int(k), "pred": pred})
        correct += int(pred == k)
    return correct >= min_correct, preds


def _filter_split(
    split_name: str,
    ds: Dataset,
    model,
    angles: Sequence[int],
    min_correct: int,
    fail_fp,
) -> Dataset:
    keep_indices: List[int] = []
    total = len(ds)
    for idx, ex in enumerate(tqdm(ds, total=total, desc=f"{split_name} | filtering")):
        pil = ex.get("image")
        if not isinstance(pil, PILImage.Image):
            pil = PILImage.fromarray(np.asarray(pil))
        keep, preds = _passes_filter(model, pil, angles, min_correct)
        if keep:
            keep_indices.append(idx)
        elif fail_fp:
            meta = {k: ex.get(k) for k in ("doc_id", "page_index", "id") if k in ex}
            fail_fp.write(json.dumps({"split": split_name, "index": idx, "meta": meta, "preds": preds}) + "\n")
    return ds.select(keep_indices)


# ----------------------------- CLI wrapper ------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Filter a local snapshot with PaddleOCR orientation predictions.")
    p.add_argument("--input", required=True, help="Path to load_from_disk() snapshot (Dataset or DatasetDict).")
    p.add_argument("--output", required=True, help="Where to save the filtered snapshot.")
    p.add_argument("--splits", nargs="+", default=["train", "validation", "test"], help="Splits to filter if present.")
    p.add_argument("--angles", nargs="+", type=int, default=[0, 1, 2, 3], choices=[0, 1, 2, 3], help="Quarter-turn angles (0-3) to test.")
    p.add_argument("--min-correct", type=int, default=2, help="Keep sample if PaddleOCR gets at least this many angles correct.")
    p.add_argument("--fail-log", default=None, help="Optional path to write JSONL of filtered-out samples.")
    return p.parse_args()


def main():
    args = _parse_args()

    if args.min_correct > len(args.angles):
        raise SystemExit("--min-correct cannot exceed number of --angles provided.")

    from paddleocr import DocImgOrientationClassification

    model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")

    obj = load_from_disk(args.input)
    if isinstance(obj, DatasetDict):
        splits: Iterable[tuple[str, Dataset]] = (
            (name, obj[name]) for name in args.splits if name in obj
        )
    elif isinstance(obj, Dataset):
        splits = [("train", obj)]
    else:
        raise SystemExit(f"Unsupported object from load_from_disk: {type(obj)}")

    fail_fp = open(args.fail_log, "w", encoding="utf-8") if args.fail_log else None

    filtered = {}
    stats = []
    for name, ds in splits:
        kept_ds = _filter_split(name, ds, model, args.angles, args.min_correct, fail_fp)
        filtered[name] = kept_ds
        stats.append((name, len(ds), len(kept_ds)))

    if fail_fp:
        fail_fp.close()

    DatasetDict(filtered).save_to_disk(args.output)

    print("\n[done] Filtered snapshot saved to", args.output)
    for name, before, after in stats:
        frac = (after / before) if before else 0.0
        print(f"  {name}: kept {after} / {before} samples ({frac:.1%})")


if __name__ == "__main__":
    main()
