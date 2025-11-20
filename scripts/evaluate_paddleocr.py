#!/usr/bin/env python3
import argparse, numpy as np, json
from pathlib import Path
from typing import List, Tuple, Any, Dict
import torch
from torch.utils.data import DataLoader
from PIL import Image

from rotdet_data import build_rotdet_dataset, build_rotdet_loader
from rotdet_hf import load_hf_dataset

# ----------------------------- Evaluation loop ------------------------------

def evaluate_paddle(
    loader: DataLoader,
    fail_log: str | None = None,
    save_fail_images: str | None = None,
    device: str = "cpu",
) -> Tuple[float, np.ndarray]:
    correct, total = 0, 0
    cm = np.zeros((4, 4), dtype=int)

    fail_fp = open(fail_log, "w", encoding="utf-8") if fail_log else None
    img_dir = Path(save_fail_images) if save_fail_images else None
    if img_dir: img_dir.mkdir(parents=True, exist_ok=True)

    from paddleocr import DocImgOrientationClassification

    model = DocImgOrientationClassification(model_name="PP-LCNet_x1_0_doc_ori")

    with torch.no_grad():
        for x, y, metas, pils in loader:
            preds: List[int] = []

            # Build a batch of PIL images that match the already-rotated samples
            pil_batch: List[Image.Image] = []
            for i in range(len(y)):
                pil = pils[i]
                if pil is None:
                    # Map datasets don't carry PILs by default; rebuild from tensors
                    arr = x[i].detach().cpu().squeeze(0).numpy()
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                    pil = Image.fromarray(arr, mode="L")
                pil_batch.append(pil)

            for pil in pil_batch:
                arr = np.asarray(pil.convert("RGB"))
                output = model.predict(arr)
                for res in output:
                    res.print(json_format=False)
                    preds.append(res.json['res']['class_ids'][0])

            y_list = y.tolist()

            for t, p in zip(y_list, preds):
                total += 1
                correct += int(p == t)
                cm[t, p] += 1

            if fail_fp or img_dir:
                for i in range(len(metas)):
                    if preds[i] == y_list[i]:
                        continue
                    record: Dict[str, Any] = {
                        "true": int(y_list[i]),
                        "pred": int(preds[i]),
                        "meta": metas[i],
                    }
                    if img_dir:
                        ridx = metas[i].get("row_idx")
                        pidx = metas[i].get("page_idx")
                        fname = f"row{ridx}_page{pidx}_true{record['true']}_pred{record['pred']}.png"
                        path = img_dir / fname
                        pil_batch[i].save(path)
                        record["saved_image"] = str(path)
                    if fail_fp:
                        fail_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    if fail_fp:
        fail_fp.close()

    acc = (correct / total) if total else 0.0
    return acc, cm

def main():
    ap = argparse.ArgumentParser()
    # same knobs you use for rotdet eval
    ap.add_argument("--dataset", default="nielsr/funsd")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--pages-per-doc", type=int, default=None)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rotate_prob", type=float, default=0.5)

    ap.add_argument("--fail_log", default=None)
    ap.add_argument("--save_fail_images", default=None)

    args = ap.parse_args()

    hf_obj = load_hf_dataset(
        args.dataset, split=args.split, config=args.config, streaming=args.streaming
    )
    ds = build_rotdet_dataset(
        hf_obj,
        streaming=args.streaming,
        rotate_prob=args.rotate_prob,
        pages_per_doc=args.pages_per_doc,
        max_samples=(args.max_samples if args.max_samples > 0 else None),
    )
    loader = build_rotdet_loader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers,
        device=args.device, streaming=args.streaming
    )

    acc, cm = evaluate_paddle(loader, args.fail_log, args.save_fail_images, device=args.device)
    print(f"\nSamples: {len(ds) if hasattr(ds, '__len__') else 'streamed'}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred) [0°, 90°, 180°, 270°]:")
    print(cm)

if __name__ == "__main__":
    main()
