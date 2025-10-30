#!/usr/bin/env python3
import argparse, numpy as np, json
from pathlib import Path
from typing import List, Tuple, Any, Dict
import torch
from torch.utils.data import DataLoader
from PIL import Image

from rotdet_data import build_rotdet_dataset, build_rotdet_loader
from rotdet_hf import load_hf_dataset

# ----------------------------- PaddleOCR wrapper -----------------------------
# Priority:
#  1) PP-StructureV3 with doc orientation classify (latest 3.x)
#  2) PP-Structure legacy 'image_orientation=True' (2.x fallback)
#  3) (optional) PaddleClas PULC text-image orientation model if available
#
# Notes on 3.x:
#  - The PP-StructureV3 demo uses a "use_doc_orientation_classify" switch to enable
#    document image orientation classification (0/90/180/270). We'll mirror that
#    behavior when instantiating the pipeline. :contentReference[oaicite:1]{index=1}
#  - PaddleOCR 3.x also refreshed default *text line* orientation models (0/180),
#    but we explicitly evaluate the *document* orientation module here. :contentReference[oaicite:2]{index=2}

def _angle_to_class(angle_deg: int) -> int:
    a = (int(round(angle_deg)) % 360 + 360) % 360
    mapping = {0:0, 90:1, 180:2, 270:3, 360:0}
    if a in mapping: return mapping[a]
    nearest = min([0,90,180,270], key=lambda t: abs(((a - t + 180) % 360) - 180))
    return mapping[nearest]

class PaddleDocOrientation:
    """Predicts one of {0,1,2,3} for {0°,90°,180°,270°} using PaddleOCR."""
    def __init__(self, device: str = "cpu", model_dir: str | None = None):
        self.device = device
        self.backend = None

        # Try PP-StructureV3 style init (3.x). The HF demo exposes this switch. :contentReference[oaicite:3]{index=3}
        try:
            from paddleocr import PPStructure  # 3.x still exports PPStructure
            # Newer builds accept doc-orientation switch; try common arg names:
            try:
                self.pp = PPStructure(
                    show_log=False,
                    # preferred 3.x flag (string seen in official demo):
                    use_doc_orientation_classify=True,
                    # keep heavy subsystems off:
                    layout=False, ocr=False, table=False,
                    use_gpu=(device.startswith("cuda")),
                )
                self.backend = "PPStructure-3x-doc-orient"
            except TypeError:
                # Fallback to legacy 2.x flag:
                self.pp = PPStructure(
                    show_log=False,
                    image_orientation=True,
                    layout=False, ocr=False, table=False,
                    use_gpu=(device.startswith("cuda")),
                )
                self.backend = "PPStructure-2x-image_orientation"
        except Exception as e:
            # Optional PaddleClas PULC fallback (if installed)
            try:
                from paddleserving_client import Client  # noqa: F401
                raise RuntimeError("PaddleClas/PULC client not wired in this evaluator.") from e
            except Exception as ee:
                raise RuntimeError(
                    "Could not initialize PaddleOCR PP-Structure. "
                    "Please ensure paddlepaddle & paddleocr (3.x recommended) are installed."
                ) from ee

    def _extract_angle_from_struct(self, info: Any) -> int | None:
        """
        PP-Structure returns either a list of dicts or a dict. We try several keys
        that various releases use to expose degrees.
        """
        if info is None:
            return None
        if isinstance(info, list):
            info = info[0] if info else {}
        if isinstance(info, dict):
            # 3.x doc orientation often surfaces as one of these:
            for k in ("doc_orientation", "orientation", "img_orientation",
                      "angle", "rotation", "img_rotation"):
                val = info.get(k, None)
                if val is not None:
                    try:
                        return int(round(float(val)))
                    except Exception:
                        pass
        # Some pipelines return tuples/lists: (angle, score)
        if isinstance(info, (list, tuple)) and len(info) > 0:
            try:
                return int(round(float(info[0])))
            except Exception:
                return None
        # As a last resort, treat direct numeric
        try:
            return int(round(float(info)))
        except Exception:
            return None

    def predict_batch(self, pils: List[Image.Image]) -> List[int]:
        preds = []
        for img in pils:
            out = self.pp(img)
            angle = self._extract_angle_from_struct(out)
            if angle is None:
                angle = 0
            preds.append(_angle_to_class(angle))
        return preds

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

    predictor = PaddleDocOrientation(device=device)

    with torch.no_grad():
        for _, y, metas, pils in loader:
            preds = predictor.predict_batch(pils)
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
                        fname = f"row{ridx}_page{pidx}_true{record['true']}_pred{record['pred']}.jpg"
                        path = img_dir / fname
                        pils[i].save(path)
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
