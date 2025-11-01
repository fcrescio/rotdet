# evaluate_rotdet.py
import argparse, numpy as np, torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
from rotdet_model import load_rotdet
from rotdet_data import build_rotdet_dataset, build_rotdet_loader
from rotdet_hf import load_hf_dataset
from datasets import load_dataset

def evaluate(model, loader, device, fail_log=None, save_fail_images=None):
    model.eval()
    correct, total = 0, 0
    cm = np.zeros((4, 4), dtype=int)

    # Prepare logging
    fail_fp = None
    img_dir = None
    if fail_log:
        fail_fp = open(fail_log, "w", encoding="utf-8")
    if save_fail_images:
        img_dir = Path(save_fail_images)
        img_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for x, y, metas, pils in loader:
            x, y = x.to(device), y.to(device)
            x = x.float().div_(255)
            for k in (1,2,3):
                mask = (y == k)
                if mask.any():
                    x[mask] = torch.rot90(x[mask], k=k, dims=(2,3))
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
            for t, p in zip(y.tolist(), pred.tolist()):
                cm[t, p] += 1


            # failures
            if fail_fp or img_dir:
                for i in range(len(metas)):
                    if pred[i].item() == y[i].item():
                        continue
                    record = {
                        "true": int(y[i].item()),
                        "pred": int(pred[i].item()),
                        "meta": metas[i],
                    }
                    saved_path = None
                    if img_dir:
                        # filename: idx-rowX-pageY_trueT_predP.jpg
                        ridx = metas[i].get("row_idx")
                        pidx = metas[i].get("page_idx")
                        fname = f"row{ridx}_page{pidx}_true{record['true']}_pred{record['pred']}.png"
                        path = img_dir / fname
                        # pils[i] is the exact PIL we evaluated (already rotated if label==1)
                        pils[i].save(path)
                        saved_path = str(path)
                        record["saved_image"] = saved_path
                    if fail_fp:
                        fail_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    if fail_fp:
        fail_fp.close()

    return (correct / total) if total else 0.0, cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="fcrescio/rotdet")
    ap.add_argument("--filename", default="model.safetensors")
    ap.add_argument("--dataset", default="nielsr/funsd")
    ap.add_argument("--config", default=None, help="HF dataset config (e.g. images / zero-shot-exp). If omitted, auto-pick.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--pages-per-doc", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--rotate_prob", type=float, default=0.5)
    ap.add_argument("--fail_log", default=None, help="Path to write JSONL with failed samples.")
    ap.add_argument("--save_fail_images", default=None, help="Directory to save failed page images.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_rotdet(args.repo_id, args.filename, device)

    hf_obj = load_hf_dataset(
        args.dataset,
        split=args.split,
        config=args.config,
        streaming=args.streaming,
    )
    ds = build_rotdet_dataset(
        hf_obj,
        streaming=args.streaming,
        rotate_prob=args.rotate_prob,
        pages_per_doc=args.pages_per_doc,
        max_samples=(args.max_samples if args.max_samples > 0 else None),
    )
    loader = build_rotdet_loader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers, device=device, streaming=args.streaming
    )

    acc, cm = evaluate(model, loader, device, args.fail_log, args.save_fail_images)
    print(f"\nSamples: {len(ds) if hasattr(ds, '__len__') else 'streamed'}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred) [Normal, Rotated]:")
    print(cm)

if __name__ == "__main__":
    main()
