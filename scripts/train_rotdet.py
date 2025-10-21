# train_rotdet.py
import argparse, os, json, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from safetensors.torch import save_file, load_file

from rotdet_model import SimpleCNN, load_rotdet
from rotdet_data import build_rotdet_loader, build_rotdet_dataset_pair
from rotdet_hf import load_hf_dataset  # transparent config picker

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y, metas, pils in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return (correct / total) if total else 0.0

def save_checkpoint(model, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.safetensors"
    state = model.state_dict()
    save_file(state, str(path))
    return path

def maybe_resume(model, resume_path: str | None, device: str):
    if not resume_path:
        return False
    p = Path(resume_path)
    if not p.exists():
        print(f"[resume] path not found: {p}")
        return False
    print(f"[resume] loading weights from {p}")
    state = load_file(str(p))
    model.load_state_dict(state)
    model.to(device)
    return True

def main():
    ap = argparse.ArgumentParser()
    # data
    ap.add_argument("--dataset", default="HuggingFaceM4/Docmatix")
    ap.add_argument("--config", default=None, help="HF dataset config (auto-picked if omitted)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--pages-per-doc", type=int, default=2)
    # validation sizing
    ap.add_argument("--val-fraction", type=float, default=0.1,
                    help="Map-style only. Fraction of rows for validation.")
    ap.add_argument("--val-pages", type=int, default=2000,
                    help="Streaming only. Exact number of page-samples for validation.")
    # train caps
    ap.add_argument("--max-train-pages", type=int, default=0,
                    help="Optional cap on number of training page-samples (0 = no cap).")
    # training
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--rotate-prob", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    # checkpoints
    ap.add_argument("--output-dir", default="checkpoints")
    ap.add_argument("--resume", default=None, help="Path to .safetensors to resume weights")
    ap.add_argument("--save-every-epoch", action="store_true")
    # model init
    ap.add_argument("--from-pretrained", action="store_true",
                    help="Start from fcrescio/rotdet weights")
    ap.add_argument("--repo-id", default="fcrescio/rotdet")
    ap.add_argument("--filename", default="model.safetensors")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- model ---
    if args.from_pretrained:
        model = load_rotdet(args.repo_id, args.filename, device)
    else:
        model = SimpleCNN().to(device)

    maybe_resume(model, args.resume, device)

    # --- data (two streams for streaming train/val) ---
    if args.streaming:
        # independent streams: one for val, one for train (skip val pages)
        val_stream = load_hf_dataset(args.dataset, split=args.split, config=args.config, streaming=True)
        train_stream = load_hf_dataset(args.dataset, split=args.split, config=args.config, streaming=True)

        train_set, val_set = build_rotdet_dataset_pair(
            val_stream,
            train_stream,  # used for training (will be skipped inside)
            streaming=True,
            rotate_prob=args.rotate_prob,
            pages_per_doc=args.pages-per-doc if hasattr(args, "pages-per-doc") else args.pages_per_doc,
            val_fraction=None,
            val_pages=args.val_pages,
            max_train_pages=(args.max_train_pages or None),
        )
        shuffle_flag = False
    else:
        ds_rows = load_hf_dataset(args.dataset, split=args.split, config=args.config, streaming=False)
        train_set, val_set = build_rotdet_dataset_pair(
            ds_rows,
            streaming=False,
            rotate_prob=args.rotate_prob,
            pages_per_doc=args.pages_per_doc,
            val_fraction=args.val_fraction,
            val_pages=None,
            max_train_pages=(args.max_train_pages or None),
            seed=args.seed,
        )
        shuffle_flag = True

    train_loader = build_rotdet_loader(
        train_set, batch_size=args.batch_size, num_workers=args.num_workers, device=device, streaming=args.streaming
    )
    val_loader = build_rotdet_loader(
        val_set, batch_size=args.batch_size, num_workers=args.num_workers, device=device, streaming=args.streaming
    )

    # --- optim ---
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    out_dir = Path(args.output_dir)
    best_acc = -1.0
    history = {"epochs": [], "best": {}}

    # --- train loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        t0 = time.time()

        for x, y, metas, pils in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
            steps += 1

        train_loss = running / max(steps, 1)
        val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}  time={dt:.1f}s")

        # save last + optionally per-epoch
        last_path = save_checkpoint(model, out_dir, "last")
        if args.save_every_epoch:
            save_checkpoint(model, out_dir, f"epoch{epoch:03d}")

        # track best
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = save_checkpoint(model, out_dir, "best")

        history["epochs"].append({
            "epoch": epoch, "train_loss": train_loss, "val_acc": val_acc, "time_s": dt
        })

    # save training summary
    history["best"] = {"val_acc": best_acc}
    (out_dir / "training_summary.json").write_text(json.dumps(history, indent=2))
    print(f"\nSaved: last -> {last_path}")
    if best_acc >= 0:
        print(f"Saved: best -> {best_path} (val_acc={best_acc:.4f})")

