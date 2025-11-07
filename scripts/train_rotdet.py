# train_rotdet.py
import argparse, os, json, time
from pathlib import Path

from datetime import datetime
from typing import Dict

import torch
import torch.optim as optim

from datasets import load_dataset, load_from_disk
from safetensors.torch import save_file, load_file

from models import available_model_names, create_model
from rotdet_model import load_rotdet
from rotdet_data import build_rotdet_loader, build_rotdet_dataset, build_rotdet_dataset_pair
from rotdet_hf import load_hf_dataset  # transparent config picker

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import numpy as np

def evaluate(model, loader, device) -> Dict:
    model.eval()
    correct, total = 0, 0
    total_loss, n_batches = 0.0, 0
    conf = torch.zeros(4, 4, dtype=torch.long)
    with torch.no_grad():
        bar = tqdm(desc="Validating", total=len(loader))
        for x, y, metas, pils in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            if hasattr(model, "compute_loss"):
                loss = model.compute_loss(logits, y)
            else:
                loss = torch.nn.functional.cross_entropy(logits, y)
            total_loss += loss.item(); n_batches += 1
            correct += (pred == y).sum().item()
            total += y.numel()
            for t, p in zip(y.view(-1), pred.view(-1)):
                conf[t.long(), p.long()] += 1
            bar.update(1)
        bar.close()
    acc = (correct / total) if total else 0.0
    val_loss = (total_loss / max(n_batches, 1))
    acc_per_class = (conf.diag().float() / conf.sum(dim=1).clamp(min=1).float()).tolist()
    return {
        "val_acc": acc,
        "val_loss": val_loss,
        "confusion": conf.tolist(),
        "acc_per_class": acc_per_class
    }

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
    ap.add_argument("--snapshot_dir",type=str,default=None,help="If set, load a pre-built snapshot (DatasetDict with 'train' and 'validation') from disk.")
    # Aim (logging locale)
    ap.add_argument("--aim", action="store_true", help="Abilita logging Aim (100% locale)")
    ap.add_argument("--aim-repo", type=str, default="runs/aim",
                    help="Cartella repository Aim locale (es. runs/aim)")
    ap.add_argument("--experiment", type=str, default="RotDet-C4",
                    help="Nome esperimento Aim")
    ap.add_argument("--run-name", type=str, default=None,
                    help="Nome run Aim (di default timestamp)")

    model_choices = available_model_names()
    ap.add_argument(
        "--model",
        choices=model_choices,
        default="c4net",
        help=f"Model architecture to use. Choices: {', '.join(model_choices)}",
    )
    ap.add_argument(
        "--model-kwargs",
        default=None,
        help="Optional JSON string with keyword arguments to override model defaults.",
    )

    args = ap.parse_args()

    model_overrides: Dict | None = None
    if args.model_kwargs:
        try:
            parsed = json.loads(args.model_kwargs)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON for --model-kwargs: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit("--model-kwargs must decode to a JSON object")
        model_overrides = parsed

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- model ---
    if args.from_pretrained:
        model = load_rotdet(
            args.repo_id,
            args.filename,
            device,
            model=args.model,
            model_kwargs=model_overrides,
        )
    else:
        overrides = model_overrides or {}
        model = create_model(args.model, **overrides).to(device)

    maybe_resume(model, args.resume, device)

    # --- Aim setup (opzionale) ---
    aim_run = None
    if args.aim:
        try:
            from aim import Run, Image
            aim_run = Run(repo=args.aim_repo, experiment=args.experiment)
            aim_run.name = args.run_name or f"{args.experiment}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # Logga iperparametri principali
            aim_run["hparams"] = dict(
                model=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                rotate_prob=args.rotate_prob,
                seed=args.seed,
            )
        except Exception as e:
            print(f"[Aim] init fallita: {e}")
            aim_run = None


    # --- data (two streams for streaming train/val) ---
    if args.snapshot_dir:
        # Pre-split snapshot: no downloads, no internal splitting
        dsd = load_from_disk(args.snapshot_dir)  # expects {'train', 'validation'}

        # If your builder can take explicit splits, build each split independently.
        # (Most repos expose a single-split builder under the hood; if yours doesn’t,
        #  see the alt. block below.)
        train_set = build_rotdet_dataset(
        dsd["train"],
        streaming=False,
        rotate_prob=args.rotate_prob,
        pages_per_doc=args.pages_per_doc,
        seed=args.seed,
        )
        val_set = build_rotdet_dataset(
        dsd["validation"],
        streaming=False,
        rotate_prob=args.rotate_prob,
        pages_per_doc=args.pages_per_doc,
        seed=args.seed,
        )
        shuffle_flag = True
    elif args.streaming:
        # independent streams: one for val, one for train (skip val pages)
        val_stream = load_hf_dataset(args.dataset, split=args.split, config=args.config, streaming=True)
        train_stream = load_hf_dataset(args.dataset, split=args.split, config=args.config, streaming=True)

        train_set, val_set = build_rotdet_dataset_pair(
            val_stream,
            train_stream,  # used for training (will be skipped inside)
            streaming=True,
            rotate_prob=args.rotate_prob,
            pages_per_doc=args.pages_per_doc,
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

    out_dir = Path(args.output_dir)
    best_acc = -1.0
    history = {"epochs": [], "best": {}}

    # --- train loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0
        t0 = time.time()

        bar = tqdm(desc="Training", total=len(train_loader))
        scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
        torch.backends.cudnn.benchmark = True
        for x, y, metas, pils in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device):
                logits = model(x)
                loss = model.compute_loss(logits, y)
            scaler.scale(loss).backward()  
            scaler.step(opt)               
            scaler.update()                
            running += loss.item()
            steps += 1
            bar.update(1)
        bar.close()

        train_loss = running / max(steps, 1)
        eval_out = evaluate(model, val_loader, device)
        dt = time.time() - t0

        print(f"epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  "
              f"val_acc={eval_out['val_acc']:.4f}  "
              f"val_loss={eval_out['val_loss']:.4f}  "
              f"time={dt:.1f}s")

        # save last + optionally per-epoch
        last_path = save_checkpoint(model, out_dir, "last")
        if args.save_every_epoch:
            save_checkpoint(model, out_dir, f"epoch{epoch:03d}")

        # track best
        if eval_out['val_acc'] > best_acc:
            best_acc = eval_out['val_acc']
            best_path = save_checkpoint(model, out_dir, "best")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": eval_out["val_acc"],
            "val_loss": eval_out["val_loss"],
            "time_s": dt,
        }
        history["epochs"].append(row)
        # --- Aim logging per epoca ---
        if aim_run is not None:
            # scalari principali
            aim_run.track(train_loss,            name="loss", step=epoch, context={"subset": "train"})
            aim_run.track(eval_out["val_loss"],  name="loss", step=epoch, context={"subset": "val"})
            aim_run.track(eval_out["val_acc"],   name="acc",  step=epoch, context={"subset": "val"})

            # learning-rate corrente (se AdamW/SGD standard)
            try:
                aim_run.track(opt.param_groups[0]["lr"], name="lr", step=epoch)
            except Exception:
                pass

            # confusion matrix come figura matplotlib
            try:
                cm = np.array(eval_out["confusion"], dtype=np.int64)
                fig = plt.figure()
                plt.imshow(cm, interpolation="nearest")
                plt.title(f"Confusion epoch {epoch}")
                plt.xticks(range(cm.shape[0]), [0, 90, 180, 270])
                plt.yticks(range(cm.shape[0]), [0, 90, 180, 270])
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
                plt.tight_layout()
                aim_run.track(Image(fig), name="confusion", step=epoch)
                plt.close(fig)
            except Exception as e:
                print(f"[Aim] confusion plot fallito: {e}")

            # accuratezza per classe
            for i, v in enumerate(eval_out["acc_per_class"]):
                aim_run.track(float(v), name="acc_per_class", step=epoch, context={"cls": str(i)})

            # checkpoint come artefatti (last sempre, best solo quando aggiornato)
            #try:
            #    from aim import File
            #    aim_run.track(File(str(last_path)), name="checkpoint", step=epoch, context={"kind": "last"})
            #    if best_acc == eval_out["val_acc"]:
            #        aim_run.track(File(str(best_path)), name="checkpoint", step=epoch, context={"kind": "best"})
            #except Exception as e:
            #    print(f"[Aim] checkpoint artifact fallito: {e}")


    # save training summary
    history["best"] = {"val_acc": best_acc}
    (out_dir / "training_summary.json").write_text(json.dumps(history, indent=2))
    print(f"\nSaved: last -> {last_path}")
    if best_acc >= 0:
        print(f"Saved: best -> {best_path} (val_acc={best_acc:.4f})")

    if aim_run is not None:
        aim_run.close()

