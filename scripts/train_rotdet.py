# train_rotdet.py
import argparse, torch, torch.nn as nn, torch.optim as optim
from datasets import load_dataset
from rotdet_model import SimpleCNN   # or load_rotdet if fine-tuning pretrained
from rotdet_data import build_rotdet_dataset, build_rotdet_loader
from rotdet_hf import load_hf_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="HuggingFaceM4/Docmatix")
    ap.add_argument("--config", default=None, help="HF dataset config; auto-chosen if omitted")
    ap.add_argument("--split", default="train")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--pages-per-doc", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)  # or: model = load_rotdet(device=device)

    hf_obj = load_hf_dataset(args.dataset, split=args.split, config=args.config, streaming=args.streaming)
    ds = build_rotdet_dataset(
        hf_obj,
        streaming=args.streaming,
        rotate_prob=0.5,
        pages_per_doc=args.pages_per_doc,
        max_samples=(args.max_samples if args.max_samples > 0 else None),
    )
    loader = build_rotdet_loader(
        ds, batch_size=args.batch_size, num_workers=args.num_workers, device=device, streaming=args.streaming
    )

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"epoch {epoch+1}: loss={running:.4f}")

if __name__ == "__main__":
    main()
