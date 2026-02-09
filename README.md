Of course 👍 — here’s a clean re-rendered version of the full **`README.md`** (no formatting breaks this time).
You can copy it directly into a `README.md` file at the root of your repo.

---

# 🌀 RotDet — Document Page Orientation Detection

RotDet is a lightweight PyTorch training and evaluation framework for detecting page orientation (“rotated vs. normal”) in scanned documents.

It re-implements and extends the [fcrescio/rotdet](https://huggingface.co/fcrescio/rotdet) model from Hugging Face Hub, supporting both **map-style** and **streaming** datasets (e.g. [HuggingFaceM4/Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix)).

---

## ✨ Features

* ✅ Training and evaluation of rotation detector CNNs
* ✅ Works with Hugging Face datasets (map-style or streaming)
* ✅ Handles multi-page documents (`images` key in Docmatix)
* ✅ Auto-splitting into train/validation sets
* ✅ Checkpointing (`last.safetensors`, `best.safetensors`)
* ✅ Resume or fine-tune from pretrained weights
* ✅ Automatic config selection (`images`, `pdf`, `zero-shot-exp`, etc.)
* ✅ Evaluation with fail logging (JSONL + failed image export)
* ✅ Modular design: reusable data, model, and script modules

---

## 🧱 Project structure

```
rotdet/
├── scripts/       
├── rotdet_data.py       # Dataset wrappers + loaders
├── rotdet_model.py      # SimpleCNN + weight loading utilities
├── rotdet_hf.py         # Hugging Face config + dataset loader helpers
├── train_rotdet.py      # Training entry point
├── evaluate_rotdet.py   # Evaluation entry point
└── pyproject.toml       # Build & CLI definitions
```

When installed, the following console commands are available:

* `rotdet-train` → train or fine-tune a model
* `rotdet-eval` → evaluate a model checkpoint

---

## ⚙️ Installation

Requires **Python ≥3.9** and **PyTorch ≥2.0**.

### Using [uv](https://docs.astral.sh/uv/)

```bash
uv pip install -e .
```

or with pip:

```bash
pip install -e .
```

---

## 🚀 Quick Start

### 1️⃣ Evaluate the pretrained model

```bash
uv run rotdet-eval \
  --repo_id fcrescio/rotdet \
  --filename model.safetensors \
  --dataset nielsr/funsd \
  --split test
```

---

### 2️⃣ Train on a Hugging Face dataset

#### Map-style example (FUNSD)

```bash
uv run rotdet-train \
  --dataset nielsr/funsd \
  --split train \
  --val-fraction 0.2 \
  --epochs 3 --batch-size 64 \
  --output-dir checkpoints/funsd
```

#### Streaming example (Docmatix)

```bash
uv run rotdet-train \
  --dataset HuggingFaceM4/Docmatix \
  --config images \
  --streaming \
  --pages-per-doc 3 \
  --val-pages 4000 \
  --max-train-pages 20000 \
  --epochs 2 --batch-size 64 \
  --output-dir checkpoints/docmatix_stream
```

---

### 3️⃣ Resume training or fine-tune

```bash
uv run rotdet-train \
  --resume checkpoints/docmatix_stream/best.safetensors \
  --dataset HuggingFaceM4/Docmatix --config images --streaming \
  --epochs 2
```

---

### 4️⃣ Evaluate a local checkpoint

```bash
uv run rotdet-eval \
  --repo_id checkpoints/docmatix_stream/best.safetensors \
  --dataset HuggingFaceM4/Docmatix \
  --config images --streaming \
  --max_samples 2000
```

---

## 🔧 Model kwargs overrides (bash-friendly)

`rotdet-train` and `rotdet-eval` accept three override formats (you can mix them):

1. **Repeatable `key=value` pairs** (most bash-friendly):
   ```bash
   uv run rotdet-train \
     --dataset nielsr/funsd \
     --model c4net \
     --model-kwarg stem_ch=24 \
     --model-kwarg widths=[32,64,192] \
     --model-kwarg head_type=orientation
   ```
   *Values are type-inferred: `true/false`, `null/none`, numbers, or JSON lists/objects. Anything else stays a string.*

2. **JSON file** (no escaping):
   ```bash
   uv run rotdet-eval \
     --repo_id checkpoints/docmatix_stream/best.safetensors \
     --dataset HuggingFaceM4/Docmatix \
     --model-kwargs-file configs/c4net_overrides.json
   ```

3. **Raw JSON string** (legacy, still supported):
   ```bash
   uv run rotdet-train \
     --dataset nielsr/funsd \
     --model-kwargs '{"stem_ch": 24, "widths": [32, 64, 192]}'
   ```

**Precedence (last wins):** `--model-kwargs-file` → `--model-kwargs` → `--model-kwarg`.

---

## 🧩 Evaluation options

### Log failed samples (JSONL)

```bash
uv run rotdet-eval \
  --repo_id checkpoints/funsd/best.safetensors \
  --dataset nielsr/funsd --split test \
  --fail-log failures.jsonl
```

Each line looks like:

```json
{"true":1,"pred":0,"meta":{"row_idx":12,"page_idx":3,"id":"doc123","image_path":"..."}}
```

---

### Save failed images for visual debugging

```bash
uv run rotdet-eval \
  --repo_id checkpoints/funsd/best.safetensors \
  --dataset HuggingFaceM4/Docmatix --config images \
  --fail-log failures.jsonl \
  --save-fail-images debug_fails/
```

This writes a `.jsonl` log plus the actual failed page images into `debug_fails/`.

---

## 🧠 Dataset support

RotDet works with any Hugging Face dataset that provides images via:

* `image` → single page per row (e.g. FUNSD)
* `images` → list of pages per document (e.g. Docmatix)
* `image_path` or `url` → path/URL to image

For Docmatix, the `config` is handled automatically:

```bash
# Automatically picks the best available config (usually 'images')
uv run rotdet-train --dataset HuggingFaceM4/Docmatix
```

or explicitly:

```bash
uv run rotdet-train --dataset HuggingFaceM4/Docmatix --config zero-shot-exp
```

### 📥 Mining Reader Service scans locally

Need a lightweight document dataset without relying on an existing Hugging Face repo? Use the new helper script to mine the
[`readerservice`](https://archive.org/details/readerservice) collection from Internet Archive and export it as a Hugging Face
`DatasetDict`:

```bash
python scripts/readerservice_miner.py \
  --output-dir data/readerservice \
  --val-fraction 0.1 \
  --max-images 2000
```

The script downloads the source documents (PDFs on Internet Archive), renders each page as an image under
`data/readerservice/images/`, saves the original files under `data/readerservice/documents/`, materializes a HF-compliant
dataset via `datasets.save_to_disk`, and produces a `manifest.json` with provenance info so you can train with
`load_from_disk("data/readerservice/hf_dataset")` directly.

---

## 🧪 Experiment management

Each training run writes its own timestamped folder with:

* `last.safetensors` — weights after the final epoch
* `best.safetensors` — best validation accuracy
* `training_summary.json` — per-epoch metrics
* `hparams.json` — full training configuration

Example:

```bash
checkpoints/docmatix_stream/
├── best.safetensors
├── last.safetensors
├── hparams.json
└── training_summary.json
```

---

## 📊 Example training summary

```json
{
  "epochs": [
    {"epoch":1,"train_loss":0.2413,"val_acc":0.974,"time_s":42.1},
    {"epoch":2,"train_loss":0.1384,"val_acc":0.982,"time_s":41.8}
  ],
  "best": {"val_acc":0.982}
}
```

---

## 🧱 Model architecture

The default `SimpleCNN` is a small three-layer convolutional classifier:

```
Input:  (1, 128, 128)
↓ Conv2d(1→16) + ReLU + MaxPool2d
↓ Conv2d(16→32) + ReLU + MaxPool2d
↓ Conv2d(32→32) + ReLU + MaxPool2d
↓ Flatten → Linear(8192→32) → ReLU → Linear(32→2)
Output: logits for [Normal, Rotated]
```

You can replace or extend it in `rotdet_model.py`.

---

## 🧰 Development

Typical workflow:

```bash
uv run rotdet-train   # train new model
uv run rotdet-eval    # evaluate checkpoint
uv run pytest         # (optional) run tests
```

---

## 🤖 MCP evaluation server

RotDet includes an MCP server that exposes the evaluation pipeline as a tool for
agentic clients. See the full usage guide, including schema details and example
client configurations:

* [`docs/evaluate-mcp-server.md`](docs/evaluate-mcp-server.md)【F:docs/evaluate-mcp-server.md†L1-L163】

---

## 💡 Experiment tips

* Use `--output-dir` per experiment (each run auto-timestamps).
* Enable `--save-every-epoch` for detailed logs.
* Add `--from-pretrained` to start from Hugging Face weights.
* Use Optuna or Hydra for hyperparameter sweeps (supported by design).
* Add `--fail-log` during evaluation to detect dataset issues.

---

## 📄 License

MIT License © 2025 — RotDet contributors
Based on the pretrained model [fcrescio/rotdet](https://huggingface.co/fcrescio/rotdet).

---

## 🔗 References

* [fcrescio/rotdet](https://huggingface.co/fcrescio/rotdet) — Original model
* [HuggingFaceM4/Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix) — Document dataset
* [Hugging Face Datasets](https://huggingface.co/docs/datasets)
* [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)

---

✅ **Summary**:
This README documents installation, training, evaluation, dataset support, fail-logging, and experiment organization — everything needed to train or analyze document rotation detection models with your RotDet project.
