# RotDet v2

Compact four-way document orientation detection with a C4-equivariant CNN.
Two separately trained variants accept **256x256** or **384x384** grayscale
inputs. Both checkpoints are approximately **1.56 MB**; 256 is the default
cost-oriented choice, while 384 performed better on development validation.

The implementation is corrected C4Net, not the experimental C4NetV2 class.
Weights are published at
[fcrescio/rotdet-v2](https://huggingface.co/fcrescio/rotdet-v2), with pinned
artifact identities in [version identities](docs/VERSIONS.md).
The old binary SimpleCNN remains
available under the `v1.0` tag and the unchanged
[v1 Hub repository](https://huggingface.co/fcrescio/rotdet).

## Install and run

Python 3.11 or newer. For a CPU-only PyTorch installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install .
rotdet --model /path/to/256 --threads 4 page.png
```

The variant directory must contain `model.safetensors` and `config.json`.
Weights are not committed to Git. The loader verifies the SHA-256 before
loading and uses the exact architecture and resolution in the configuration.

```python
from pathlib import Path
import torch
from rotdet import Detector

torch.set_num_threads(4)  # The benchmark uses four CPU threads.
detector = Detector("/path/to/256")
result = detector.predict(Path("page.png").read_bytes())
print(result["correction_cw_degrees"])
```

For Hub downloads, install `.[hub]` and use:

```python
detector = Detector.from_pretrained(
    256, revision="6662ebee315bb481d2919b30922d42356a8361ab")
```

Use `384` for the larger input variant. An explicit revision is required
for reproducible artifact selection.

## Output contract

| Class | Observed orientation | Clockwise correction |
|---|---|---|
| 0 | Upright | 0 degrees |
| 1 | 90 degrees counterclockwise | 90 degrees |
| 2 | 180 degrees | 180 degrees |
| 3 | 270 degrees counterclockwise | 270 degrees |

The API returns the class, orientation, correction, softmax confidence and
four probabilities. Confidence is **not calibrated**; no automatic abstention
threshold is provided. The detector never modifies or rotates input files.
It handles images, not PDF decoding or fine-angle deskewing.

## Evidence and limitations

On 164 provisionally LLM-labeled independent pages, accuracy was 90.85%
(256), 89.63% (384), and 93.90% (Paddle PP-LCNet_x1_0_doc_ori). Measured
CPU batch1 medians were 29.4, 49.0 and 74.2 ms respectively. **These are
not human-gold accuracy estimates or universal performance guarantees.**
Paired uncertainty does not establish superiority or equivalence.

See [benchmark details](docs/BENCHMARK.md),
[training provenance and recipe](docs/DATA_PROVENANCE.md), and
[Archive.org source references](docs/training_sources.json).
Datasets are deliberately not redistributed. Source licenses were not
specified; source citations do not grant reuse permission.

## Development

```bash
pip install '.[test]'
python -m pytest tests
```

Tests cover quarter-turn equivariance, orientation-channel layout,
preprocessing, strict checkpoint loading and corruption rejection at both
resolutions. Public checkpoint parity is additionally verified against the
frozen lab candidates; verification does not retrain or rescore the benchmark.

Legacy evaluation/training scripts remain under `scripts/`, with their
[original documentation](docs/LEGACY_README.md). They are not the v2 training
entrypoint. Install `.[legacy]` only when working on those historical tools.

## License

Code: MIT, as already declared by this project. Source documents are not
covered by that license. No explicit weight license has been assigned yet;
publication alone does not grant redistribution permission. Weight licensing
is stated separately in the Hub model card.
