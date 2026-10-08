# RotDet v2 evaluation

## Independent corpus (provisional annotations)

192 pages from 51 documents; numeric orientation labels on 164 pages
(85.42% coverage); 28 uncertain pages excluded from accuracy. A vision LLM
provided labels before model scoring; **human gold labels: zero**. The
corpus and both seed42 candidates were frozen before evaluation. Documents
are not redistributed.

| Model | Correct / labeled | Natural orientation accuracy | CPU batch1 median |
|---|---:|---:|---:|
| RotDet v2 256 | 149 / 164 | 90.85% | 29.4 ms |
| RotDet v2 384 | 147 / 164 | 89.63% | 49.0 ms |
| PP-LCNet_x1_0_doc_ori | 154 / 164 | 93.90% | 74.2 ms |

CPU-only, four threads, batch size one, original page bytes, each model's
declared preprocessing; warmup excluded. Paddle was evaluated in a separate
Paddle 3.3.1 environment using its native preprocessing. Timings are
environment-specific, not cross-platform guarantees. RotDet uses PyTorch.

Document-paired bootstrap intervals for model differences include zero:
neither model superiority nor equivalence is established. The two-page
256/384 gap does not establish that 384 failed to generalize.

Both RotDet variants have 1,560,900-byte safetensors files (about 1.56 MB),
versus approximately 6.87 MB for the Paddle reference weights. Model size
does not include framework/runtime memory. Larger inputs cost more compute
and activation memory, not more parameters.

## Document-pure development validation

670 pages / 56 documents, all four artificial quarter-turn rotations,
same frozen split and recipe, three matched seeds (42, 7, 123).

| Resolution | Mean validation accuracy |
|---|---:|
| 128 | 93.22% |
| 256 | 95.76% |
| 384 | 96.70% |

These are development validation results, not independent test accuracy.
The recovery split was document-pure; it must not be confused with the
older legacy document-leaking split. Sources are mostly historical printed
documents; sparse, handwritten, blank and mixed-orientation pages may fail.

## Verification records

- Independent RotDet run: `20261008-021545-733ea7`.
- Paddle completion run: `20261008-094122-677038`.
- Frozen independent lock SHA-256:
  `cecebea8ed4b7db39f81e8b6306d91a14823f716e065e47ec8b6646521723ecc`.
- Earlier standalone 256 clean CPU verification:
  `20261008-095749-c7ab73`. The public package has its own verification.

The private evaluation corpus cannot be downloaded from this repository;
published numbers are transparent experimental results, not a claim that
the complete independent benchmark is publicly reproducible. Source
references and the training protocol are documented separately.
