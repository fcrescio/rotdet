# Licensing scope

Copyright (c) 2026 Francesco Crescioli.

From release v2.0.1, RotDet v2 code and the two frozen checkpoints are
offered under the GNU General Public License, version 3 only
(`GPL-3.0-only`). The complete license text is in the root LICENSE file
and accompanies the weights on Hugging Face.

This program and the checkpoints are free software: you can redistribute
them and/or modify them under the terms of the GNU General Public License
as published by the Free Software Foundation, version 3 of the License.
They are distributed in the hope that they will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the license for the full terms.

## Checkpoints covered

- 256: SHA-256
  `e67fa9560d45ea3ddabbac511e743affb99ef6fe403f91ab299ced08523aa3bb`.
- 384: SHA-256
  `5688a435babf46543ec827303864d37741940ad8b5e5b40b5b5c14008fa9f12b`.

The grant identifies the actual checkpoint bytes, including downloads from
the earlier frozen revision. It does not cover other models or datasets.
Architecture source, inference implementation and checkpoint configuration
are public in this repository; the training recipe and source citations
are in docs/DATA_PROVENANCE.md and docs/training_sources.json. This is not
a claim that the training dataset is redistributable or that the private
benchmark is available.

## Historical notices and third-party material

- Previously released code, including the immutable v2.0 tag, was offered
  under MIT. Those grants are not withdrawn. The prior notice is retained
  at docs/licenses/MIT-previous-releases.txt; it also applies to the
  unchanged legacy scripts in scripts/.
- The v1 SimpleCNN weights remain under their existing CC-BY-4.0 license
  in the original Hugging Face repository, which is not modified.
- Dependencies retain their own licenses.
- Original documents, scans, OCR, private evaluation labels and datasets
  are excluded. Citing Archive.org does not license those materials.

GPL is a copyleft license, not a noncommercial license, and this release
does not add a noncommercial condition or an AGPL network-use provision.
The scope of GPL obligations for a particular ML integration should not
be inferred from the model card alone; consult the complete license.
