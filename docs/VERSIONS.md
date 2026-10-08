# Version and artifact identities

## v1.0

The annotated Git tag points at `e2dc056`, the initial implementation
compatible with the currently published binary SimpleCNN. Strict loading of
the downloaded weights was verified, with 276,290 parameters and `(1, 2)`
output for grayscale 128x128 input. This is a compatible implementation,
not a recovered original training commit.

- Hub repository: https://huggingface.co/fcrescio/rotdet
- Hub revision: `234b591413e3dcaa9477a0de9a0bd28166440fe4`
- Weight SHA-256: `04a065668430d50830c3f5cf14e9535242cb5e7e0faa9f4a90c86ccd351871af`
- Existing published weight license: CC-BY-4.0.

## v2

Public release name: RotDet v2. Architecture: corrected **C4Net**, not the
experimental `C4NetV2` registry entry. Both input resolutions use the same
architecture but separately trained weights, with training seed 42 selected
before independent evaluation. Checkpoint hashes and exact configurations
are in `models/256/config.json` and `models/384/config.json`.

The frozen model source came from the independent recovery lab; private
experiment artifacts and dataset images are deliberately not copied into
this repository. The existing `scripts/` files are legacy experiments and
are not a recipe for reproducing the v2 checkpoints.

The binary v1 API and weights are incompatible with the four-class v2 API.
Use the v1.0 tag when reproducing the old model. The new `rotdet` package
does not silently load old checkpoints or auto-rotate source files.

Hub publication and a final v2.0 release tag follow verification of the
actually downloaded published weights. Until then, the GitHub code is a
release candidate, not evidence that weights are available on the Hub.
