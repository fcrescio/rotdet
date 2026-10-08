# Public v2 package verification

Verified locally on 2026-10-08 in a fresh virtual environment created
inside a disposable CPU-only Docker container. No inherited site packages;
torch absence checked before installation. Dependencies installed from
public indexes: torch 2.14.1+cpu, numpy 2.5.3, Pillow 12.3.0,
safetensors 0.8.0, pytest 9.1.1. The public package wheel installed
successfully. Library-path and Python-path inheritance were removed.

- 10 focused tests passed, covering both resolutions.
- Frozen checkpoint hashes verified before loading.
- 256: all 40 frozen fixture views matched the predeclared classes.
- 384: all 40 frozen fixture views matched the predeclared classes.
- Fixture bytes checked against frozen SHA-256 records.
- Installed CLI executed successfully for both variants.
- Both variants contain **389,089 trainable parameters**. A previous lab
  metadata count of 389,573 includes state buffers; it is not the parameter
  count.

The first verification attempt installed and tested successfully but failed
at fixture lookup due to an incorrect lock-schema key in the verification
script. This was corrected; a second clean-install attempt completed all
checks. No checkpoint, label, dataset or model selection was changed.

The initial read-only Hugging Face token returned HTTP 403 when creating
the repository. A subsequently supplied write token resolved this blocker;
both checkpoint variants and their configs are now public. Linux x86_64
CPU was tested; other platforms were not.

## Published artifact verification

Verified the public repository `fcrescio/rotdet-v2` at immutable revision
`6662ebee315bb481d2919b30922d42356a8361ab` in another fresh CPU virtual
environment, with an empty Hugging Face cache and no authentication token.
Dependencies included huggingface_hub 2.2.0 and torch 2.14.1+cpu.

- Anonymous snapshot downloads succeeded for both variant directories.
- Public `Detector.from_pretrained` loaded each pinned variant.
- Downloaded config checkpoint hashes matched the original frozen lock.
- The loader verified the downloaded safetensors SHA-256 before loading.
- All 40 frozen fixture views matched for **each** downloaded variant.
- CLI prediction succeeded using the downloaded files for both variants.
- All 10 public unit tests passed in this installed environment.

No training, tuning, label revision or accuracy re-evaluation was performed.
Only weights, configs, a model card and the generated Hub git attributes
were published; no original or derived dataset images were uploaded.

To repeat the external verification, add
`--hub-revision 6662ebee315bb481d2919b30922d42356a8361ab` to the
verification entrypoint's existing fixture/lock arguments.

The available GitHub token does not have the `workflow` scope. The tested
CPU workflow is supplied as `tools/cpu-tests.workflow.yml`, not activated
under `.github/workflows/`. No successful GitHub Actions run is claimed.

The verification entrypoint is `tools/verify_release.py`. Its fixture and
lock inputs are deliberately outside Git because the fixture images are
not cleared for redistribution. Public unit tests need no private inputs.

## v2.0.1 GPL licensing verification

Repeated the clean CPU installation and anonymous Hub checks against GPL
revision `7be7c172c43d0c5a842b8685d4bbd3d7f1567c10`.
The installed distribution reports version 2.0.1 and license
`GPL-3.0-only`; package license metadata includes both the full GPL text
and the retained previous MIT notice. All 10 tests passed, both CLI paths
worked, and all 40 frozen fixture views matched for each downloaded variant.
The complete GPL license copies match the system's canonical GPL-3 text
byte for byte. Hub metadata reports `gpl-3.0` and the NOTICE specifies
version 3 only. Checkpoint bytes, architectures and configs are unchanged.
