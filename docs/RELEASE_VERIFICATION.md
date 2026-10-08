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

This verifies the prepared local public package, not downloaded Hub
artifacts. Hugging Face creation was attempted with the available token and
returned HTTP 403 (read-only authentication). Hub upload and subsequent
download verification remain pending. Linux x86_64 CPU was tested; other
platforms were not.

The available GitHub token does not have the `workflow` scope. The tested
CPU workflow is supplied as `tools/cpu-tests.workflow.yml`, not activated
under `.github/workflows/`. No successful GitHub Actions run is claimed.

The verification entrypoint is `tools/verify_release.py`. Its fixture and
lock inputs are deliberately outside Git because the fixture images are
not cleared for redistribution. Public unit tests need no private inputs.
