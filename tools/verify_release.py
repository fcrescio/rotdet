# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Francesco Crescioli
"""Install the public package in a new CPU-only venv and verify frozen parity.

This is artifact verification, not another accuracy evaluation. Fixtures stay
outside the repository. Run with --artifacts DIR --lock FILE --fixtures DIR.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--hub-revision", help="Verify anonymous downloads from this Hub commit")
    args = parser.parse_args()
    env = {k: v for k, v in os.environ.items()
           if k not in {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}}
    with tempfile.TemporaryDirectory(prefix="rotdet-public-") as tmp:
        root = Path(tmp)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = str(root / "venv/bin/python")
        source = root / "source"
        env["HF_HOME"] = str(root / "hub-cache")
        env["HF_HUB_OFFLINE"] = "0"
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        shutil.copytree(Path(__file__).resolve().parents[1], source,
                        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.egg-info"))

        def run(*command):
            subprocess.run(command, env=env, cwd=root, check=True, timeout=900)

        run(python, "-c", "import importlib.util; assert importlib.util.find_spec('torch') is None")
        run(python, "-m", "pip", "install", "--upgrade", "pip")
        run(python, "-m", "pip", "install", "torch", "--index-url",
            "https://download.pytorch.org/whl/cpu")
        run(python, "-m", "pip", "install", f"{source}[test,hub]")
        run(python, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(source / "tests"))
        code = '''
import hashlib, importlib.metadata, json, sys, torch
from pathlib import Path
from rotdet import Detector
torch.set_num_threads(4)
artifacts, lock_path, fixtures = map(Path, sys.argv[1:4])
hub_revision = sys.argv[4]
lock = json.loads(lock_path.read_text())
results = {}
assert importlib.metadata.version("rotdet") == "2.0.1"
assert importlib.metadata.metadata("rotdet")["License"] == "GPL-3.0-only"
license_files = importlib.metadata.metadata("rotdet").get_all("License-File")
assert any(p.endswith("LICENSE") for p in license_files)
assert any(p.endswith("MIT-previous-releases.txt") for p in license_files)
for size in (256, 384):
    directory = artifacts / str(size)
    if hub_revision:
        from huggingface_hub import snapshot_download
        directory = Path(snapshot_download("fcrescio/rotdet-v2",
                         revision=hub_revision, allow_patterns=[f"{size}/*"])) / str(size)
        metadata = json.loads((directory / "config.json").read_text())
        assert metadata["sha256"] == lock["candidates"]["c4net_" + str(size)]["sha256"]
        detector = Detector.from_pretrained(size, revision=hub_revision)
    else:
        detector = Detector(directory)
    checks = 0
    for name, fixture in lock["parity_fixtures"]["pages"].items():
        data = (fixtures / (name + ".bin")).read_bytes()
        assert hashlib.sha256(data).hexdigest() == fixture["bytes_sha256"]
        expected = fixture["expected"]["c4net_" + str(size)]
        for k in range(4):
            result = detector.predict(data, k)
            assert result["class"] == expected[str(k)], (size, name, k, result)
            checks += 1
    results[str(size)] = {"frozen_parity_views": checks,
                         "directory": str(directory),
                         "trainable_parameters": sum(p.numel() for p in detector.model.parameters()),
                         "torch": torch.__version__}
print(json.dumps(results, sort_keys=True))
'''
        output = subprocess.check_output([python, "-c", code, str(args.artifacts),
                                          str(args.lock), str(args.fixtures),
                                          args.hub_revision or ""],
                                         env=env, cwd=root, timeout=120, text=True)
        results = json.loads(output)
        run(str(root / "venv/bin/rotdet"), "--model", results["256"]["directory"],
            str(args.fixtures / "f00.bin"))
        run(str(root / "venv/bin/rotdet"), "--model", results["384"]["directory"],
            str(args.fixtures / "f00.bin"))
        print(json.dumps({"clean_cpu_install": True, "hub_revision": args.hub_revision,
                          "variants": results}, indent=2))


if __name__ == "__main__":
    main()
