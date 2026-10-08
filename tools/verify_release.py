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
    args = parser.parse_args()
    env = {k: v for k, v in os.environ.items()
           if k not in {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LD_PRELOAD"}}
    with tempfile.TemporaryDirectory(prefix="rotdet-public-") as tmp:
        root = Path(tmp)
        venv.EnvBuilder(with_pip=True).create(root / "venv")
        python = str(root / "venv/bin/python")
        source = root / "source"
        shutil.copytree(Path(__file__).resolve().parents[1], source,
                        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.egg-info"))

        def run(*command):
            subprocess.run(command, env=env, cwd=root, check=True, timeout=900)

        run(python, "-c", "import importlib.util; assert importlib.util.find_spec('torch') is None")
        run(python, "-m", "pip", "install", "--upgrade", "pip")
        run(python, "-m", "pip", "install", "torch", "--index-url",
            "https://download.pytorch.org/whl/cpu")
        run(python, "-m", "pip", "install", f"{source}[test]")
        run(python, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(source / "tests"))
        code = '''
import hashlib, json, sys, torch
from pathlib import Path
from rotdet import Detector
torch.set_num_threads(4)
artifacts, lock_path, fixtures = map(Path, sys.argv[1:])
lock = json.loads(lock_path.read_text())
results = {}
for size in (256, 384):
    detector = Detector(artifacts / str(size))
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
                         "trainable_parameters": sum(p.numel() for p in detector.model.parameters()),
                         "torch": torch.__version__}
print(json.dumps(results, sort_keys=True))
'''
        output = subprocess.check_output([python, "-c", code, str(args.artifacts),
                                          str(args.lock), str(args.fixtures)],
                                         env=env, cwd=root, timeout=120, text=True)
        results = json.loads(output)
        run(str(root / "venv/bin/rotdet"), "--model", str(args.artifacts / "256"),
            str(args.fixtures / "f00.bin"))
        run(str(root / "venv/bin/rotdet"), "--model", str(args.artifacts / "384"),
            str(args.fixtures / "f00.bin"))
        print(json.dumps({"clean_cpu_install": True, "variants": results}, indent=2))


if __name__ == "__main__":
    main()
