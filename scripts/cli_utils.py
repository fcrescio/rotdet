from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    return value


def _parse_key_value_pairs(pairs: Iterable[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw_value = pair.partition("=")
        if not sep:
            raise SystemExit(f"Invalid --model-kwarg '{pair}'; expected key=value.")
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --model-kwarg '{pair}'; missing key before '='.")
        overrides[key] = _parse_scalar(raw_value)
    return overrides


def parse_model_kwargs(
    json_arg: str | None,
    json_file: str | None,
    kv_pairs: Iterable[str] | None,
) -> Dict[str, Any] | None:
    overrides: Dict[str, Any] = {}

    if json_file:
        file_path = Path(json_file)
        try:
            payload = json.loads(file_path.read_text())
        except FileNotFoundError as exc:
            raise SystemExit(f"--model-kwargs-file not found: {json_file}") from exc
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON in --model-kwargs-file: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit("--model-kwargs-file must decode to a JSON object")
        overrides.update(payload)

    if json_arg:
        try:
            parsed = json.loads(json_arg)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON for --model-kwargs: {exc}") from exc
        if not isinstance(parsed, dict):
            raise SystemExit("--model-kwargs must decode to a JSON object")
        overrides.update(parsed)

    if kv_pairs:
        overrides.update(_parse_key_value_pairs(kv_pairs))

    return overrides or None
