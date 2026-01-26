from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from .base import RotDetModel
from .c4net import C4Net
from .simple_cnn import SimpleCNN
from .tiny import RotDetTiny, RotDetTinyBN

ModelFactory = Callable[..., RotDetModel]
RegistryEntry = Tuple[ModelFactory, Dict[str, Any]]

_MODEL_REGISTRY: Dict[str, RegistryEntry] = {
    "simple_cnn": (SimpleCNN, {"in_ch": 1, "num_classes": 4}),
    "rotdet_tiny": (RotDetTiny, {"in_ch": 1, "num_classes": 4}),
    "rotdet_tiny_bn": (RotDetTinyBN, {"in_ch": 1, "num_classes": 4}),
    "c4net": (
        C4Net,
        {
            "in_ch": 1,
            "stem_ch": 16,
            "widths": (32, 64, 128),
            "num_classes": 4,
            "head_type": "equivariant",
        },
    ),
    "c4neti": (
        C4Net,
        {
            "in_ch": 1,
            "stem_ch": 16,
            "widths": (32, 64, 128),
            "num_classes": 4,
            "head_type": "invariant",
        },
    ),
    "c4neto": (
        C4Net,
        {
            "in_ch": 1,
            "stem_ch": 16,
            "widths": (32, 64, 128),
            "num_classes": 4,
            "head_type": "orientation",
        },
    ),
}


def available_model_names() -> list[str]:
    """Return the list of registered model identifiers."""

    return sorted(_MODEL_REGISTRY.keys())


def default_model_kwargs(name: str) -> Dict[str, Any]:
    """Return the default keyword arguments for a registered model."""

    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'")
    return dict(_MODEL_REGISTRY[name][1])


def _coerce_tuple_overrides(
    defaults: Dict[str, Any], params: Dict[str, Any]
) -> Dict[str, Any]:
    """Coerce list overrides into tuples when defaults declare tuple values."""

    coerced = dict(params)
    for key, default_value in defaults.items():
        if isinstance(default_value, tuple) and isinstance(coerced.get(key), list):
            coerced[key] = tuple(coerced[key])
    return coerced


def create_model(name: str, **overrides: Any) -> RotDetModel:
    """Instantiate a registered model, applying optional overrides."""

    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'")
    factory, defaults = _MODEL_REGISTRY[name]
    params = {**defaults, **overrides}
    params = _coerce_tuple_overrides(defaults, params)
    return factory(**params)


__all__ = [
    "RotDetModel",
    "SimpleCNN",
    "RotDetTiny",
    "RotDetTinyBN",
    "C4Net",
    "available_model_names",
    "create_model",
    "default_model_kwargs",
]
