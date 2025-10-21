# rotdet_hf.py
from __future__ import annotations
from typing import Optional
from datasets import load_dataset, get_dataset_config_names

PREFERRED_CONFIGS = ["images", "zero-shot-exp"]

def pick_config(dataset_id: str, requested: Optional[str] = None) -> Optional[str]:
    """
    If `requested` is provided and exists -> use it.
    Otherwise prefer 'images', then 'zero-shot-exp', else first available.
    Returns None if the dataset has no configs.
    """
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception:
        # dataset may have no named configs
        return requested  # could still be None

    if not configs:
        return None

    if requested:
        if requested in configs:
            return requested
        # fall back if the requested one doesn't exist
        # (you might prefer to raise instead)
    for cand in PREFERRED_CONFIGS:
        if cand in configs:
            return cand
    return configs[0]  # last resort

def load_hf_dataset(
    dataset_id: str,
    split: str,
    config: Optional[str] = None,
    streaming: bool = False,
):
    """
    Loads the dataset with a sensible config:
      - if `config` is provided and valid, use it
      - else automatically choose a preferred available config
    """
    cfg = pick_config(dataset_id, requested=config)
    kwargs = dict(split=split, streaming=streaming)
    if cfg is not None:
        return load_dataset(dataset_id, cfg, **kwargs)
    return load_dataset(dataset_id, **kwargs)
