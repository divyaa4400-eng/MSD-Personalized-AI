from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .synthetic_data import MODALITY_NAMES


def build_forced_mask(
    n_samples: int,
    scenario: Dict,
    seed: int = 0,
) -> np.ndarray:
    mask = np.ones((n_samples, len(MODALITY_NAMES)), dtype=np.float32)
    missing = scenario.get("missing_modalities", []) or []
    for name in missing:
        if name not in MODALITY_NAMES:
            raise ValueError(f"Unknown modality in robustness scenario: {name}")
        mask[:, MODALITY_NAMES.index(name)] = 0.0

    rate = float(scenario.get("random_missing_rate", 0.0) or 0.0)
    if rate > 0:
        rng = np.random.default_rng(seed)
        random_drop = rng.random(mask.shape) < rate
        mask[random_drop] = 0.0
        empty = np.where(mask.sum(axis=1) == 0)[0]
        for i in empty:
            mask[i, rng.integers(0, mask.shape[1])] = 1.0
    return mask


def performance_drop(clean: Dict[str, float], corrupted: Dict[str, float]) -> Dict[str, float]:
    keys = ["accuracy", "f1_macro", "auc_macro_ovr", "auc_high_risk"]
    return {f"delta_{k}": float(corrupted.get(k, np.nan) - clean.get(k, np.nan)) for k in keys}
