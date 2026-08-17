from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .evaluation import compute_metrics, predict_model
from .preprocessing import fit_preprocessing
from .statistics import fold_summary
from .synthetic_data import SyntheticCohort
from .training import make_loader, train_model
from .utils import save_json


def development_test_split(cohort: SyntheticCohort, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    fraction = float(config.get("data", {}).get("development_fraction", 0.85))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=int(config.get("seed", 0)))
    dev, test = next(splitter.split(np.zeros(len(cohort.risk_class)), cohort.risk_class))
    return np.asarray(dev), np.asarray(test)


def run_cross_validation(
    cohort: SyntheticCohort,
    dev_indices: Sequence[int],
    config: Dict,
    device: torch.device,
) -> Tuple[List[Dict[str, float]], Dict[str, Dict[str, float]]]:
    dev_indices = np.asarray(dev_indices)
    y = cohort.risk_class[dev_indices]
    n_folds = int(config.get("training", {}).get("n_folds", 5))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(config.get("seed", 0)))
    rows: List[Dict[str, float]] = []

    for fold, (tr_local, va_local) in enumerate(skf.split(np.zeros(len(dev_indices)), y), start=1):
        tr = dev_indices[tr_local]
        va = dev_indices[va_local]
        state = fit_preprocessing(cohort, tr)
        model, info = train_model(cohort, tr, state, config, device, val_indices=va)
        loader = make_loader(cohort, va, state, config, training=False)
        pred = predict_model(model, loader, device)
        metrics = compute_metrics(
            pred["label"], pred["risk_true"], pred["probs"], pred["risk_pred"],
            int(config.get("calibration", {}).get("n_bins", 10)),
        )
        rows.append({"fold": fold, **metrics, "epochs_completed": info["epochs_completed"]})

    keys = ["accuracy", "f1_macro", "auc_macro_ovr", "auc_high_risk", "mse", "mae", "pearson_r", "brier_multiclass", "ece"]
    return rows, fold_summary(rows, keys)


def train_final_model(
    cohort: SyntheticCohort,
    dev_indices: Sequence[int],
    config: Dict,
    device: torch.device,
):
    dev_indices = np.asarray(dev_indices)
    state = fit_preprocessing(cohort, dev_indices)
    final_epochs = int(config.get("training", {}).get("final_epochs", config.get("training", {}).get("epochs", 16)))
    model, info = train_model(cohort, dev_indices, state, config, device, val_indices=None, epochs=final_epochs)
    return model, state, info


def evaluate_indices(
    model,
    cohort: SyntheticCohort,
    indices: Sequence[int],
    state,
    config: Dict,
    device: torch.device,
    forced_mask=None,
    image_noise_std: float = 0.0,
    structured_noise_std: float = 0.0,
):
    loader = make_loader(
        cohort,
        indices,
        state,
        config,
        training=False,
        forced_mask=forced_mask,
        image_noise_std=image_noise_std,
        structured_noise_std=structured_noise_std,
    )
    pred = predict_model(model, loader, device)
    metrics = compute_metrics(
        pred["label"], pred["risk_true"], pred["probs"], pred["risk_pred"],
        int(config.get("calibration", {}).get("n_bins", 10)),
    )
    return metrics, pred, loader


def save_split_manifest(cohort: SyntheticCohort, dev: Sequence[int], test: Sequence[int], path: str | Path) -> None:
    save_json({
        "development_indices": list(map(int, dev)),
        "test_indices": list(map(int, test)),
        "development_subject_ids": cohort.subject_id[np.asarray(dev)].tolist(),
        "test_subject_ids": cohort.subject_id[np.asarray(test)].tolist(),
    }, path)
