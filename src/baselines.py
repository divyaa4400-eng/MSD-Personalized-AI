from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .preprocessing import PreprocessingState, transform_structured
from .synthetic_data import SyntheticCohort


def handcrafted_features(cohort: SyntheticCohort, state: PreprocessingState) -> np.ndarray:
    structured = transform_structured(cohort, state)
    image_stats = []
    for arr in (cohort.mri, cohort.ct, cohort.xray):
        flat = arr.reshape(len(arr), -1)
        image_stats.extend([
            flat.mean(axis=1),
            flat.std(axis=1),
            np.quantile(flat, 0.25, axis=1),
            np.quantile(flat, 0.75, axis=1),
        ])
    return np.column_stack(image_stats + [structured["clinical"], structured["biomechanical"]]).astype(np.float32)


def fit_classical_baseline(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict,
):
    cfg = config.get("classical", {})
    if name == "logistic_regression":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(config.get("seed", 0)))
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(cfg.get("random_forest_trees", 300)),
            class_weight="balanced",
            random_state=int(config.get("seed", 0)),
            n_jobs=-1,
        )
    elif name == "svm":
        model = SVC(
            C=float(cfg.get("svm_c", 2.0)),
            gamma=cfg.get("svm_gamma", "scale"),
            probability=True,
            class_weight="balanced",
            random_state=int(config.get("seed", 0)),
        )
    else:
        raise ValueError(f"Unknown classical baseline: {name}")
    model.fit(x_train, y_train)
    return model
