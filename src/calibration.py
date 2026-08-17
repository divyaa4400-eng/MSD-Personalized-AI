from __future__ import annotations

from typing import Dict

import numpy as np

from .evaluation import expected_calibration_error, multiclass_brier


def reliability_bins(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict[str, np.ndarray]:
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi if hi < 1 else confidence <= hi)
        centers.append((lo + hi) / 2)
        counts.append(int(m.sum()))
        accs.append(float(correct[m].mean()) if m.any() else np.nan)
        confs.append(float(confidence[m].mean()) if m.any() else np.nan)
    return {
        "bin_center": np.asarray(centers),
        "accuracy": np.asarray(accs),
        "confidence": np.asarray(confs),
        "count": np.asarray(counts),
    }


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    return {
        "brier_multiclass": multiclass_brier(y_true, probs, probs.shape[1]),
        "ece": expected_calibration_error(y_true, probs, n_bins),
    }
