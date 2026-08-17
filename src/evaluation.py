from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, n_classes: int = 3) -> float:
    onehot = np.eye(n_classes, dtype=float)[y_true.astype(int)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi if hi < 1 else confidence <= hi)
        if np.any(in_bin):
            ece += np.mean(in_bin) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def compute_metrics(
    y_true: np.ndarray,
    risk_true: np.ndarray,
    probs: np.ndarray,
    risk_pred: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    y_pred = probs.argmax(axis=1)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "mse": float(mean_squared_error(risk_true, risk_pred)),
        "mae": float(mean_absolute_error(risk_true, risk_pred)),
        "brier_multiclass": multiclass_brier(y_true, probs, probs.shape[1]),
        "ece": expected_calibration_error(y_true, probs, n_bins),
    }
    if np.std(risk_true) > 0 and np.std(risk_pred) > 0:
        out["pearson_r"] = float(np.corrcoef(risk_true, risk_pred)[0, 1])
    else:
        out["pearson_r"] = 0.0

    try:
        out["auc_macro_ovr"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        out["auc_macro_ovr"] = float("nan")

    high_true = (y_true == probs.shape[1] - 1).astype(int)
    high_prob = probs[:, -1]
    try:
        out["auc_high_risk"] = float(roc_auc_score(high_true, high_prob))
        out["pr_auc_high_risk"] = float(average_precision_score(high_true, high_prob))
        out["brier_high_risk"] = float(brier_score_loss(high_true, high_prob))
    except ValueError:
        out["auc_high_risk"] = float("nan")
        out["pr_auc_high_risk"] = float("nan")
        out["brier_high_risk"] = float("nan")
    return out


@torch.no_grad()
def predict_model(model: torch.nn.Module, loader: Iterable, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    risks, labels, risk_true, probs, attention, indices = [], [], [], [], [], []
    for batch in loader:
        inputs = {
            "mri": batch["mri"].to(device),
            "ct": batch["ct"].to(device),
            "xray": batch["xray"].to(device),
            "clinical": batch["clinical"].to(device),
            "biomechanical": batch["biomechanical"].to(device),
            "modality_mask": batch["modality_mask"].to(device),
        }
        out = model(**inputs)
        risks.append(out["risk"].cpu().numpy())
        probs.append(torch.softmax(out["logits"], dim=1).cpu().numpy())
        attention.append(out["attention"].cpu().numpy())
        labels.append(batch["label"].numpy())
        risk_true.append(batch["risk"].numpy())
        indices.append(batch["index"].numpy())
    return {
        "risk_pred": np.concatenate(risks),
        "probs": np.concatenate(probs),
        "attention": np.concatenate(attention),
        "label": np.concatenate(labels),
        "risk_true": np.concatenate(risk_true),
        "index": np.concatenate(indices),
    }
