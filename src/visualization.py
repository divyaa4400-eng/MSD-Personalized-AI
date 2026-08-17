from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true, probs, path):
    high = (np.asarray(y_true) == probs.shape[1] - 1).astype(int)
    fpr, tpr, _ = roc_curve(high, probs[:, -1])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, label="High-risk discrimination")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend()
    _save(fig, path)


def plot_pr(y_true, probs, path):
    high = (np.asarray(y_true) == probs.shape[1] - 1).astype(int)
    precision, recall, _ = precision_recall_curve(high, probs[:, -1])
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curve")
    _save(fig, path)


def plot_confusion(y_true, probs, path):
    cm = confusion_matrix(y_true, probs.argmax(axis=1), labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1, 2], labels=["Low", "Moderate", "High"])
    ax.set_yticks([0, 1, 2], labels=["Low", "Moderate", "High"])
    ax.set(xlabel="Predicted", ylabel="Reference", title="Risk-class confusion matrix")
    fig.colorbar(im, ax=ax)
    _save(fig, path)


def plot_reliability(bin_data: Dict[str, np.ndarray], path):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    valid = np.isfinite(bin_data["accuracy"]) & np.isfinite(bin_data["confidence"])
    ax.plot(bin_data["confidence"][valid], bin_data["accuracy"][valid], marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    ax.set(xlabel="Mean confidence", ylabel="Empirical accuracy", title="Reliability diagram", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    _save(fig, path)


def plot_predicted_vs_reference(risk_true, risk_pred, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(risk_true, risk_pred, s=18, alpha=0.65)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set(xlabel="Reference synthetic risk", ylabel="Predicted risk", title="Predicted versus reference risk", xlim=(0, 1), ylim=(0, 1))
    _save(fig, path)


def plot_cv_stability(rows: Sequence[Dict[str, float]], path):
    folds = [int(r["fold"]) for r in rows]
    acc = [r["accuracy"] for r in rows]
    auc = [r.get("auc_macro_ovr", np.nan) for r in rows]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(folds, acc, marker="o", label="Accuracy")
    ax.plot(folds, auc, marker="s", label="Macro AUC")
    ax.set(xlabel="Fold", ylabel="Score", title="Cross-validation stability", ylim=(0, 1))
    ax.legend()
    _save(fig, path)


def plot_gradcam(image: np.ndarray, cam: np.ndarray, path, title: str):
    image = np.squeeze(image)
    cam = np.squeeze(cam)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(image, cmap="gray")
    ax.imshow(cam, alpha=0.45)
    ax.set_title(title)
    ax.axis("off")
    _save(fig, path)
