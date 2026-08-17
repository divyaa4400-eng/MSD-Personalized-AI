from __future__ import annotations

import math
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from src.evaluation import compute_metrics, predict_model
from src.experiment import development_test_split
from src.models import MultiModalMSDNet
from src.preprocessing import fit_preprocessing
from src.synthetic_data import MODALITY_NAMES, generate_synthetic_cohort
from src.training import make_loader, train_model
from src.utils import load_yaml, resolve_device, seed_everything


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)
    print(f"✓ {message}")


def main():
    cfg = load_yaml("configs/default.yaml")
    cfg = deepcopy(cfg)
    cfg["data"]["n_subjects"] = 120
    cfg["data"]["image_size"] = 16
    cfg["training"]["epochs"] = 1
    cfg["training"]["final_epochs"] = 1
    cfg["training"]["batch_size"] = 24
    cfg["training"]["patience"] = 1
    cfg["explainability"]["permutation_repeats"] = 1
    cfg["runtime"]["device"] = "cpu"

    seed = int(cfg["seed"])
    seed_everything(seed)
    a = generate_synthetic_cohort(cfg)
    seed_everything(seed)
    b = generate_synthetic_cohort(cfg)

    check(len(a.subject_id) == 120, "configured sample count generated")
    check(len(set(a.subject_id.tolist())) == len(a.subject_id), "subject identifiers are unique")
    check(np.allclose(a.risk_score, b.risk_score), "synthetic targets are deterministic for a fixed seed")
    check(np.allclose(a.mri, b.mri), "synthetic imaging is deterministic for a fixed seed")
    check(a.mri.shape == (120, 1, 16, 16), "MRI tensor shape is valid")
    check(a.ct.shape == a.mri.shape and a.xray.shape == a.mri.shape, "all imaging modalities have consistent dimensions")
    check(np.all((a.risk_score >= 0) & (a.risk_score <= 1)), "continuous risk target is bounded")
    check(set(np.unique(a.risk_class)).issubset({0, 1, 2}), "risk classes are valid")

    dev, test = development_test_split(a, cfg)
    check(len(set(dev).intersection(set(test))) == 0, "development and independent test partitions do not overlap")
    check(len(dev) + len(test) == 120, "split covers the complete cohort")

    state = fit_preprocessing(a, dev)
    loader = make_loader(a, test, state, cfg, training=False)
    batch = next(iter(loader))
    check(batch["modality_mask"].shape[1] == len(MODALITY_NAMES), "modality mask contains all five streams")

    device = resolve_device("cpu")
    model = MultiModalMSDNet(cfg, a.clinical.shape[1], a.biomechanical.shape[1]).to(device)
    out = model(
        batch["mri"], batch["ct"], batch["xray"], batch["clinical"], batch["biomechanical"], batch["modality_mask"]
    )
    check(out["risk"].shape[0] == len(batch["risk"]), "model regression head produces one risk value per subject")
    check(out["logits"].shape[1] == 3, "model classification head produces three risk classes")
    check(torch.allclose(out["attention"].sum(dim=1), torch.ones(len(batch["risk"])), atol=1e-5), "attention weights sum to one")

    model, info = train_model(a, dev, state, cfg, device, val_indices=None, epochs=1)
    pred = predict_model(model, loader, device)
    metrics = compute_metrics(pred["label"], pred["risk_true"], pred["probs"], pred["risk_pred"])
    check(all(np.isfinite(v) for v in metrics.values()), "evaluation metrics are finite")
    check(info["epochs_completed"] == 1, "training loop completes a configured epoch")

    print("\nREPRODUCIBILITY CHECK: PASSED")


if __name__ == "__main__":
    main()
