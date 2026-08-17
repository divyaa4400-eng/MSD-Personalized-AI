from __future__ import annotations

import time
from copy import deepcopy
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .evaluation import compute_metrics, predict_model
from .models import MultiModalMSDNet
from .preprocessing import MSDDataset, PreprocessingState
from .synthetic_data import MODALITY_NAMES, SyntheticCohort


def _config_forced_mask(config: Dict, n_samples: int) -> Optional[np.ndarray]:
    data_cfg = config.get("data", {})
    active = data_cfg.get("active_modalities")
    mask = np.ones((n_samples, len(MODALITY_NAMES)), dtype=np.float32)
    changed = False
    if active is not None:
        mask[:] = 0.0
        for name in active:
            if name not in MODALITY_NAMES:
                raise ValueError(f"Unknown active modality: {name}")
            mask[:, MODALITY_NAMES.index(name)] = 1.0
        changed = True
    if bool(data_cfg.get("disable_clinical", False)):
        mask[:, MODALITY_NAMES.index("clinical")] = 0.0
        changed = True
    if bool(data_cfg.get("disable_biomechanics", False)):
        mask[:, MODALITY_NAMES.index("biomechanical")] = 0.0
        changed = True
    if changed and np.any(mask.sum(axis=1) == 0):
        raise ValueError("Configuration disables all modalities")
    return mask if changed else None


def make_loader(
    cohort: SyntheticCohort,
    indices: Sequence[int],
    state: PreprocessingState,
    config: Dict,
    training: bool,
    forced_mask: Optional[np.ndarray] = None,
    image_noise_std: float = 0.0,
    structured_noise_std: float = 0.0,
) -> DataLoader:
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    if forced_mask is None:
        forced_mask = _config_forced_mask(config, len(indices))
    dataset = MSDDataset(
        cohort,
        indices,
        state,
        training=training,
        modality_dropout=float(data_cfg.get("modality_dropout", 0.0)) if training else 0.0,
        seed=int(config.get("seed", 0)),
        forced_mask=forced_mask,
        image_noise_std=image_noise_std,
        structured_noise_std=structured_noise_std,
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=training,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )


def _batch_loss(out, batch, config: Dict, device: torch.device):
    train_cfg = config.get("training", {})
    risk_true = batch["risk"].to(device)
    labels = batch["label"].to(device)
    uncertainty = batch["uncertainty"].to(device)

    sq = (out["risk"] - risk_true) ** 2
    if bool(train_cfg.get("uncertainty_weighting", True)):
        # Borderline/high-disagreement samples receive a bounded extra weight.
        w = 1.0 + uncertainty / uncertainty.mean().clamp_min(1e-6)
        reg_loss = (w * sq).mean()
    else:
        reg_loss = sq.mean()
    cls_loss = nn.functional.cross_entropy(out["logits"], labels)
    total = float(train_cfg.get("regression_weight", 1.0)) * reg_loss + float(train_cfg.get("classification_weight", 1.0)) * cls_loss
    return total, reg_loss.detach(), cls_loss.detach()


def train_model(
    cohort: SyntheticCohort,
    train_indices: Sequence[int],
    state: PreprocessingState,
    config: Dict,
    device: torch.device,
    val_indices: Optional[Sequence[int]] = None,
    epochs: Optional[int] = None,
) -> Tuple[MultiModalMSDNet, Dict]:
    model = MultiModalMSDNet(config, cohort.clinical.shape[1], cohort.biomechanical.shape[1]).to(device)
    train_loader = make_loader(cohort, train_indices, state, config, training=True)
    val_loader = make_loader(cohort, val_indices, state, config, training=False) if val_indices is not None else None

    train_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    max_epochs = int(epochs if epochs is not None else train_cfg.get("epochs", 16))
    patience = int(train_cfg.get("patience", 4))

    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    history = []
    start = time.perf_counter()

    for epoch in range(max_epochs):
        model.train()
        totals = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch["mri"].to(device),
                batch["ct"].to(device),
                batch["xray"].to(device),
                batch["clinical"].to(device),
                batch["biomechanical"].to(device),
                batch["modality_mask"].to(device),
            )
            loss, _, _ = _batch_loss(out, batch, config, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            totals.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(totals))
        val_loss = train_loss
        val_metrics = None
        if val_loader is not None:
            pred = predict_model(model, val_loader, device)
            val_metrics = compute_metrics(pred["label"], pred["risk_true"], pred["probs"], pred["risk_pred"])
            val_loss = val_metrics["mse"] + (1.0 - val_metrics["f1_macro"])

        scheduler.step(val_loss)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "selection_loss": float(val_loss),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **({f"val_{k}": v for k, v in val_metrics.items()} if val_metrics else {}),
        })

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if val_loader is not None and stale >= patience:
            break

    model.load_state_dict(best_state)
    return model, {
        "history": history,
        "best_selection_loss": best_loss,
        "training_seconds": time.perf_counter() - start,
        "epochs_completed": len(history),
    }
