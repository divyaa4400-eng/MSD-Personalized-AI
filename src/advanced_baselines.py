from __future__ import annotations

from copy import deepcopy
from typing import Dict, Sequence, Tuple

import torch
from torch import nn

from .training import _batch_loss, make_loader


class ImagingBackboneBaseline(nn.Module):
    """Three-modality imaging baseline using torchvision backbones.

    MRI, CT and X-ray surrogates are stacked as three input channels. Structured
    features are intentionally ignored for a clean imaging-only comparison.
    """

    def __init__(self, name: str, n_classes: int = 3):
        super().__init__()
        try:
            from torchvision import models
        except Exception as exc:
            raise RuntimeError("torchvision is required for advanced imaging baselines") from exc

        name = name.lower()
        if name == "resnet101":
            backbone = models.resnet101(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif name == "densenet121":
            backbone = models.densenet121(weights=None)
            feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
        elif name in {"vit", "vit_b_16", "vision_transformer"}:
            backbone = models.vit_b_16(weights=None, image_size=32)
            feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
        else:
            raise ValueError(f"Unsupported advanced baseline: {name}")
        self.backbone = backbone
        self.risk_head = nn.Sequential(nn.Linear(feature_dim, 1), nn.Sigmoid())
        self.class_head = nn.Linear(feature_dim, n_classes)

    def forward(self, mri, ct, xray, clinical, biomechanical, modality_mask):
        x = torch.cat([mri, ct, xray], dim=1)
        if x.shape[-1] != 32 or x.shape[-2] != 32:
            x = nn.functional.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        h = self.backbone(x)
        risk = self.risk_head(h).squeeze(-1)
        logits = self.class_head(h)
        # Uniform imaging-only attention for compatibility with common evaluators.
        att = torch.zeros((x.shape[0], 5), device=x.device, dtype=x.dtype)
        att[:, :3] = 1.0 / 3.0
        return {"risk": risk, "logits": logits, "attention": att, "embedding": h}


def train_advanced_baseline(cohort, train_indices: Sequence[int], state, config: Dict, device: torch.device, name: str):
    model = ImagingBackboneBaseline(name, int(config.get("model", {}).get("n_classes", 3))).to(device)
    loader = make_loader(cohort, train_indices, state, config, training=True)
    train_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("advanced_baseline_epochs", 6))
    model.train()
    for _ in range(epochs):
        for batch in loader:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model
