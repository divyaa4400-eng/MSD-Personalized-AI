from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .fusion import ConcatFusion, GatedAttentionFusion, MeanFusion


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        return self.activation(x + y if self.enabled else y)


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 64, residual: bool = True) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block1 = ResidualBlock(16, residual)
        self.down1 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block2 = ResidualBlock(32, residual)
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.block3 = ResidualBlock(64, residual)
        self.final_conv = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(64, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = torch.relu(self.final_conv(x))
        return self.project(self.pool(x))


class StructuredEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalMSDNet(nn.Module):
    def __init__(self, config: Dict, clinical_dim: int = 9, biomech_dim: int = 5) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        dim = int(model_cfg.get("fusion_dim", 64))
        hidden = int(model_cfg.get("structured_hidden_dim", 64))
        dropout = float(model_cfg.get("dropout", 0.2))
        residual = bool(model_cfg.get("residual", True))
        n_classes = int(model_cfg.get("n_classes", 3))

        self.mri_encoder = ImageEncoder(dim, residual)
        self.ct_encoder = ImageEncoder(dim, residual)
        self.xray_encoder = ImageEncoder(dim, residual)
        self.clinical_encoder = StructuredEncoder(clinical_dim, dim, hidden, dropout)
        self.biomech_encoder = StructuredEncoder(biomech_dim, dim, hidden, dropout)

        fusion_type = str(model_cfg.get("fusion_type", "gated_attention")).lower()
        if fusion_type == "gated_attention":
            self.fusion = GatedAttentionFusion(dim, 5, dropout)
        elif fusion_type == "mean":
            self.fusion = MeanFusion()
        elif fusion_type == "concat":
            self.fusion = ConcatFusion(dim, 5, dropout)
        else:
            raise ValueError(f"Unsupported fusion_type={fusion_type}")

        self.trunk = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.risk_head = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.class_head = nn.Linear(dim, n_classes)

    def forward(
        self,
        mri: torch.Tensor,
        ct: torch.Tensor,
        xray: torch.Tensor,
        clinical: torch.Tensor,
        biomechanical: torch.Tensor,
        modality_mask: torch.Tensor,
    ):
        embeddings = torch.stack(
            [
                self.mri_encoder(mri),
                self.ct_encoder(ct),
                self.xray_encoder(xray),
                self.clinical_encoder(clinical),
                self.biomech_encoder(biomechanical),
            ],
            dim=1,
        )
        fused, attention = self.fusion(embeddings, modality_mask)
        h = self.trunk(fused)
        risk = self.risk_head(h).squeeze(-1)
        logits = self.class_head(h)
        return {"risk": risk, "logits": logits, "attention": attention, "embedding": h}
