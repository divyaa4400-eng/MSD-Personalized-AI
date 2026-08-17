from __future__ import annotations

import torch
from torch import nn


class GatedAttentionFusion(nn.Module):
    def __init__(self, dim: int, n_modalities: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        self.context = nn.Sequential(
            nn.Linear(dim * n_modalities, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
        )
        self.h_proj = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.score = nn.Linear(dim, 1, bias=False)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor):
        # embeddings: [B, M, D], mask: [B, M]
        masked = embeddings * mask.unsqueeze(-1)
        context = self.context(masked.flatten(1))
        energy = self.score(torch.tanh(self.h_proj(embeddings) + self.c_proj(context).unsqueeze(1))).squeeze(-1)
        energy = energy.masked_fill(mask <= 0, -1e9)
        attention = torch.softmax(energy, dim=1)
        fused = torch.sum(attention.unsqueeze(-1) * embeddings, dim=1)
        fused = self.dropout(fused * self.gate(context) + context)
        return fused, attention


class MeanFusion(nn.Module):
    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor):
        weights = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return fused, weights


class ConcatFusion(nn.Module):
    def __init__(self, dim: int, n_modalities: int = 5, dropout: float = 0.1) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(dim * n_modalities, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor):
        masked = embeddings * mask.unsqueeze(-1)
        fused = self.project(masked.flatten(1))
        weights = mask / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return fused, weights
