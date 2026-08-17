from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .synthetic_data import MODALITY_NAMES, SyntheticCohort


@dataclass
class PreprocessingState:
    clinical_median: np.ndarray
    biomech_median: np.ndarray
    clinical_scaler: StandardScaler
    biomech_scaler: StandardScaler


def _fit_imputer_and_scaler(x: np.ndarray, train_indices: np.ndarray):
    train = x[train_indices].copy()
    med = np.nanmedian(train, axis=0)
    train = np.where(np.isnan(train), med[None, :], train)
    scaler = StandardScaler().fit(train)
    return med.astype(np.float32), scaler


def fit_preprocessing(cohort: SyntheticCohort, train_indices: Sequence[int]) -> PreprocessingState:
    train_indices = np.asarray(train_indices, dtype=int)
    c_med, c_scaler = _fit_imputer_and_scaler(cohort.clinical, train_indices)
    b_med, b_scaler = _fit_imputer_and_scaler(cohort.biomechanical, train_indices)
    return PreprocessingState(c_med, b_med, c_scaler, b_scaler)


def transform_structured(cohort: SyntheticCohort, state: PreprocessingState) -> Dict[str, np.ndarray]:
    clinical = np.where(np.isnan(cohort.clinical), state.clinical_median[None, :], cohort.clinical)
    biomech = np.where(np.isnan(cohort.biomechanical), state.biomech_median[None, :], cohort.biomechanical)
    clinical = state.clinical_scaler.transform(clinical).astype(np.float32)
    biomech = state.biomech_scaler.transform(biomech).astype(np.float32)
    return {"clinical": clinical, "biomechanical": biomech}


class MSDDataset(Dataset):
    def __init__(
        self,
        cohort: SyntheticCohort,
        indices: Sequence[int],
        state: PreprocessingState,
        training: bool = False,
        modality_dropout: float = 0.0,
        seed: int = 0,
        forced_mask: Optional[np.ndarray] = None,
        image_noise_std: float = 0.0,
        structured_noise_std: float = 0.0,
    ) -> None:
        self.cohort = cohort
        self.indices = np.asarray(indices, dtype=int)
        self.structured = transform_structured(cohort, state)
        self.training = training
        self.modality_dropout = float(modality_dropout)
        self.seed = int(seed)
        self.forced_mask = forced_mask
        self.image_noise_std = float(image_noise_std)
        self.structured_noise_std = float(structured_noise_std)

    def __len__(self) -> int:
        return len(self.indices)

    def _mask(self, local_idx: int) -> np.ndarray:
        if self.forced_mask is not None:
            mask = self.forced_mask[local_idx].astype(np.float32).copy()
        else:
            mask = np.ones(len(MODALITY_NAMES), dtype=np.float32)
        if self.training and self.modality_dropout > 0:
            rng = np.random.default_rng(self.seed + int(self.indices[local_idx]) * 7919)
            dropped = rng.random(len(MODALITY_NAMES)) < self.modality_dropout
            if dropped.all():
                dropped[rng.integers(0, len(MODALITY_NAMES))] = False
            mask[dropped] = 0.0
        if mask.sum() == 0:
            mask[0] = 1.0
        return mask

    def __getitem__(self, local_idx: int) -> Dict[str, torch.Tensor]:
        i = int(self.indices[local_idx])
        mri = self.cohort.mri[i].copy()
        ct = self.cohort.ct[i].copy()
        xray = self.cohort.xray[i].copy()
        clinical = self.structured["clinical"][i].copy()
        biomech = self.structured["biomechanical"][i].copy()

        if self.image_noise_std > 0 or self.structured_noise_std > 0:
            rng = np.random.default_rng(self.seed + i * 1543 + 17)
            if self.image_noise_std > 0:
                for arr in (mri, ct, xray):
                    arr += rng.normal(0, self.image_noise_std, arr.shape).astype(np.float32)
                    np.clip(arr, 0, 1, out=arr)
            if self.structured_noise_std > 0:
                clinical += rng.normal(0, self.structured_noise_std, clinical.shape).astype(np.float32)
                biomech += rng.normal(0, self.structured_noise_std, biomech.shape).astype(np.float32)

        mask = self._mask(local_idx)
        if mask[0] == 0:
            mri[:] = 0
        if mask[1] == 0:
            ct[:] = 0
        if mask[2] == 0:
            xray[:] = 0
        if mask[3] == 0:
            clinical[:] = 0
        if mask[4] == 0:
            biomech[:] = 0

        return {
            "mri": torch.from_numpy(mri),
            "ct": torch.from_numpy(ct),
            "xray": torch.from_numpy(xray),
            "clinical": torch.from_numpy(clinical),
            "biomechanical": torch.from_numpy(biomech),
            "risk": torch.tensor(self.cohort.risk_score[i], dtype=torch.float32),
            "label": torch.tensor(self.cohort.risk_class[i], dtype=torch.long),
            "uncertainty": torch.tensor(self.cohort.uncertainty[i], dtype=torch.float32),
            "modality_mask": torch.from_numpy(mask),
            "index": torch.tensor(i, dtype=torch.long),
        }
