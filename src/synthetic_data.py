from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


CLINICAL_FEATURE_NAMES: List[str] = [
    "age",
    "sex_binary",
    "bmi",
    "smoking",
    "physical_activity",
    "occupational_risk",
    "hypertension",
    "diabetes",
    "family_history",
]

BIOMECH_FEATURE_NAMES: List[str] = [
    "gait_variability",
    "joint_flexibility",
    "isometric_strength",
    "postural_misalignment",
    "joint_load_index",
]

MODALITY_NAMES = ["mri", "ct", "xray", "clinical", "biomechanical"]


@dataclass
class SyntheticCohort:
    subject_id: np.ndarray
    mri: np.ndarray
    ct: np.ndarray
    xray: np.ndarray
    clinical: np.ndarray
    biomechanical: np.ndarray
    risk_score: np.ndarray
    risk_class: np.ndarray
    uncertainty: np.ndarray
    latent_burden: np.ndarray
    modality_scores: np.ndarray
    clinical_feature_names: List[str]
    biomech_feature_names: List[str]

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "subject_id": self.subject_id,
            "mri": self.mri,
            "ct": self.ct,
            "xray": self.xray,
            "clinical": self.clinical,
            "biomechanical": self.biomechanical,
            "risk_score": self.risk_score,
            "risk_class": self.risk_class,
            "uncertainty": self.uncertainty,
            "latent_burden": self.latent_burden,
            "modality_scores": self.modality_scores,
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _normalized_grid(size: int) -> Tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return xx, yy


def _render_modality(
    rng: np.random.Generator,
    severity: np.ndarray,
    bmi_z: np.ndarray,
    age_z: np.ndarray,
    size: int,
    modality: str,
) -> np.ndarray:
    """Generate modality-specific 2-D image surrogates.

    These images are numerical phantoms for algorithmic reproducibility only.
    They do not represent real radiological acquisitions.
    """
    n = len(severity)
    xx, yy = _normalized_grid(size)
    xx = xx[None, :, :]
    yy = yy[None, :, :]

    sev = severity[:, None, None]
    bmi = bmi_z[:, None, None]
    age = age_z[:, None, None]

    asym = rng.normal(0, 0.05, n).astype(np.float32)[:, None, None]
    joint_shift = (0.15 * sev + asym).astype(np.float32)

    bone_left = np.exp(-(((xx + 0.34 + joint_shift) / 0.19) ** 2 + (yy / 0.70) ** 2))
    bone_right = np.exp(-(((xx - 0.34 - joint_shift) / 0.19) ** 2 + (yy / 0.70) ** 2))
    cartilage = np.exp(-((yy / (0.12 + 0.07 * (1 - sev))) ** 2 + (xx / 0.72) ** 4))
    marrow = np.exp(-((xx / 0.73) ** 2 + (yy / 0.78) ** 2))
    osteophyte = np.exp(-(((np.abs(xx) - 0.52) / 0.10) ** 2 + ((yy + 0.08) / 0.18) ** 2)) * sev

    texture = (
        0.35 * np.sin((7.0 + 1.2 * sev) * xx + 2.4 * yy)
        + 0.20 * np.cos(5.2 * yy - 1.8 * xx)
    )

    if modality == "mri":
        image = (
            0.30 * marrow
            + 0.55 * cartilage * (1.0 - 0.55 * sev)
            + 0.22 * (bone_left + bone_right)
            + 0.12 * texture
            + 0.10 * bmi
        )
        noise_std = 0.055 + 0.015 * sev
    elif modality == "ct":
        image = (
            0.75 * (bone_left + bone_right) * (1.0 + 0.18 * age)
            + 0.35 * osteophyte
            + 0.10 * marrow
            + 0.05 * texture
        )
        noise_std = 0.035 + 0.010 * sev
    elif modality == "xray":
        projection = np.sqrt(np.maximum(bone_left + bone_right, 0.0))
        joint_space_loss = sev * np.exp(-((yy / 0.15) ** 2))
        image = 0.68 * projection + 0.22 * osteophyte - 0.28 * joint_space_loss + 0.07 * age
        noise_std = 0.045 + 0.010 * sev
    else:
        raise ValueError(f"Unknown modality: {modality}")

    image = image + rng.normal(0.0, 1.0, image.shape) * noise_std
    p01 = np.quantile(image, 0.01, axis=(1, 2), keepdims=True)
    p99 = np.quantile(image, 0.99, axis=(1, 2), keepdims=True)
    image = (image - p01) / np.maximum(p99 - p01, 1e-6)
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    return image[:, None, :, :]


def generate_synthetic_cohort(config: Dict) -> SyntheticCohort:
    data_cfg = config.get("data", config)
    seed = int(config.get("seed", 20260817))
    rng = np.random.default_rng(seed)
    n = int(data_cfg.get("n_subjects", 1200))
    size = int(data_cfg.get("image_size", 32))
    annotation_noise = float(data_cfg.get("annotation_noise", 0.035))
    missing_rate = float(data_cfg.get("missing_rate", 0.03))
    thresholds = data_cfg.get("risk_thresholds", [0.40, 0.67])

    subject_id = np.array([f"SYN-{i+1:05d}" for i in range(n)], dtype=object)

    age = np.clip(rng.normal(52.6, 14.3, n), 18, 75)
    # Preserve the manuscript table's approximately 640:560 sex composition at n=1200.
    n_binary1 = int(round(n * (640.0 / 1200.0)))
    sex = np.concatenate([np.ones(n_binary1), np.zeros(n - n_binary1)]).astype(float)
    rng.shuffle(sex)
    bmi = np.clip(rng.normal(27.1, 4.8, n), 17.0, 44.0)
    smoking_p = _sigmoid(-1.45 + 0.25 * sex + 0.18 * (age - 50) / 15)
    smoking = rng.binomial(1, smoking_p)
    activity = np.clip(rng.beta(2.4, 2.0, n), 0, 1)
    occupation = np.clip(rng.beta(2.0, 2.2, n) + 0.08 * sex, 0, 1)
    hypertension_p = _sigmoid(-1.15 + 0.055 * (age - 50) + 0.06 * (bmi - 27))
    diabetes_p = _sigmoid(-1.55 + 0.045 * (age - 50) + 0.11 * (bmi - 27))
    hypertension = rng.binomial(1, hypertension_p)
    diabetes = rng.binomial(1, diabetes_p)
    family_history = rng.binomial(1, 0.28 + 0.05 * sex)

    age_z = (age - 52.6) / 14.3
    bmi_z = (bmi - 27.1) / 4.8

    gait = np.clip(0.35 + 0.12 * age_z + 0.17 * bmi_z + 0.18 * occupation + rng.normal(0, 0.10, n), 0, 1)
    flexibility = np.clip(0.72 - 0.14 * age_z - 0.11 * bmi_z - 0.08 * occupation + rng.normal(0, 0.10, n), 0, 1)
    strength = np.clip(0.68 - 0.16 * age_z - 0.12 * bmi_z + 0.15 * activity + rng.normal(0, 0.11, n), 0, 1)
    posture = np.clip(0.25 + 0.13 * age_z + 0.13 * bmi_z + 0.16 * occupation + rng.normal(0, 0.10, n), 0, 1)
    joint_load = np.clip(0.30 + 0.20 * bmi_z + 0.18 * occupation + 0.16 * gait + rng.normal(0, 0.10, n), 0, 1)

    clinical_linear = (
        0.48 * age_z
        + 0.32 * bmi_z
        + 0.28 * smoking
        - 0.42 * activity
        + 0.50 * occupation
        + 0.24 * hypertension
        + 0.30 * diabetes
        + 0.31 * family_history
    )
    biomech_linear = (
        0.75 * gait
        - 0.62 * flexibility
        - 0.58 * strength
        + 0.78 * posture
        + 0.82 * joint_load
    )
    interaction = 0.28 * bmi_z * occupation + 0.22 * age_z * gait + 0.18 * diabetes * joint_load
    latent_burden = _sigmoid(-0.82 + 0.58 * clinical_linear + 0.67 * biomech_linear + interaction + rng.normal(0, 0.24, n))

    mri_score = np.clip(latent_burden + 0.06 * (1 - flexibility) + rng.normal(0, 0.045, n), 0, 1)
    ct_score = np.clip(latent_burden + 0.05 * np.maximum(age_z, 0) + rng.normal(0, 0.045, n), 0, 1)
    xray_score = np.clip(latent_burden + 0.05 * joint_load + rng.normal(0, 0.050, n), 0, 1)
    modality_scores = np.stack([mri_score, ct_score, xray_score], axis=1).astype(np.float32)

    mri = _render_modality(rng, mri_score, bmi_z, age_z, size, "mri")
    ct = _render_modality(rng, ct_score, bmi_z, age_z, size, "ct")
    xray = _render_modality(rng, xray_score, bmi_z, age_z, size, "xray")

    clinical = np.column_stack([
        age,
        sex,
        bmi,
        smoking,
        activity,
        occupation,
        hypertension,
        diabetes,
        family_history,
    ]).astype(np.float32)
    biomechanical = np.column_stack([gait, flexibility, strength, posture, joint_load]).astype(np.float32)

    clinical_component = _sigmoid(-0.55 + 0.65 * clinical_linear)
    biomech_component = _sigmoid(-0.60 + 0.92 * biomech_linear)
    imaging_component = modality_scores.mean(axis=1)

    base_risk = (
        0.48 * imaging_component
        + 0.30 * clinical_component
        + 0.18 * biomech_component
        + 0.04 * latent_burden * occupation
    )
    uncertainty = np.clip(
        0.025
        + 0.06 * np.exp(-((base_risk - thresholds[0]) / 0.09) ** 2)
        + 0.06 * np.exp(-((base_risk - thresholds[1]) / 0.09) ** 2),
        0.02,
        0.14,
    )
    risk_score = np.clip(base_risk + rng.normal(0, annotation_noise + uncertainty * 0.35, n), 0, 1).astype(np.float32)
    risk_class = np.digitize(risk_score, bins=np.asarray(thresholds, dtype=float)).astype(np.int64)

    # Introduce small, realistic missingness only into structured raw variables.
    # Missing values are imputed from development statistics in preprocessing.
    # Core demographic fields (age, sex, BMI) remain complete; missingness is
    # introduced only in secondary clinical and biomechanical measurements.
    if missing_rate > 0:
        clinical_miss = rng.random(clinical[:, 3:].shape) < missing_rate
        clinical[:, 3:][clinical_miss] = np.nan
        biomech_miss = rng.random(biomechanical.shape) < missing_rate
        biomechanical[biomech_miss] = np.nan

    if bool(data_cfg.get("disable_clinical", False)):
        clinical[:] = 0.0
    if bool(data_cfg.get("disable_biomechanics", False)):
        biomechanical[:] = 0.0

    return SyntheticCohort(
        subject_id=subject_id,
        mri=mri,
        ct=ct,
        xray=xray,
        clinical=clinical,
        biomechanical=biomechanical,
        risk_score=risk_score,
        risk_class=risk_class,
        uncertainty=uncertainty.astype(np.float32),
        latent_burden=latent_burden.astype(np.float32),
        modality_scores=modality_scores,
        clinical_feature_names=CLINICAL_FEATURE_NAMES.copy(),
        biomech_feature_names=BIOMECH_FEATURE_NAMES.copy(),
    )


def cohort_summary(cohort: SyntheticCohort) -> Dict[str, float]:
    clinical = cohort.clinical
    age = clinical[:, 0]
    sex = clinical[:, 1]
    return {
        "n_subjects": int(len(cohort.subject_id)),
        "mean_age": float(np.nanmean(age)),
        "sd_age": float(np.nanstd(age, ddof=1)),
        "male_or_binary1_count": int(np.nansum(sex == 1)),
        "female_or_binary0_count": int(np.nansum(sex == 0)),
        "class_0_count": int(np.sum(cohort.risk_class == 0)),
        "class_1_count": int(np.sum(cohort.risk_class == 1)),
        "class_2_count": int(np.sum(cohort.risk_class == 2)),
        "risk_mean": float(np.mean(cohort.risk_score)),
        "risk_sd": float(np.std(cohort.risk_score, ddof=1)),
    }
