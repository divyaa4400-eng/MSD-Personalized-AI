from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.calibration import calibration_metrics, reliability_bins
from src.experiment import development_test_split, evaluate_indices, run_cross_validation, save_split_manifest, train_final_model
from src.explainability import gradcam_for_sample, grouped_permutation_attribution
from src.preprocessing import MSDDataset
from src.statistics import bootstrap_ci
from src.synthetic_data import MODALITY_NAMES, cohort_summary, generate_synthetic_cohort
from src.utils import count_trainable_parameters, ensure_dir, environment_metadata, load_yaml, save_json, save_yaml, seed_everything, resolve_device
from src.visualization import (
    plot_confusion,
    plot_cv_stability,
    plot_gradcam,
    plot_pr,
    plot_predicted_vs_reference,
    plot_reliability,
    plot_roc,
)


def bootstrap_metric_cis(pred, n_boot: int, seed: int):
    from src.evaluation import compute_metrics
    rng = np.random.default_rng(seed)
    n = len(pred["label"])
    collected = {k: [] for k in ["accuracy", "f1_macro", "auc_macro_ovr", "auc_high_risk", "mse", "pearson_r"]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = compute_metrics(pred["label"][idx], pred["risk_true"][idx], pred["probs"][idx], pred["risk_pred"][idx])
        for k in collected:
            if np.isfinite(m[k]):
                collected[k].append(m[k])
    return {k: bootstrap_ci(v, n_boot=min(1000, max(200, len(v))), seed=seed + i) for i, (k, v) in enumerate(collected.items())}


def main():
    parser = argparse.ArgumentParser(description="Run complete synthetic multimodal MSD reproducibility pipeline.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    seed_everything(int(config.get("seed", 0)), bool(config.get("runtime", {}).get("deterministic", True)))
    device = resolve_device(config.get("runtime", {}).get("device", "auto"))
    out_dir = ensure_dir(config.get("output_dir", "outputs/main"))
    save_yaml(config, out_dir / "resolved_config.yaml")

    start = time.perf_counter()
    cohort = generate_synthetic_cohort(config)
    summary = cohort_summary(cohort)
    pd.DataFrame([summary]).to_csv(out_dir / "data_summary.csv", index=False)

    dev, test = development_test_split(cohort, config)
    if set(dev).intersection(set(test)):
        raise RuntimeError("Development/test leakage detected")
    save_split_manifest(cohort, dev, test, out_dir / "split_manifest.json")

    cv_rows, cv_summary = run_cross_validation(cohort, dev, config, device)
    pd.DataFrame(cv_rows).to_csv(out_dir / "cv_metrics.csv", index=False)
    save_json(cv_summary, out_dir / "cv_summary.json")
    plot_cv_stability(cv_rows, out_dir / "cv_stability.png")

    model, state, train_info = train_final_model(cohort, dev, config, device)
    torch.save(model.state_dict(), out_dir / "model_state.pt")
    joblib.dump(state, out_dir / "preprocessing_state.joblib")
    save_json(train_info, out_dir / "final_training_info.json")

    test_metrics, pred, test_loader = evaluate_indices(model, cohort, test, state, config, device)
    save_json(test_metrics, out_dir / "test_metrics.json")
    ci = bootstrap_metric_cis(pred, int(config.get("calibration", {}).get("bootstrap_samples", 500)), int(config.get("seed", 0)))
    save_json(ci, out_dir / "test_metric_confidence_intervals.json")

    cal = calibration_metrics(pred["label"], pred["probs"], int(config.get("calibration", {}).get("n_bins", 10)))
    bins = reliability_bins(pred["label"], pred["probs"], int(config.get("calibration", {}).get("n_bins", 10)))
    save_json(cal, out_dir / "calibration_metrics.json")
    pd.DataFrame(bins).to_csv(out_dir / "reliability_bins.csv", index=False)

    attention = pd.DataFrame(pred["attention"], columns=MODALITY_NAMES)
    attention.insert(0, "subject_id", cohort.subject_id[pred["index"]])
    attention.to_csv(out_dir / "attention_weights.csv", index=False)
    pd.DataFrame({"modality": MODALITY_NAMES, "mean_attention": pred["attention"].mean(axis=0)}).to_csv(out_dir / "attention_summary.csv", index=False)

    # Explanations use a bounded test subset to keep CPU reproduction practical.
    explain_count = min(64, len(test))
    explain_indices = test[:explain_count]
    explain_loader = DataLoader(MSDDataset(cohort, explain_indices, state, training=False), batch_size=16, shuffle=False)
    grouped = grouped_permutation_attribution(
        model,
        explain_loader,
        device,
        repeats=int(config.get("explainability", {}).get("permutation_repeats", 8)),
        seed=int(config.get("seed", 0)),
    )
    pd.DataFrame([{"modality": k, "relative_importance": v} for k, v in grouped.items()]).to_csv(out_dir / "grouped_attribution.csv", index=False)

    # Single-subject Grad-CAM per imaging branch.
    one_loader = DataLoader(MSDDataset(cohort, [int(test[0])], state, training=False), batch_size=1, shuffle=False)
    one = next(iter(one_loader))
    for modality in ("mri", "ct", "xray"):
        cam = gradcam_for_sample(model, one, modality, device)
        plot_gradcam(one[modality][0].numpy(), cam, out_dir / f"gradcam_{modality}.png", f"Grad-CAM: {modality.upper()}")

    plot_roc(pred["label"], pred["probs"], out_dir / "roc_curve.png")
    plot_pr(pred["label"], pred["probs"], out_dir / "precision_recall_curve.png")
    plot_confusion(pred["label"], pred["probs"], out_dir / "confusion_matrix.png")
    plot_reliability(bins, out_dir / "reliability_diagram.png")
    plot_predicted_vs_reference(pred["risk_true"], pred["risk_pred"], out_dir / "predicted_vs_reference.png")

    metadata = environment_metadata(device)
    metadata.update({
        "seed": int(config.get("seed", 0)),
        "trainable_parameters": count_trainable_parameters(model),
        "development_size": int(len(dev)),
        "test_size": int(len(test)),
        "total_runtime_seconds": time.perf_counter() - start,
        "data_source": "fully synthetic computational cohort",
    })
    save_json(metadata, out_dir / "run_metadata.json")

    print("\nPrimary reproducibility run completed.")
    print(f"Output directory: {out_dir}")
    for key in ["accuracy", "f1_macro", "auc_macro_ovr", "auc_high_risk", "mse", "pearson_r", "brier_multiclass", "ece"]:
        print(f"{key}: {test_metrics.get(key)}")


if __name__ == "__main__":
    main()
