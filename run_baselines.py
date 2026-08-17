from __future__ import annotations

import argparse
from copy import deepcopy

import numpy as np
import pandas as pd

from src.baselines import fit_classical_baseline, handcrafted_features
from src.advanced_baselines import train_advanced_baseline
from src.evaluation import compute_metrics
from src.experiment import development_test_split, evaluate_indices, train_final_model
from src.preprocessing import fit_preprocessing
from src.synthetic_data import generate_synthetic_cohort
from src.utils import deep_update, ensure_dir, load_yaml, resolve_device, save_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run baseline comparisons.")
    parser.add_argument("--config", default="configs/baselines.yaml")
    args = parser.parse_args()
    wrapper = load_yaml(args.config)
    base = load_yaml(wrapper.get("base_config", "configs/default.yaml"))
    base["output_dir"] = wrapper.get("output_dir", "outputs/baselines")
    base["classical"] = wrapper.get("classical", {})
    seed_everything(int(base.get("seed", 0)))
    device = resolve_device(base.get("runtime", {}).get("device", "auto"))
    out = ensure_dir(base["output_dir"])

    cohort = generate_synthetic_cohort(base)
    dev, test = development_test_split(cohort, base)
    state = fit_preprocessing(cohort, dev)
    x = handcrafted_features(cohort, state)
    rows = []

    for name in wrapper.get("baselines", []):
        if name in {"logistic_regression", "random_forest", "svm"}:
            model = fit_classical_baseline(name, x[dev], cohort.risk_class[dev], base)
            probs = model.predict_proba(x[test])
            risk_pred = probs @ np.linspace(0.2, 0.85, probs.shape[1])
            metrics = compute_metrics(cohort.risk_class[test], cohort.risk_score[test], probs, risk_pred)
        else:
            cfg = deepcopy(base)
            if name == "clinical_mlp":
                cfg.setdefault("data", {})["active_modalities"] = ["clinical"]
                cfg.setdefault("model", {})["fusion_type"] = "mean"
            elif name == "imaging_only":
                cfg.setdefault("data", {})["active_modalities"] = ["mri", "ct", "xray"]
                cfg.setdefault("model", {})["fusion_type"] = "mean"
            elif name == "concat_fusion":
                cfg.setdefault("model", {})["fusion_type"] = "concat"
            elif name == "gated_attention":
                cfg.setdefault("model", {})["fusion_type"] = "gated_attention"
            else:
                raise ValueError(f"Unsupported baseline: {name}")
            model, prep, _ = train_final_model(cohort, dev, cfg, device)
            metrics, _, _ = evaluate_indices(model, cohort, test, prep, cfg, device)
        rows.append({"model": name, **metrics})
        print(name, metrics)

    advanced_cfg = wrapper.get("advanced_baselines", {}) or {}
    if bool(advanced_cfg.get("enabled", False)):
        for name in advanced_cfg.get("models", []):
            adv_model = train_advanced_baseline(cohort, dev, state, base, device, name)
            metrics, _, _ = evaluate_indices(adv_model, cohort, test, state, base, device)
            rows.append({"model": name, **metrics})
            print(name, metrics)

    df = pd.DataFrame(rows)
    df.to_csv(out / "baseline_results.csv", index=False)
    save_json(rows, out / "baseline_results.json")
    print(f"Baseline comparison written to {out}")


if __name__ == "__main__":
    main()
