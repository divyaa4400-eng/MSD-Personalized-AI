from __future__ import annotations

import argparse

import pandas as pd

from src.experiment import development_test_split, evaluate_indices, train_final_model
from src.robustness import build_forced_mask, performance_drop
from src.synthetic_data import generate_synthetic_cohort
from src.utils import ensure_dir, load_yaml, resolve_device, save_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run missing-modality and noise robustness experiments.")
    parser.add_argument("--config", default="configs/robustness.yaml")
    args = parser.parse_args()
    wrapper = load_yaml(args.config)
    base = load_yaml(wrapper.get("base_config", "configs/default.yaml"))
    base["output_dir"] = wrapper.get("output_dir", "outputs/robustness")
    seed_everything(int(base.get("seed", 0)))
    device = resolve_device(base.get("runtime", {}).get("device", "auto"))
    out = ensure_dir(base["output_dir"])

    cohort = generate_synthetic_cohort(base)
    dev, test = development_test_split(cohort, base)
    model, prep, _ = train_final_model(cohort, dev, base, device)

    clean_metrics = None
    rows = []
    for i, (name, scenario) in enumerate(wrapper.get("scenarios", {}).items()):
        scenario = scenario or {}
        mask = build_forced_mask(len(test), scenario, seed=int(base.get("seed", 0)) + i)
        metrics, _, _ = evaluate_indices(
            model,
            cohort,
            test,
            prep,
            base,
            device,
            forced_mask=mask,
            image_noise_std=float(scenario.get("image_noise_std", 0.0) or 0.0),
            structured_noise_std=float(scenario.get("structured_noise_std", 0.0) or 0.0),
        )
        if name == "clean":
            clean_metrics = metrics
        if clean_metrics is None:
            clean_metrics = metrics
        rows.append({"scenario": name, **metrics, **performance_drop(clean_metrics, metrics)})
        print(name, metrics)

    pd.DataFrame(rows).to_csv(out / "robustness_results.csv", index=False)
    save_json(rows, out / "robustness_results.json")
    print(f"Robustness results written to {out}")


if __name__ == "__main__":
    main()
