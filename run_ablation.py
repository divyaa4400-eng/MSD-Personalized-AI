from __future__ import annotations

import argparse
from copy import deepcopy

import pandas as pd

from src.experiment import development_test_split, evaluate_indices, train_final_model
from src.synthetic_data import generate_synthetic_cohort
from src.utils import apply_dotted_overrides, ensure_dir, load_yaml, resolve_device, save_json, seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run component ablation experiments.")
    parser.add_argument("--config", default="configs/ablation.yaml")
    args = parser.parse_args()
    wrapper = load_yaml(args.config)
    base = load_yaml(wrapper.get("base_config", "configs/default.yaml"))
    base["output_dir"] = wrapper.get("output_dir", "outputs/ablation")
    seed_everything(int(base.get("seed", 0)))
    device = resolve_device(base.get("runtime", {}).get("device", "auto"))
    out = ensure_dir(base["output_dir"])

    # Cohort is generated once so every ablation sees identical samples and labels.
    cohort = generate_synthetic_cohort(base)
    dev, test = development_test_split(cohort, base)
    rows = []
    for name, overrides in wrapper.get("experiments", {}).items():
        cfg = apply_dotted_overrides(base, overrides or {})
        model, prep, _ = train_final_model(cohort, dev, cfg, device)
        metrics, _, _ = evaluate_indices(model, cohort, test, prep, cfg, device)
        rows.append({"configuration": name, **metrics})
        print(name, metrics)

    pd.DataFrame(rows).to_csv(out / "ablation_results.csv", index=False)
    save_json(rows, out / "ablation_results.json")
    print(f"Ablation results written to {out}")


if __name__ == "__main__":
    main()
