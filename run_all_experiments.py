from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run all reproducibility experiment families sequentially.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    args = parser.parse_args()
    py = args.python
    run([py, "run_full_pipeline.py", "--config", "configs/default.yaml"])
    run([py, "run_baselines.py", "--config", "configs/baselines.yaml"])
    run([py, "run_ablation.py", "--config", "configs/ablation.yaml"])
    run([py, "run_robustness.py", "--config", "configs/robustness.yaml"])
    print("\nALL EXPERIMENT FAMILIES COMPLETED")


if __name__ == "__main__":
    main()
