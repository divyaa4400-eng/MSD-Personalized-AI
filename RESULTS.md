# Results

This file intentionally does not contain fixed performance claims.

Run the experiments to produce machine-generated results:

```bash
python run_full_pipeline.py --config configs/default.yaml
python run_baselines.py --config configs/baselines.yaml
python run_ablation.py --config configs/ablation.yaml
python run_robustness.py --config configs/robustness.yaml
```

The generated CSV/JSON outputs should be used to populate or verify the manuscript tables. If the reproduced values differ from values currently written in the manuscript, the calculated results should be investigated and the manuscript should be updated rather than forcing the code to match predetermined numbers.
