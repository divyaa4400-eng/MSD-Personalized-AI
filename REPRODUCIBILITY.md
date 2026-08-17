# Reproducibility Protocol

## Primary reproducibility command

```bash
python run_full_pipeline.py --config configs/default.yaml
```

## Determinism

The pipeline seeds Python, NumPy, and PyTorch. Split indices and configuration snapshots are written to the output directory. Exact floating-point equality across different operating systems, BLAS implementations, GPUs, and PyTorch releases is not guaranteed; statistical reproducibility is the intended standard.

## Development and independent test design

The cohort is stratified into:

- 85% development set;
- 15% locked independent test set.

Five-fold stratified cross-validation is performed only within the development set. The independent test set is not used for hyperparameter selection.

## Required outputs

A successful primary run should generate:

- cohort summary;
- split manifest;
- fold-level validation metrics;
- final independent-test metrics;
- calibration metrics;
- model weights;
- attention summaries;
- grouped attribution;
- performance and explainability figures; and
- runtime/environment metadata.

## Integrity rules

1. No target or metric is hard-coded.
2. Test subjects are never used in model fitting.
3. Baselines use the same split manifest when run in the same output directory.
4. Ablations change only the component under study where practical.
5. Statistical comparisons use fold-level or bootstrap distributions rather than a single point estimate.
6. Synthetic data must not be described as real hospital data.

## Verification

Run:

```bash
python verify_reproducibility.py
```

The verifier performs fast structural and numerical checks without requiring a full long training run.
