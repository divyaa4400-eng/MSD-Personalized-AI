from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy import stats


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, confidence: float = 0.95, seed: int = 0) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n_boot)])
    alpha = 1.0 - confidence
    return {
        "mean": float(x.mean()),
        "lower": float(np.quantile(boot, alpha / 2)),
        "upper": float(np.quantile(boot, 1 - alpha / 2)),
    }


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return {"t_statistic": float("nan"), "p_value": float("nan"), "mean_difference": float("nan"), "cohens_dz": float("nan")}
    d = x[m] - y[m]
    t, p = stats.ttest_rel(x[m], y[m])
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("inf")
    return {
        "t_statistic": float(t),
        "p_value": float(p),
        "mean_difference": float(d.mean()),
        "cohens_dz": float(dz),
    }


def fold_summary(rows: Iterable[Dict[str, float]], metric_keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    rows = list(rows)
    out = {}
    for key in metric_keys:
        vals = np.asarray([r.get(key, np.nan) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[key] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
        }
    return out
