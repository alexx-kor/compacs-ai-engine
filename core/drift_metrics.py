"""Population Stability Index (PSI) and Kolmogorov–Smirnov drift tests."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

PSI_NO_DRIFT = 0.1
PSI_MODERATE = 0.25

_EPS = 1e-6


def _as_array(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty 1-D list")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must be finite numbers")
    return arr


def quantile_bin_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        vmin = float(reference.min())
        vmax = float(reference.max())
        if vmin == vmax:
            return np.array([vmin - _EPS, vmax + _EPS])
        return np.linspace(vmin, vmax, bins + 1)
    edges[0] -= _EPS
    edges[-1] += _EPS
    return edges


def bucket_percentages(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        raise ValueError("empty histogram")
    pct = counts.astype(float) / float(total)
    return np.clip(pct, _EPS, None)


def population_stability_index(
    reference: list[float],
    current: list[float],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """PSI with quantile bins fitted on reference distribution."""
    ref = _as_array(reference)
    cur = _as_array(current)
    edges = quantile_bin_edges(ref, bins)
    edges[0] = min(float(edges[0]), float(cur.min())) - _EPS
    edges[-1] = max(float(edges[-1]), float(cur.max())) + _EPS
    ref_pct = bucket_percentages(ref, edges)
    cur_pct = bucket_percentages(cur, edges)
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    if psi < PSI_NO_DRIFT:
        verdict = "no_drift"
    elif psi < PSI_MODERATE:
        verdict = "moderate_drift"
    else:
        verdict = "significant_drift"

    return {
        "psi": round(psi, 6),
        "verdict": verdict,
        "bins": int(edges.size - 1),
        "reference_pct": [round(float(x), 6) for x in ref_pct],
        "current_pct": [round(float(x), 6) for x in cur_pct],
        "bin_edges": [round(float(x), 6) for x in edges],
    }


def kolmogorov_smirnov_statistic(reference: list[float], current: list[float]) -> float:
    ref = np.sort(_as_array(reference))
    cur = np.sort(_as_array(current))
    data = np.sort(np.concatenate([ref, cur]))
    cdf_ref = np.searchsorted(ref, data, side="right") / ref.size
    cdf_cur = np.searchsorted(cur, data, side="right") / cur.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def kolmogorov_smirnov_pvalue(statistic: float, n_ref: int, n_cur: int) -> float:
    """Asymptotic two-sample KS p-value (no scipy)."""
    if n_ref <= 0 or n_cur <= 0:
        return 1.0
    if statistic <= 0.0:
        return 1.0
    ne = n_ref * n_cur / (n_ref + n_cur)
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * statistic
    if lam == 0.0:
        return 1.0
    total = 0.0
    for j in range(1, 101):
        total += (-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam)
    return float(min(1.0, max(0.0, 2.0 * total)))


def kolmogorov_smirnov_test(
    reference: list[float],
    current: list[float],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    ref = _as_array(reference)
    cur = _as_array(current)
    statistic = kolmogorov_smirnov_statistic(reference, current)
    p_value = kolmogorov_smirnov_pvalue(statistic, ref.size, cur.size)
    drift_detected = p_value < alpha
    return {
        "statistic": round(statistic, 6),
        "p_value": round(p_value, 6),
        "alpha": alpha,
        "drift_detected": drift_detected,
        "verdict": "drift" if drift_detected else "no_drift",
    }


def drift_summary(
    reference: list[float],
    current: list[float],
    *,
    metric_name: str,
    bins: int = 10,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if len(reference) < 2 or len(current) < 2:
        return {
            "metric": metric_name,
            "reference_n": len(reference),
            "current_n": len(current),
            "skipped": True,
            "reason": "need at least 2 values in each sample",
        }
    psi = population_stability_index(reference, current, bins=bins)
    ks = kolmogorov_smirnov_test(reference, current, alpha=alpha)
    overall = "drift"
    if psi["verdict"] == "no_drift" and ks["verdict"] == "no_drift":
        overall = "no_drift"
    elif psi["verdict"] == "significant_drift" or ks["drift_detected"]:
        overall = "significant_drift"
    else:
        overall = "moderate_drift"

    return {
        "metric": metric_name,
        "reference_n": len(reference),
        "current_n": len(current),
        "skipped": False,
        "psi": psi,
        "ks": ks,
        "overall_verdict": overall,
    }
