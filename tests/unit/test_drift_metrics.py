from __future__ import annotations

from core.drift_metrics import (
    kolmogorov_smirnov_test,
    population_stability_index,
)


def test_psi_no_drift_for_identical_distributions() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    result = population_stability_index(values, values, bins=4)
    assert result["psi"] < 0.01
    assert result["verdict"] == "no_drift"


def test_psi_detects_shifted_distribution() -> None:
    reference = [float(x) for x in range(0, 100)]
    current = [float(x) for x in range(50, 150)]
    result = population_stability_index(reference, current, bins=10)
    assert result["psi"] >= 0.1
    assert result["verdict"] in ("moderate_drift", "significant_drift")


def test_ks_detects_different_distributions() -> None:
    reference = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    current = [50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0]
    result = kolmogorov_smirnov_test(reference, current, alpha=0.05)
    assert result["statistic"] > 0.5
    assert result["drift_detected"] is True
    assert result["verdict"] == "drift"


def test_ks_no_drift_for_same_distribution() -> None:
    values = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
    result = kolmogorov_smirnov_test(values, values, alpha=0.05)
    assert result["statistic"] == 0.0
    assert result["drift_detected"] is False
