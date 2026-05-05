from __future__ import annotations

import numpy as np
import pytest

from flightrisk.eval.metrics import (
    calibration_table,
    expected_calibration_error,
    lift_at_decile,
    risk_metrics,
)


def test_perfect_calibration_yields_zero_ece() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    ece = expected_calibration_error(y, p, n_bins=20)
    assert ece < 0.05


def test_decile_lift_top_decile_above_one_for_skilled_model() -> None:
    rng = np.random.default_rng(1)
    n = 1000
    y = rng.binomial(1, 0.2, n)
    p = np.where(y == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n))
    assert lift_at_decile(y, p) > 1.5


def test_lift_at_decile_validates_range() -> None:
    with pytest.raises(ValueError):
        lift_at_decile(np.array([0, 1]), np.array([0.1, 0.9]), decile=0)
    with pytest.raises(ValueError):
        lift_at_decile(np.array([0, 1]), np.array([0.1, 0.9]), decile=11)


def test_risk_metrics_returns_finite_values() -> None:
    rng = np.random.default_rng(2)
    n = 2000
    y = rng.binomial(1, 0.3, n)
    p = np.clip(0.3 + 0.4 * (y - 0.3) + rng.normal(0, 0.1, n), 0, 1)
    m = risk_metrics(y, p)
    for value in m.as_dict().values():
        assert np.isfinite(value)
    assert 0.0 <= m.auc <= 1.0
    assert 0.0 <= m.ece <= 1.0


def test_calibration_table_structure() -> None:
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.5, 200)
    p = rng.uniform(0, 1, 200)
    table = calibration_table(y, p, n_bins=10)
    assert {"bin", "count", "mean_predicted", "empirical_rate"}.issubset(table.columns)
    assert (table["count"] > 0).all()
