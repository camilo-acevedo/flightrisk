from __future__ import annotations

import numpy as np
import pytest

from flightrisk.eval.uplift_metrics import qini_coefficient, uplift_at_k, uplift_metrics


def _rct(n: int = 2000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, size=n)
    base_rate = 0.2
    score = rng.uniform(0, 1, n)
    treated_lift = 0.4 * (score > 0.5)
    p = base_rate + treated_lift * treatment
    outcome = (rng.uniform(0, 1, n) < p).astype(int)
    return score, treatment, outcome


def test_qini_coefficient_positive_for_skilled_ranking() -> None:
    score, treatment, outcome = _rct()
    assert qini_coefficient(score, treatment, outcome) > 0.0


def test_qini_coefficient_random_ranking_is_near_zero() -> None:
    score, treatment, outcome = _rct(seed=1)
    rng = np.random.default_rng(2)
    random_score = rng.uniform(0, 1, size=len(score))
    assert abs(qini_coefficient(random_score, treatment, outcome)) < 0.05


def test_uplift_at_k_validates_range() -> None:
    score, treatment, outcome = _rct(n=400)
    with pytest.raises(ValueError):
        uplift_at_k(score, treatment, outcome, k_percent=0)
    with pytest.raises(ValueError):
        uplift_at_k(score, treatment, outcome, k_percent=101)


def test_uplift_metrics_returns_per_k_dict() -> None:
    score, treatment, outcome = _rct()
    m = uplift_metrics(uplift=score, treatment=treatment, outcome=outcome, k_percentiles=(10, 20))
    assert set(m.uplift_at_k) == {10, 20}
    flat = m.as_dict()
    assert "uplift_at_10" in flat and "uplift_at_20" in flat
    assert np.isfinite(flat["qini"])
