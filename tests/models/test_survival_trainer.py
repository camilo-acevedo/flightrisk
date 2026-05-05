from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightrisk.models.survival.trainer import train_survival_model


def _synthetic(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    risk = np.exp(0.7 * f0 - 0.4 * f1 + 0.2 * f2)
    durations_true = rng.exponential(scale=180.0 / risk)
    horizon = 180.0
    durations = np.minimum(durations_true, horizon)
    events = (durations_true <= horizon).astype(int)
    features = pd.DataFrame(
        {
            "msno": [f"u{i}" for i in range(n)],
            "f0": f0,
            "f1": f1,
            "f2": f2,
        }
    )
    return features, durations, events


@pytest.mark.parametrize("estimator", ["cox", "rsf"])
def test_train_survival_model_end_to_end(estimator: str) -> None:
    features, durations, events = _synthetic()
    n = len(features)
    train_idx = np.arange(0, int(0.7 * n))
    test_idx = np.arange(int(0.7 * n), n)

    result = train_survival_model(
        features,
        durations,
        events,
        train_idx=train_idx,
        test_idx=test_idx,
        estimator=estimator,
        horizons_days=(30, 90, 150),
    )
    assert result.metrics.c_index > 0.55
    assert 0.0 <= result.metrics.integrated_brier <= 0.5
    assert result.survival_at_horizons.shape == (len(test_idx), 3)
    assert (result.survival_at_horizons >= 0.0).all()
    assert (result.survival_at_horizons <= 1.0).all()


def test_train_survival_model_rejects_unknown_estimator() -> None:
    features, durations, events = _synthetic(n=120)
    with pytest.raises(ValueError):
        train_survival_model(
            features,
            durations,
            events,
            train_idx=np.arange(80),
            test_idx=np.arange(80, 120),
            estimator="bogus",
        )
