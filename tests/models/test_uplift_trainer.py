from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightrisk.models.uplift.meta_learners import (
    DRLearner,
    LightGBMUpliftParams,
    TLearner,
    XLearner,
)
from flightrisk.models.uplift.trainer import train_uplift_model


def _rct(n: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    treatment = rng.integers(0, 2, size=n)
    base = 1 / (1 + np.exp(-(0.3 * f0 - 0.2 * f1)))
    tau = 0.5 * (f0 > 0.0).astype(float)
    p = np.clip(base + tau * treatment, 0.01, 0.99)
    outcome = (rng.uniform(0, 1, n) < p).astype(int)
    features = pd.DataFrame({"f0": f0, "f1": f1, "f2": f2})
    return features, treatment, outcome


@pytest.mark.parametrize("learner_cls", [TLearner, XLearner, DRLearner])
def test_meta_learners_recover_positive_uplift_where_expected(learner_cls) -> None:
    features, treatment, outcome = _rct()
    fast = LightGBMUpliftParams(n_estimators=200, num_leaves=31, min_data_in_leaf=50)
    learner = learner_cls(fast).fit(features, treatment, outcome)
    uplift = learner.predict_uplift(features)
    assert np.mean(uplift[features["f0"] > 0]) > np.mean(uplift[features["f0"] <= 0])


@pytest.mark.parametrize("estimator", ["t_learner", "x_learner", "dr_learner"])
def test_train_uplift_model_end_to_end(estimator: str) -> None:
    features, treatment, outcome = _rct()
    n = len(features)
    train_idx = np.arange(0, int(0.7 * n))
    test_idx = np.arange(int(0.7 * n), n)
    result = train_uplift_model(
        features,
        treatment,
        outcome,
        train_idx=train_idx,
        test_idx=test_idx,
        estimator=estimator,
        k_percentiles=(10, 20, 30),
    )
    assert np.isfinite(result.metrics.qini)
    assert result.uplift_test.shape == (len(test_idx),)
    assert {10, 20, 30} == set(result.metrics.uplift_at_k)
