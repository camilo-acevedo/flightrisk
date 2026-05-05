from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightrisk.models.risk.calibration import CalibratedRiskModel
from flightrisk.models.risk.lightgbm_model import LightGBMRiskModel, LightGBMRiskParams
from flightrisk.models.risk.trainer import train_risk_model


def _synthetic(n: int = 1500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    f0 = rng.normal(size=n)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    logits = 0.8 * f0 - 0.6 * f1 + 0.3 * f2 - 1.2
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-logits))).astype(int)
    features = pd.DataFrame(
        {
            "msno": [f"u{i}" for i in range(n)],
            "f0": f0,
            "f1": f1,
            "f2": f2,
        }
    )
    return features, y


def test_lightgbm_wrapper_fits_and_predicts() -> None:
    features, y = _synthetic()
    X = features.drop(columns=["msno"])
    model = LightGBMRiskModel(LightGBMRiskParams(n_estimators=50, num_leaves=15))
    model.fit(X.iloc[:1000], y[:1000], X_val=X.iloc[1000:1200], y_val=y[1000:1200])
    proba = model.predict_proba(X.iloc[1200:])
    assert proba.shape == (300,)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_calibrator_requires_fit_before_predict() -> None:
    features, y = _synthetic()
    base = LightGBMRiskModel(LightGBMRiskParams(n_estimators=50)).fit(
        features.drop(columns=["msno"]).iloc[:800], y[:800]
    )
    calibrated = CalibratedRiskModel(base, method="isotonic")
    with pytest.raises(RuntimeError):
        calibrated.predict_proba(features.drop(columns=["msno"]).iloc[800:])


@pytest.mark.parametrize("calibration", ["isotonic", "platt", None])
def test_train_risk_model_end_to_end(calibration: str | None) -> None:
    features, y = _synthetic()
    n = len(features)
    train_idx = np.arange(0, int(0.6 * n))
    val_idx = np.arange(int(0.6 * n), int(0.8 * n))
    test_idx = np.arange(int(0.8 * n), n)

    result = train_risk_model(
        features,
        y,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        estimator="lightgbm",
        calibration=calibration,
    )
    assert result.metrics.auc > 0.6
    assert 0.0 <= result.metrics.ece <= 1.0
    assert "f0" in result.feature_names
    assert not result.calibration.empty
