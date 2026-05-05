from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightrisk.features.orange import OrangeFeatureConfig, prepare_orange_features


def _frame_with_missing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f0": [1.0, np.nan, 3.0, 4.0],
            "f1": [10.0, 20.0, np.nan, 40.0],
            "f_const": [1.0, 1.0, 1.0, 1.0],
        }
    )


def test_prepare_orange_features_median_fill() -> None:
    out = prepare_orange_features(_frame_with_missing())
    assert not out.isna().any().any()
    assert out.loc[1, "f0"] == 3.0


def test_prepare_orange_features_zero_fill() -> None:
    out = prepare_orange_features(
        _frame_with_missing(), config=OrangeFeatureConfig(fill_strategy="zero")
    )
    assert out.loc[1, "f0"] == 0.0


def test_prepare_orange_features_drops_low_variance() -> None:
    out = prepare_orange_features(
        _frame_with_missing(),
        config=OrangeFeatureConfig(drop_low_variance_threshold=1e-6),
    )
    assert "f_const" not in out.columns


def test_prepare_orange_features_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        prepare_orange_features(
            _frame_with_missing(), config=OrangeFeatureConfig(fill_strategy="bogus")
        )
