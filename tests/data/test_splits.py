from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flightrisk.data.splits import stratified_rct_folds, temporal_split


def test_temporal_split_partitions_rows() -> None:
    ts = pd.Series(pd.date_range("2017-01-01", periods=10, freq="D"))
    split = temporal_split(ts, train_cutoff="2017-01-04", val_cutoff="2017-01-07")
    all_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert sorted(all_idx.tolist()) == list(range(10))
    assert split.train_idx.tolist() == [0, 1, 2, 3]
    assert split.val_idx.tolist() == [4, 5, 6]
    assert split.test_idx.tolist() == [7, 8, 9]


def test_temporal_split_rejects_inverted_cutoffs() -> None:
    ts = pd.Series(pd.date_range("2017-01-01", periods=4, freq="D"))
    with pytest.raises(ValueError):
        temporal_split(ts, train_cutoff="2017-01-03", val_cutoff="2017-01-02")


def test_stratified_rct_folds_preserve_treatment_ratio() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    treatment = pd.Series(rng.integers(0, 2, n))
    outcome = pd.Series(rng.integers(0, 2, n))
    folds = stratified_rct_folds(treatment, outcome, n_splits=5, seed=42)

    assert len(folds) == 5
    overall_t = treatment.mean()
    for _, val_idx in folds:
        fold_t = treatment.iloc[val_idx].mean()
        assert abs(fold_t - overall_t) < 0.05


def test_stratified_rct_folds_rejects_length_mismatch() -> None:
    treatment = pd.Series([0, 1, 0])
    outcome = pd.Series([0, 1])
    with pytest.raises(ValueError):
        stratified_rct_folds(treatment, outcome)
