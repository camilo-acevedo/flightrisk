from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class TemporalSplit:
    """Indices for a strict temporal train / validation / test split.

    :param train_idx: Row positions in the train slice.
    :param val_idx: Row positions in the validation slice.
    :param test_idx: Row positions in the test slice.
    :param train_cutoff: Last timestamp included in train.
    :param val_cutoff: Last timestamp included in validation.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_cutoff: pd.Timestamp
    val_cutoff: pd.Timestamp


def temporal_split(
    timestamps: pd.Series,
    *,
    train_cutoff: str | pd.Timestamp,
    val_cutoff: str | pd.Timestamp,
) -> TemporalSplit:
    """Split rows into past / near-past / future slices, in that order.

    No row may appear in more than one slice. Rows beyond ``val_cutoff`` go to
    test; rows in ``(train_cutoff, val_cutoff]`` go to validation; the rest
    forms the train slice.

    :param timestamps: Per-row timestamps. Must be convertible to datetime.
    :param train_cutoff: Last timestamp kept in train.
    :param val_cutoff: Last timestamp kept in validation. Must be later than
        ``train_cutoff``.
    :returns: A :class:`TemporalSplit` with non-overlapping integer indices.
    :raises ValueError: If ``val_cutoff`` does not lie strictly after
        ``train_cutoff``.
    """
    train_cutoff_ts = pd.Timestamp(train_cutoff)
    val_cutoff_ts = pd.Timestamp(val_cutoff)
    if val_cutoff_ts <= train_cutoff_ts:
        raise ValueError("val_cutoff must be strictly after train_cutoff")

    ts = pd.to_datetime(timestamps).reset_index(drop=True)
    train_mask = ts <= train_cutoff_ts
    val_mask = (ts > train_cutoff_ts) & (ts <= val_cutoff_ts)
    test_mask = ts > val_cutoff_ts
    return TemporalSplit(
        train_idx=np.flatnonzero(train_mask.values),
        val_idx=np.flatnonzero(val_mask.values),
        test_idx=np.flatnonzero(test_mask.values),
        train_cutoff=train_cutoff_ts,
        val_cutoff=val_cutoff_ts,
    )


def stratified_rct_folds(
    treatment: pd.Series,
    outcome: pd.Series,
    *,
    n_splits: int = 5,
    seed: int = 1337,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build cross-validation folds that preserve the treatment/outcome ratio.

    Stratification is done on the joint ``treatment * 2 + outcome`` label so
    each fold sees a representative slice of the four cells.

    :param treatment: Binary 0/1 treatment indicator.
    :param outcome: Binary 0/1 outcome indicator.
    :param n_splits: Number of folds.
    :param seed: Random seed for fold ordering.
    :returns: List of ``(train_idx, val_idx)`` tuples.
    :raises ValueError: If treatment and outcome have different lengths.
    """
    if len(treatment) != len(outcome):
        raise ValueError("treatment and outcome must share the same length")

    joint = treatment.astype(int).values * 2 + outcome.astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(train, val) for train, val in skf.split(joint, joint)]
