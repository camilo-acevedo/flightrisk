from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass(frozen=True)
class RiskMetrics:
    """Metrics emitted by the risk track.

    :param auc: Area under the ROC curve.
    :param pr_auc: Average precision (area under the precision-recall curve).
    :param brier: Brier score (lower is better).
    :param ece: Expected calibration error.
    :param decile_lift: Lift in the top decile relative to the overall churn rate.
    """

    auc: float
    pr_auc: float
    brier: float
    ece: float
    decile_lift: float

    def as_dict(self) -> Mapping[str, float]:
        """Return the metrics as a plain ``dict`` for MLflow logging.

        :returns: Mapping from metric name to value.
        """
        return {
            "auc": self.auc,
            "pr_auc": self.pr_auc,
            "brier": self.brier,
            "ece": self.ece,
            "decile_lift": self.decile_lift,
        }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 20
) -> float:
    """Compute the expected calibration error with equal-width probability bins.

    :param y_true: Binary ground truth (0/1).
    :param y_prob: Predicted probabilities in ``[0, 1]``.
    :param n_bins: Number of equal-width probability bins.
    :returns: ECE in ``[0, 1]``.
    :raises ValueError: If shapes mismatch or ``n_bins < 1``.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must share the same length")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        weight = mask.sum() / n
        bin_acc = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += weight * abs(bin_acc - bin_conf)
    return float(ece)


def lift_at_decile(y_true: np.ndarray, y_prob: np.ndarray, *, decile: int = 1) -> float:
    """Compute the lift in the top ``decile`` (1 = top 10%) versus base rate.

    :param y_true: Binary ground truth.
    :param y_prob: Predicted probabilities.
    :param decile: Decile rank, 1 for the top 10%, 2 for the top 20%, etc.
    :returns: Ratio of in-decile churn rate to overall churn rate.
    :raises ValueError: If ``decile`` falls outside ``[1, 10]``.
    """
    if not 1 <= decile <= 10:
        raise ValueError("decile must lie in [1, 10]")
    n = len(y_true)
    cutoff = max(int(np.ceil(n * decile / 10)), 1)
    order = np.argsort(-y_prob, kind="stable")
    top = order[:cutoff]
    base_rate = float(y_true.mean())
    if base_rate <= 0:
        return float("nan")
    return float(y_true[top].mean()) / base_rate


def risk_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, *, n_calibration_bins: int = 20
) -> RiskMetrics:
    """Compute the standard suite of risk metrics in one call.

    :param y_true: Binary ground truth.
    :param y_prob: Predicted probabilities.
    :param n_calibration_bins: Number of bins for ECE.
    :returns: A :class:`RiskMetrics` bundle.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)
    return RiskMetrics(
        auc=float(roc_auc_score(y_true_arr, y_prob_arr)),
        pr_auc=float(average_precision_score(y_true_arr, y_prob_arr)),
        brier=float(brier_score_loss(y_true_arr, y_prob_arr)),
        ece=expected_calibration_error(y_true_arr, y_prob_arr, n_bins=n_calibration_bins),
        decile_lift=lift_at_decile(y_true_arr, y_prob_arr, decile=1),
    )


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, *, n_bins: int = 20) -> pd.DataFrame:
    """Return a per-bin calibration table for plotting reliability diagrams.

    :param y_true: Binary ground truth.
    :param y_prob: Predicted probabilities.
    :param n_bins: Number of equal-width bins.
    :returns: Frame with columns ``bin``, ``count``, ``mean_predicted``,
        ``empirical_rate``.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "count": int(mask.sum()),
                "mean_predicted": float(np.mean(y_prob[mask])),
                "empirical_rate": float(np.mean(y_true[mask])),
            }
        )
    return pd.DataFrame(rows)
