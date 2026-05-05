from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UpliftMetrics:
    """Metrics emitted by the uplift track.

    :param qini: Qini coefficient (area between Qini curve and random line).
    :param auuc: Area under the uplift curve.
    :param uplift_at_k: Mapping from ``k`` (percent) to estimated uplift in the
        top-``k%`` slice.
    """

    qini: float
    auuc: float
    uplift_at_k: Mapping[int, float]

    def as_dict(self) -> Mapping[str, float]:
        """Return the metrics as a flat ``dict`` for MLflow logging.

        :returns: Mapping of scalar metrics, with one entry per ``k``.
        """
        out = {"qini": self.qini, "auuc": self.auuc}
        for k, value in self.uplift_at_k.items():
            out[f"uplift_at_{k}"] = float(value)
        return out


def _ordered_frame(uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    """Sort observations by descending uplift and return a tidy frame.

    :param uplift: Per-row uplift score.
    :param treatment: 0/1 treatment indicator.
    :param outcome: 0/1 outcome.
    :returns: Sorted frame with ``rank``, ``treatment``, ``outcome``.
    """
    order = np.argsort(-np.asarray(uplift), kind="stable")
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(order) + 1),
            "treatment": np.asarray(treatment).astype(int)[order],
            "outcome": np.asarray(outcome).astype(int)[order],
        }
    )


def qini_curve(uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    """Build the Qini curve in cumulative form.

    :param uplift: Per-row uplift score.
    :param treatment: 0/1 treatment indicator.
    :param outcome: 0/1 outcome.
    :returns: Frame with ``rank``, ``cum_treated_outcomes``, ``cum_control_outcomes``,
        ``cum_treated_n``, ``cum_control_n``, ``qini`` columns.
    """
    df = _ordered_frame(uplift, treatment, outcome)
    df["cum_treated_outcomes"] = (df["outcome"] * df["treatment"]).cumsum()
    df["cum_control_outcomes"] = (df["outcome"] * (1 - df["treatment"])).cumsum()
    df["cum_treated_n"] = df["treatment"].cumsum()
    df["cum_control_n"] = (1 - df["treatment"]).cumsum()
    n_treated_total = df["cum_treated_n"].iloc[-1]
    if n_treated_total == 0:
        df["qini"] = np.nan
        return df
    df["qini"] = df["cum_treated_outcomes"] - df["cum_control_outcomes"] * df[
        "cum_treated_n"
    ] / np.maximum(df["cum_control_n"], 1)
    return df


def qini_coefficient(uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
    """Return the Qini coefficient against the random-targeting baseline.

    :param uplift: Per-row uplift score.
    :param treatment: 0/1 treatment indicator.
    :param outcome: 0/1 outcome.
    :returns: Float Qini coefficient.
    """
    df = qini_curve(uplift, treatment, outcome)
    if df["qini"].isna().all():
        return float("nan")
    n = len(df)
    actual_area = float(np.trapezoid(df["qini"].values, df["rank"].values))
    final = float(df["qini"].iloc[-1])
    random_area = final * n / 2.0
    return (actual_area - random_area) / float(max(n * n, 1))


def auuc(uplift: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
    """Compute the area under the uplift curve.

    :param uplift: Per-row uplift score.
    :param treatment: 0/1 treatment indicator.
    :param outcome: 0/1 outcome.
    :returns: AUUC normalised by the number of observations.
    """
    df = qini_curve(uplift, treatment, outcome)
    treated_rate = df["cum_treated_outcomes"] / np.maximum(df["cum_treated_n"], 1)
    control_rate = df["cum_control_outcomes"] / np.maximum(df["cum_control_n"], 1)
    lift = (treated_rate - control_rate) * df["rank"]
    return float(np.trapezoid(lift.values, df["rank"].values) / max(len(df), 1))


def uplift_at_k(
    uplift: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    k_percent: float,
) -> float:
    """Estimate the uplift in the top-``k%`` ranked slice.

    :param uplift: Per-row uplift score.
    :param treatment: 0/1 treatment indicator.
    :param outcome: 0/1 outcome.
    :param k_percent: Top-percentile cutoff in ``(0, 100]``.
    :returns: Difference between treated outcome rate and control outcome rate
        within the top slice; ``nan`` if either arm is empty in the slice.
    :raises ValueError: If ``k_percent`` is outside ``(0, 100]``.
    """
    if not 0 < k_percent <= 100:
        raise ValueError("k_percent must lie in (0, 100]")
    df = _ordered_frame(uplift, treatment, outcome)
    cutoff = max(int(np.ceil(len(df) * k_percent / 100.0)), 1)
    top = df.iloc[:cutoff]
    treated = top[top["treatment"] == 1]
    control = top[top["treatment"] == 0]
    if treated.empty or control.empty:
        return float("nan")
    return float(treated["outcome"].mean() - control["outcome"].mean())


def uplift_metrics(
    *,
    uplift: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    k_percentiles: tuple[int, ...] = (10, 20, 30),
) -> UpliftMetrics:
    """Compute the standard uplift metric suite in one call.

    :param uplift: Per-row uplift score.
    :param treatment: 0/1 treatment indicator.
    :param outcome: 0/1 outcome.
    :param k_percentiles: Top-percentiles for ``uplift_at_k``.
    :returns: An :class:`UpliftMetrics` bundle.
    """
    return UpliftMetrics(
        qini=qini_coefficient(uplift, treatment, outcome),
        auuc=auuc(uplift, treatment, outcome),
        uplift_at_k={
            int(k): uplift_at_k(uplift, treatment, outcome, k_percent=float(k))
            for k in k_percentiles
        },
    )
