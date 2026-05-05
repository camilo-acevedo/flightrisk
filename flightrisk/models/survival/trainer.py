from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from flightrisk.eval.survival_metrics import SurvivalMetrics, survival_metrics
from flightrisk.models.survival.cox_model import CoxParams, CoxSurvivalModel
from flightrisk.models.survival.rsf_model import RSFParams, RSFSurvivalModel
from flightrisk.utils.logging import get_logger

_log = get_logger(__name__)

SurvivalEstimator = Literal["cox", "rsf"]


class _SurvivalModel(Protocol):
    """Protocol satisfied by every survival wrapper."""

    def fit(self, X: pd.DataFrame, durations: np.ndarray, events: np.ndarray) -> _SurvivalModel:
        """Fit the model in place and return self."""
        ...

    def hazard_at_horizon(self, X: pd.DataFrame, *, horizon_days: int) -> np.ndarray:
        """Return ``1 - S(horizon)`` per row."""
        ...


@dataclass
class SurvivalTrainingResult:
    """Bundle returned by :func:`train_survival_model`.

    :param model: The fitted estimator.
    :param metrics: Aggregated survival metrics.
    :param horizons_days: Horizons evaluated.
    :param survival_at_horizons: ``(n_test, len(horizons))`` S(t) matrix.
    :param feature_names: Feature ordering used at fit time.
    """

    model: object
    metrics: SurvivalMetrics
    horizons_days: np.ndarray
    survival_at_horizons: np.ndarray
    feature_names: list[str] = field(default_factory=list)


def _build_estimator(name: SurvivalEstimator) -> _SurvivalModel:
    """Instantiate the configured survival estimator.

    :param name: ``"cox"`` or ``"rsf"``.
    :returns: A fresh, unfitted wrapper.
    :raises ValueError: If ``name`` is unknown.
    """
    if name == "cox":
        return CoxSurvivalModel(CoxParams())
    if name == "rsf":
        return RSFSurvivalModel(RSFParams())
    raise ValueError(f"unknown survival estimator: {name!r}")


def _drop_id_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``msno`` ID column before fitting numeric models.

    :param features: Source feature frame.
    :returns: A copy without ID columns.
    """
    drop = [c for c in features.columns if c == "msno"]
    return features.drop(columns=drop)


def _to_numeric(features: pd.DataFrame) -> pd.DataFrame:
    """Coerce features to ``float64`` and drop fully-NaN columns.

    Non-numeric columns are one-hot encoded with the first category as the
    baseline so the resulting frame is suitable for both Cox and RSF.

    :param features: Source feature frame.
    :returns: Numeric, dense feature frame.
    """
    df = features.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dummy_na=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.fillna(df.median(numeric_only=True))
    return df.astype("float64")


def train_survival_model(
    features: pd.DataFrame,
    durations: np.ndarray,
    events: np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    estimator: SurvivalEstimator = "rsf",
    horizons_days: tuple[int, ...] = (30, 60, 90),
) -> SurvivalTrainingResult:
    """Train a survival model end to end.

    :param features: Wide feature frame (with optional ``msno`` ID column).
    :param durations: Aligned follow-up times in days.
    :param events: Aligned 0/1 event indicators.
    :param train_idx: Row positions used for fitting.
    :param test_idx: Row positions used for evaluation.
    :param estimator: ``"cox"`` or ``"rsf"``.
    :param horizons_days: Horizons evaluated for the metric suite.
    :returns: A :class:`SurvivalTrainingResult` with the fitted model.
    """
    X_full = _to_numeric(_drop_id_columns(features)).reset_index(drop=True)
    y_dur = np.asarray(durations).astype(float).ravel()
    y_evt = np.asarray(events).astype(int).ravel()

    X_train, dur_train, evt_train = X_full.iloc[train_idx], y_dur[train_idx], y_evt[train_idx]
    X_test, dur_test, evt_test = X_full.iloc[test_idx], y_dur[test_idx], y_evt[test_idx]

    _log.info(
        "training survival[%s] on %d / %d (train/test); event rate train=%.3f test=%.3f",
        estimator,
        len(X_train),
        len(X_test),
        float(evt_train.mean()),
        float(evt_test.mean()),
    )
    model = _build_estimator(estimator)
    model.fit(X_train, dur_train, evt_train)

    horizons = np.asarray(horizons_days, dtype=float)
    if hasattr(model, "survival_at_times"):
        survival = model.survival_at_times(X_test, horizons)
    else:
        survival = np.column_stack(
            [1.0 - model.hazard_at_horizon(X_test, horizon_days=int(h)) for h in horizons]
        )
    risk_scores = -np.log(np.clip(survival[:, -1], 1e-9, 1.0))

    metrics = survival_metrics(
        train_durations=dur_train,
        train_events=evt_train,
        test_durations=dur_test,
        test_events=evt_test,
        risk_scores=risk_scores,
        survival_at_horizons=survival,
        horizons=horizons,
    )
    _log.info(
        "survival metrics c_index=%.4f td_auc_mean=%.4f integrated_brier=%.4f",
        metrics.c_index,
        metrics.time_dependent_auc_mean,
        metrics.integrated_brier,
    )
    return SurvivalTrainingResult(
        model=model,
        metrics=metrics,
        horizons_days=horizons,
        survival_at_horizons=survival,
        feature_names=X_full.columns.tolist(),
    )


def save_survival_curve_plot(
    survival_at_horizons: np.ndarray, horizons_days: np.ndarray, *, output: Path, n_curves: int = 30
) -> Path:
    """Render a small selection of S(t) curves as a sanity-check plot.

    :param survival_at_horizons: ``(n_samples, len(horizons))`` matrix of S(t).
    :param horizons_days: Matching horizons in days.
    :param output: Destination PNG path.
    :param n_curves: Number of randomly-selected curves to draw.
    :returns: The path written.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from flightrisk.eval.style import PALETTE, annotate_axis, apply_style

    apply_style()
    rng = np.random.default_rng(0)
    n = survival_at_horizons.shape[0]
    sample_idx = rng.choice(n, size=min(n_curves, n), replace=False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for i in sample_idx:
        ax.step(
            horizons_days,
            survival_at_horizons[i],
            where="post",
            color=PALETTE["survival"],
            alpha=0.18,
            linewidth=1.4,
        )
    median_curve = np.median(survival_at_horizons, axis=0)
    ax.step(
        horizons_days,
        median_curve,
        where="post",
        color=PALETTE["accent"],
        linewidth=2.6,
        label="cohort median",
    )
    ax.set_xlabel("Days since cutoff")
    ax.set_ylabel("S(t)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Survival — S(t) for {len(sample_idx)} sampled customers", loc="left")
    ax.grid(True, axis="both")
    ax.legend(loc="lower left", framealpha=0.9)
    annotate_axis(ax, source="flightrisk · Track B")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
