from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from flightrisk.eval.metrics import RiskMetrics, calibration_table, risk_metrics
from flightrisk.models.risk.calibration import CalibratedRiskModel, CalibrationMethod
from flightrisk.models.risk.lightgbm_model import LightGBMRiskModel, LightGBMRiskParams
from flightrisk.models.risk.xgboost_model import XGBoostRiskModel, XGBoostRiskParams
from flightrisk.utils.logging import get_logger


class _RiskScorer(Protocol):
    """Protocol satisfied by every wrapper that produces probabilities."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities."""
        ...


class _RiskBase(_RiskScorer, Protocol):
    """Base risk wrapper: scorer plus a fit method with early stopping support."""

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        X_val: pd.DataFrame | None = ...,
        y_val: np.ndarray | None = ...,
    ) -> _RiskBase:
        """Fit the model in place and return self."""
        ...


_log = get_logger(__name__)


@dataclass
class RiskTrainingResult:
    """Bundle of artifacts produced by :func:`train_risk_model`.

    :param model: The fitted estimator (calibrated when ``calibration`` is set).
    :param metrics: Evaluation metrics on the test slice.
    :param calibration: Calibration table on the test slice.
    :param feature_names: Feature ordering used at fit time.
    """

    model: object
    metrics: RiskMetrics
    calibration: pd.DataFrame
    feature_names: list[str] = field(default_factory=list)


def _build_base(estimator: str) -> _RiskBase:
    """Instantiate the configured base estimator.

    :param estimator: Either ``"lightgbm"`` or ``"xgboost"``.
    :returns: A fresh, unfitted wrapper instance.
    :raises ValueError: If ``estimator`` is unknown.
    """
    if estimator == "lightgbm":
        return LightGBMRiskModel(LightGBMRiskParams())
    if estimator == "xgboost":
        return XGBoostRiskModel(XGBoostRiskParams())
    raise ValueError(f"unknown risk estimator: {estimator!r}")


def _drop_id_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Drop the ``msno`` ID column (and any obvious ID columns) before fitting.

    :param features: Source feature frame.
    :returns: The frame without ID columns; original is unchanged.
    """
    drop = [c for c in features.columns if c == "msno"]
    return features.drop(columns=drop)


def _to_numeric(features: pd.DataFrame) -> pd.DataFrame:
    """Coerce features to ``float64`` and one-hot encode categorical columns.

    LightGBM and XGBoost both reject ``object``-dtype columns; one-hot
    encoding with the first category as baseline keeps the matrix dense and
    numeric.

    :param features: Source feature frame.
    :returns: Numeric, dense feature frame.
    """
    df = features.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dummy_na=False)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df.astype("float64")


def train_risk_model(
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    estimator: str = "lightgbm",
    calibration: CalibrationMethod | None = "isotonic",
) -> RiskTrainingResult:
    """Train a risk model end to end with optional calibration on validation.

    :param features: Wide feature frame (with optional ``msno`` ID column).
    :param labels: Aligned 0/1 churn labels.
    :param train_idx: Row positions used for training.
    :param val_idx: Row positions used for early stopping and calibration.
    :param test_idx: Row positions held out for final metrics.
    :param estimator: ``"lightgbm"`` or ``"xgboost"``.
    :param calibration: ``"isotonic"``, ``"platt"``, or ``None`` to skip.
    :returns: A :class:`RiskTrainingResult` with the fitted model and metrics.
    """
    X_full = _to_numeric(_drop_id_columns(features)).reset_index(drop=True)
    y_full = np.asarray(labels).astype(int).ravel()

    X_train, y_train = X_full.iloc[train_idx], y_full[train_idx]
    X_val, y_val = X_full.iloc[val_idx], y_full[val_idx]
    X_test, y_test = X_full.iloc[test_idx], y_full[test_idx]

    _log.info(
        "training risk[%s] on %d / %d / %d (train/val/test)",
        estimator,
        len(X_train),
        len(X_val),
        len(X_test),
    )
    base = _build_base(estimator)
    base.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    model: _RiskScorer
    if calibration is not None:
        _log.info("fitting %s calibrator on validation slice", calibration)
        model = CalibratedRiskModel(base, method=calibration).fit(X_val, y_val)
    else:
        model = base

    test_proba = model.predict_proba(X_test)
    metrics = risk_metrics(y_test, test_proba)
    calib = calibration_table(y_test, test_proba)
    _log.info(
        "risk metrics auc=%.4f pr_auc=%.4f brier=%.4f ece=%.4f decile_lift=%.2f",
        metrics.auc,
        metrics.pr_auc,
        metrics.brier,
        metrics.ece,
        metrics.decile_lift,
    )
    return RiskTrainingResult(
        model=model,
        metrics=metrics,
        calibration=calib,
        feature_names=X_full.columns.tolist(),
    )


def save_calibration_plot(table: pd.DataFrame, *, output: Path) -> Path:
    """Render and save a reliability diagram from a calibration table.

    :param table: Output of :func:`flightrisk.eval.metrics.calibration_table`.
    :param output: Destination PNG path.
    :returns: The path written.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from flightrisk.eval.style import PALETTE, annotate_axis, apply_style

    apply_style()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color=PALETTE["perfect"],
        linewidth=1.2,
        label="perfect calibration",
    )
    ax.scatter(
        table["mean_predicted"],
        table["empirical_rate"],
        s=table["count"].clip(upper=240) * 1.1,
        c=PALETTE["risk"],
        alpha=0.78,
        edgecolors="white",
        linewidths=1.0,
        label="bins (size = count)",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical churn rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Risk — Reliability diagram", loc="left")
    ax.grid(True, axis="both")
    ax.legend(loc="upper left", framealpha=0.9)
    annotate_axis(ax, source="flightrisk · Track A")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return output
