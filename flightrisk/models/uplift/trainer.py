from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from flightrisk.eval.uplift_metrics import UpliftMetrics, qini_curve, uplift_metrics
from flightrisk.models.uplift.causal_forest import CausalForestParams, CausalForestUpliftModel
from flightrisk.models.uplift.meta_learners import (
    DRLearner,
    LightGBMUpliftParams,
    TLearner,
    XLearner,
)
from flightrisk.utils.logging import get_logger

_log = get_logger(__name__)

UpliftEstimator = Literal["t_learner", "x_learner", "dr_learner", "causal_forest"]


class _UpliftModel(Protocol):
    """Protocol shared by every uplift wrapper."""

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> _UpliftModel:
        """Fit the uplift model."""
        ...

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Predict per-row uplift."""
        ...


@dataclass
class UpliftTrainingResult:
    """Bundle returned by :func:`train_uplift_model`.

    :param model: The fitted estimator.
    :param metrics: Aggregated uplift metrics.
    :param qini_curve: Frame describing the Qini curve on the test slice.
    :param uplift_test: Per-row uplift on the test slice.
    :param feature_names: Feature ordering used at fit time.
    """

    model: object
    metrics: UpliftMetrics
    qini_curve: pd.DataFrame
    uplift_test: np.ndarray
    feature_names: list[str] = field(default_factory=list)


def _build_estimator(name: UpliftEstimator) -> _UpliftModel:
    """Instantiate the configured uplift estimator.

    :param name: One of ``"t_learner"``, ``"x_learner"``, ``"dr_learner"``,
        ``"causal_forest"``.
    :returns: A fresh, unfitted wrapper.
    :raises ValueError: If ``name`` is unknown.
    """
    if name == "t_learner":
        return TLearner(LightGBMUpliftParams())
    if name == "x_learner":
        return XLearner(LightGBMUpliftParams())
    if name == "dr_learner":
        return DRLearner(LightGBMUpliftParams())
    if name == "causal_forest":
        return CausalForestUpliftModel(CausalForestParams())
    raise ValueError(f"unknown uplift estimator: {name!r}")


def train_uplift_model(
    features: pd.DataFrame,
    treatment: pd.Series | np.ndarray,
    outcome: pd.Series | np.ndarray,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    estimator: UpliftEstimator = "t_learner",
    k_percentiles: tuple[int, ...] = (10, 20, 30),
) -> UpliftTrainingResult:
    """Train an uplift model end to end on the Orange Belgium RCT.

    :param features: Wide feature frame.
    :param treatment: 0/1 treatment indicator.
    :param outcome: 0/1 outcome indicator.
    :param train_idx: Row positions used for fitting.
    :param test_idx: Row positions used for evaluation.
    :param estimator: One of the supported uplift estimators.
    :param k_percentiles: Top-percentiles for ``uplift_at_k``.
    :returns: An :class:`UpliftTrainingResult` with the fitted model and metrics.
    """
    X_full = features.reset_index(drop=True)
    treatment_arr = np.asarray(treatment).astype(int).ravel()
    outcome_arr = np.asarray(outcome).astype(int).ravel()

    X_train, t_train, y_train = (
        X_full.iloc[train_idx],
        treatment_arr[train_idx],
        outcome_arr[train_idx],
    )
    X_test, t_test, y_test = X_full.iloc[test_idx], treatment_arr[test_idx], outcome_arr[test_idx]

    _log.info(
        "training uplift[%s] on %d / %d (train/test); treated rate train=%.3f test=%.3f",
        estimator,
        len(X_train),
        len(X_test),
        float(t_train.mean()),
        float(t_test.mean()),
    )
    model = _build_estimator(estimator)
    model.fit(X_train, t_train, y_train)
    uplift = model.predict_uplift(X_test)

    metrics = uplift_metrics(
        uplift=uplift,
        treatment=t_test,
        outcome=y_test,
        k_percentiles=k_percentiles,
    )
    curve = qini_curve(uplift, t_test, y_test)
    _log.info(
        "uplift metrics qini=%.4f auuc=%.4f uplift@%d%%=%.4f",
        metrics.qini,
        metrics.auuc,
        k_percentiles[0],
        metrics.uplift_at_k[k_percentiles[0]],
    )
    return UpliftTrainingResult(
        model=model,
        metrics=metrics,
        qini_curve=curve,
        uplift_test=uplift,
        feature_names=X_full.columns.tolist(),
    )


def save_qini_plot(curve: pd.DataFrame, *, output: Path) -> Path:
    """Render and save a Qini curve from the curve frame.

    :param curve: Output of :func:`flightrisk.eval.uplift_metrics.qini_curve`.
    :param output: Destination PNG path.
    :returns: The path written.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["rank"], curve["qini"], label="model")
    ax.plot(
        [curve["rank"].iloc[0], curve["rank"].iloc[-1]],
        [0, curve["qini"].iloc[-1]],
        linestyle="--",
        color="grey",
        label="random",
    )
    ax.set_xlabel("ranked observations")
    ax.set_ylabel("incremental retained outcomes")
    ax.set_title("Qini curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
