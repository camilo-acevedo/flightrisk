from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml

from flightrisk.eval.metrics import RiskMetrics, risk_metrics
from flightrisk.models.risk.calibration import CalibratedRiskModel
from flightrisk.models.risk.lightgbm_model import LightGBMRiskModel, LightGBMRiskParams
from flightrisk.models.risk.trainer import _drop_id_columns, _to_numeric
from flightrisk.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass(frozen=True)
class SweepSpace:
    """Optuna search space configuration loaded from YAML.

    :param name: Human-readable sweep name; logged to MLflow.
    :param direction: ``"maximize"`` or ``"minimize"``.
    :param metric: Metric to optimise; one of the keys returned by
        :class:`flightrisk.eval.metrics.RiskMetrics.as_dict`.
    :param n_trials: Maximum number of Optuna trials.
    :param timeout_seconds: Optional global timeout in seconds.
    :param search_space: Mapping from parameter name to a ``{type, low, high}``
        block (with optional ``log`` for log-uniform floats).
    """

    name: str
    direction: str
    metric: str
    n_trials: int
    timeout_seconds: int | None
    search_space: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_sweep_space(path: str | Path) -> SweepSpace:
    """Load a sweep YAML into a :class:`SweepSpace` dataclass.

    :param path: Path to the YAML file (e.g. ``configs/sweep/risk_lgbm.yaml``).
    :returns: A populated :class:`SweepSpace`.
    :raises FileNotFoundError: If ``path`` is missing.
    """
    cfg = yaml.safe_load(Path(path).read_text())
    return SweepSpace(
        name=cfg.get("name", "risk_lgbm_sweep"),
        direction=cfg.get("direction", "maximize"),
        metric=cfg.get("metric", "auc"),
        n_trials=int(cfg.get("n_trials", 50)),
        timeout_seconds=cfg.get("timeout_seconds"),
        search_space=cfg["search_space"],
    )


def _suggest(trial: optuna.Trial, space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Translate the YAML search-space block into Optuna ``trial.suggest_*`` calls.

    :param trial: The Optuna trial.
    :param space: Mapping from parameter name to its config block.
    :returns: A dict of sampled hyperparameters.
    :raises ValueError: If a parameter has an unknown ``type``.
    """
    params: dict[str, Any] = {}
    for key, cfg in space.items():
        kind = cfg["type"]
        if kind == "float":
            params[key] = trial.suggest_float(
                key, float(cfg["low"]), float(cfg["high"]), log=bool(cfg.get("log", False))
            )
        elif kind == "int":
            params[key] = trial.suggest_int(key, int(cfg["low"]), int(cfg["high"]))
        elif kind == "categorical":
            params[key] = trial.suggest_categorical(key, list(cfg["choices"]))
        else:
            raise ValueError(f"unknown search-space type: {kind!r}")
    return params


@dataclass
class SweepResult:
    """Bundle returned by :func:`run_risk_sweep`.

    :param best_params: Best hyperparameters found.
    :param best_metric: Best metric value (per the sweep direction).
    :param best_metrics: All risk metrics on the held-out validation slice for
        the best trial.
    :param study: The underlying Optuna study (kept for inspection).
    """

    best_params: dict[str, Any]
    best_metric: float
    best_metrics: RiskMetrics
    study: optuna.Study


def run_risk_sweep(
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    space: SweepSpace,
    seed: int = 1337,
    mlflow_callback: Any | None = None,
) -> SweepResult:
    """Optimise LightGBM risk hyperparameters on a held-out validation slice.

    :param features: Wide feature frame (with optional ``msno`` ID column).
    :param labels: Aligned 0/1 churn labels.
    :param train_idx: Row positions used for fitting each trial's model.
    :param val_idx: Row positions used both for early stopping and for
        scoring the trial's final metric.
    :param space: Sweep configuration loaded by :func:`load_sweep_space`.
    :param seed: Reproducibility seed for the sampler.
    :param mlflow_callback: Optional :class:`optuna.integration.MLflowCallback`
        instance; when provided, every trial is logged as an MLflow run.
    :returns: A :class:`SweepResult` with the best params, best metric, and
        the underlying study object.
    """
    X_full = _to_numeric(_drop_id_columns(features)).reset_index(drop=True)
    y_full = np.asarray(labels).astype(int).ravel()

    X_train, y_train = X_full.iloc[train_idx], y_full[train_idx]
    X_val, y_val = X_full.iloc[val_idx], y_full[val_idx]

    metric_name = space.metric

    def objective(trial: optuna.Trial) -> float:
        params = _suggest(trial, space.search_space)
        model = LightGBMRiskModel(LightGBMRiskParams(**params))
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        calibrated = CalibratedRiskModel(model, method="isotonic").fit(X_val, y_val)
        proba = calibrated.predict_proba(X_val)
        metrics = risk_metrics(y_val, proba)
        return float(metrics.as_dict()[metric_name])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=space.direction, sampler=sampler, study_name=space.name)
    callbacks = [mlflow_callback] if mlflow_callback is not None else None
    study.optimize(
        objective,
        n_trials=space.n_trials,
        timeout=space.timeout_seconds,
        callbacks=callbacks,
        show_progress_bar=False,
    )

    best_params = dict(study.best_params)
    best_model = LightGBMRiskModel(LightGBMRiskParams(**best_params))
    best_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    best_calibrated = CalibratedRiskModel(best_model, method="isotonic").fit(X_val, y_val)
    best_metrics = risk_metrics(y_val, best_calibrated.predict_proba(X_val))
    _log.info(
        "sweep best %s=%.4f params=%s",
        metric_name,
        study.best_value,
        best_params,
    )
    return SweepResult(
        best_params=best_params,
        best_metric=float(study.best_value),
        best_metrics=best_metrics,
        study=study,
    )
