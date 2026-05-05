from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass(frozen=True)
class XGBoostRiskParams:
    """Hyperparameters for the XGBoost risk learner.

    :param objective: XGBoost objective. Defaults to ``"binary:logistic"``.
    :param eta: Boosting learning rate.
    :param max_depth: Maximum tree depth.
    :param min_child_weight: Minimum sum of instance weights per leaf.
    :param subsample: Row subsample ratio per iteration.
    :param colsample_bytree: Column subsample ratio per tree.
    :param reg_lambda: L2 regularisation strength.
    :param n_estimators: Maximum boosting rounds.
    :param early_stopping_rounds: Patience used when a validation set is given.
    :param tree_method: ``"hist"`` is fast and works on CPU and GPU.
    :param eval_metric: Default evaluation metric.
    """

    objective: str = "binary:logistic"
    eta: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 5.0
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_lambda: float = 1.0
    n_estimators: int = 800
    early_stopping_rounds: int = 50
    tree_method: str = "hist"
    eval_metric: str = "logloss"

    def as_native(self) -> Mapping[str, Any]:
        """Return the params in the form XGBoost expects.

        :returns: Mapping consumable by :class:`xgboost.XGBClassifier`.
        """
        return {
            "objective": self.objective,
            "learning_rate": self.eta,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "n_estimators": self.n_estimators,
            "tree_method": self.tree_method,
            "eval_metric": self.eval_metric,
        }


class XGBoostRiskModel:
    """Thin wrapper around :class:`xgboost.XGBClassifier` for the risk track."""

    def __init__(self, params: XGBoostRiskParams | None = None) -> None:
        """Initialise the wrapper.

        :param params: Optional hyperparameters; defaults are model-ready.
        """
        self.params = params or XGBoostRiskParams()
        self._estimator: xgb.XGBClassifier | None = None
        self._feature_names: list[str] | None = None

    @property
    def estimator(self) -> xgb.XGBClassifier:
        """Return the underlying fitted estimator.

        :returns: The fitted :class:`xgboost.XGBClassifier`.
        :raises RuntimeError: If the model has not been fitted.
        """
        if self._estimator is None:
            raise RuntimeError("XGBoostRiskModel has not been fitted")
        return self._estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostRiskModel":
        """Fit the model with optional early stopping on a validation slice.

        :param X: Training features.
        :param y: Training labels (0/1).
        :param X_val: Optional validation features.
        :param y_val: Optional validation labels.
        :returns: ``self`` for chaining.
        """
        self._feature_names = X.columns.tolist()
        kwargs: dict[str, Any] = dict(self.params.as_native())
        if X_val is not None and y_val is not None:
            kwargs["early_stopping_rounds"] = self.params.early_stopping_rounds
        estimator = xgb.XGBClassifier(**kwargs)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val[self._feature_names], y_val)]
        estimator.fit(X[self._feature_names], y, eval_set=eval_set, verbose=False)
        self._estimator = estimator
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the positive-class probability.

        :param X: Features whose columns must include the ones seen at fit time.
        :returns: 1-D array of probabilities in ``[0, 1]``.
        :raises RuntimeError: If the model has not been fitted.
        """
        if self._feature_names is None:
            raise RuntimeError("XGBoostRiskModel has not been fitted")
        return self.estimator.predict_proba(X[self._feature_names])[:, 1]
