from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LightGBMRiskParams:
    """Hyperparameters for the LightGBM risk learner.

    :param objective: LightGBM objective. Defaults to ``"binary"``.
    :param learning_rate: Boosting learning rate.
    :param num_leaves: Maximum number of leaves per tree.
    :param max_depth: Maximum tree depth (``-1`` means unbounded).
    :param min_data_in_leaf: Minimum samples per leaf.
    :param feature_fraction: Column subsample ratio per tree.
    :param bagging_fraction: Row subsample ratio per iteration.
    :param bagging_freq: Iterations between bagging refreshes.
    :param lambda_l2: L2 regularisation strength.
    :param n_estimators: Maximum boosting rounds.
    :param early_stopping_rounds: Patience used when a validation set is given.
    :param verbose: LightGBM verbosity flag.
    """

    objective: str = "binary"
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.85
    bagging_fraction: float = 0.85
    bagging_freq: int = 5
    lambda_l2: float = 1.0
    n_estimators: int = 800
    early_stopping_rounds: int = 50
    verbose: int = -1

    def as_native(self) -> Mapping[str, Any]:
        """Return the params in the form LightGBM expects.

        :returns: Mapping consumable by :class:`lightgbm.LGBMClassifier`.
        """
        return {
            "objective": self.objective,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "lambda_l2": self.lambda_l2,
            "n_estimators": self.n_estimators,
            "verbose": self.verbose,
        }


class LightGBMRiskModel:
    """Thin wrapper around :class:`lightgbm.LGBMClassifier` for the risk track.

    The wrapper records the feature ordering used at fit time so prediction
    matches the training schema even when the upstream feature builder shuffles
    columns or drops auxiliary IDs.
    """

    def __init__(self, params: LightGBMRiskParams | None = None) -> None:
        """Initialise the wrapper.

        :param params: Optional hyperparameters; defaults are model-ready.
        """
        self.params = params or LightGBMRiskParams()
        self._estimator: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] | None = None

    @property
    def estimator(self) -> lgb.LGBMClassifier:
        """Return the underlying fitted estimator.

        :returns: The fitted :class:`lightgbm.LGBMClassifier`.
        :raises RuntimeError: If the model has not been fitted.
        """
        if self._estimator is None:
            raise RuntimeError("LightGBMRiskModel has not been fitted")
        return self._estimator

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
        categorical_features: list[str] | None = None,
    ) -> "LightGBMRiskModel":
        """Fit the model with optional early stopping on a validation slice.

        :param X: Training features.
        :param y: Training labels (0/1).
        :param X_val: Optional validation features.
        :param y_val: Optional validation labels.
        :param categorical_features: Optional categorical column names.
        :returns: ``self`` for chaining.
        """
        self._feature_names = X.columns.tolist()
        params = self.params.as_native()
        estimator = lgb.LGBMClassifier(**params)

        callbacks = [lgb.log_evaluation(period=0)]
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val[self._feature_names], y_val)]
            callbacks.append(lgb.early_stopping(self.params.early_stopping_rounds, verbose=False))

        estimator.fit(
            X[self._feature_names],
            y,
            eval_set=eval_set,
            categorical_feature=categorical_features or "auto",
            callbacks=callbacks,
        )
        self._estimator = estimator
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the positive-class probability.

        :param X: Features whose columns must include the ones seen at fit time.
        :returns: 1-D array of probabilities in ``[0, 1]``.
        :raises RuntimeError: If the model has not been fitted.
        """
        if self._feature_names is None:
            raise RuntimeError("LightGBMRiskModel has not been fitted")
        return self.estimator.predict_proba(X[self._feature_names])[:, 1]
