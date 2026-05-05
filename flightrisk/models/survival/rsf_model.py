from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest

from flightrisk.models.survival.labels import to_structured_array


@dataclass(frozen=True)
class RSFParams:
    """Hyperparameters for the Random Survival Forest.

    :param n_estimators: Number of trees in the forest.
    :param max_depth: Maximum tree depth (``None`` means unbounded).
    :param min_samples_leaf: Minimum samples per leaf.
    :param min_samples_split: Minimum samples per split.
    :param max_features: Number of features considered at each split.
    :param n_jobs: Parallelism level (``-1`` means use all cores).
    :param random_state: Seed for tree bagging.
    """

    n_estimators: int = 400
    max_depth: int | None = None
    min_samples_leaf: int = 20
    min_samples_split: int = 40
    max_features: str = "sqrt"
    n_jobs: int = -1
    random_state: int = 1337

    def as_native(self) -> Mapping[str, Any]:
        """Return the params in the form scikit-survival expects.

        :returns: Mapping consumable by :class:`sksurv.ensemble.RandomSurvivalForest`.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }


class RSFSurvivalModel:
    """Wrapper around :class:`sksurv.ensemble.RandomSurvivalForest`."""

    def __init__(self, params: RSFParams | None = None) -> None:
        """Initialise the wrapper.

        :param params: Optional hyperparameters.
        """
        self.params = params or RSFParams()
        self._estimator: RandomSurvivalForest | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        durations: np.ndarray,
        events: np.ndarray,
    ) -> RSFSurvivalModel:
        """Fit the random survival forest.

        :param X: Numeric feature frame.
        :param durations: Durations or follow-up times.
        :param events: 0/1 event indicators.
        :returns: ``self`` for chaining.
        """
        self._feature_names = X.columns.tolist()
        y = to_structured_array(np.asarray(durations), np.asarray(events))
        estimator = RandomSurvivalForest(**self.params.as_native())
        estimator.fit(X[self._feature_names].values, y)
        self._estimator = estimator
        return self

    @property
    def estimator(self) -> RandomSurvivalForest:
        """Return the fitted estimator.

        :raises RuntimeError: If the model has not been fitted.
        :returns: The fitted :class:`RandomSurvivalForest`.
        """
        if self._estimator is None:
            raise RuntimeError("RSFSurvivalModel has not been fitted")
        return self._estimator

    def predict_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Return the model's risk score (cumulative hazard) per row.

        :param X: Features compatible with the fit.
        :returns: 1-D array of risk scores; higher means earlier predicted churn.
        """
        if self._feature_names is None:
            raise RuntimeError("RSFSurvivalModel has not been fitted")
        return self.estimator.predict(X[self._feature_names].values)

    def survival_at_times(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """Return S(t) evaluated at ``times`` for each row in ``X``.

        :param X: Features compatible with the fit.
        :param times: 1-D array of times.
        :returns: ``(n_samples, len(times))`` array of survival probabilities.
        """
        if self._feature_names is None:
            raise RuntimeError("RSFSurvivalModel has not been fitted")
        survival_funcs = self.estimator.predict_survival_function(
            X[self._feature_names].values, return_array=False
        )
        out = np.empty((len(survival_funcs), len(times)), dtype=float)
        for i, fn in enumerate(survival_funcs):
            out[i] = fn(times)
        return out

    def hazard_at_horizon(self, X: pd.DataFrame, *, horizon_days: int) -> np.ndarray:
        """Return ``1 - S(horizon)`` per row.

        :param X: Features compatible with the fit.
        :param horizon_days: Horizon at which to evaluate the hazard.
        :returns: 1-D array of hazard probabilities in ``[0, 1]``.
        """
        survival = self.survival_at_times(X, np.array([horizon_days], dtype=float))
        return 1.0 - survival.ravel()
