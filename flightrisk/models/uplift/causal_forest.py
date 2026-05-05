from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


@dataclass(frozen=True)
class CausalForestParams:
    """Hyperparameters for the econml :class:`CausalForestDML` wrapper.

    :param n_estimators: Number of forest trees.
    :param min_samples_leaf: Minimum samples per leaf.
    :param max_depth: Maximum tree depth (``None`` means unbounded).
    :param random_state: Seed for tree bagging.
    :param discrete_treatment: Whether the treatment is binary/categorical.
    """

    n_estimators: int = 200
    min_samples_leaf: int = 30
    max_depth: int | None = None
    random_state: int = 1337
    discrete_treatment: bool = True


class CausalForestUpliftModel:
    """Wrap :class:`econml.dml.CausalForestDML` with a uniform uplift API."""

    def __init__(self, params: CausalForestParams | None = None) -> None:
        """Initialise the wrapper.

        :param params: Optional hyperparameters.
        """
        self.params = params or CausalForestParams()
        self._estimator: CausalForestDML | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray
    ) -> CausalForestUpliftModel:
        """Fit the causal forest with default outcome and treatment nuisance models.

        :param X: Feature frame.
        :param treatment: 0/1 treatment indicator.
        :param outcome: 0/1 outcome.
        :returns: ``self`` for chaining.
        """
        self._feature_names = X.columns.tolist()
        estimator = CausalForestDML(
            model_y=GradientBoostingRegressor(random_state=self.params.random_state),
            model_t=GradientBoostingClassifier(random_state=self.params.random_state),
            n_estimators=self.params.n_estimators,
            min_samples_leaf=self.params.min_samples_leaf,
            max_depth=self.params.max_depth,
            random_state=self.params.random_state,
            discrete_treatment=self.params.discrete_treatment,
        )
        estimator.fit(
            Y=np.asarray(outcome).astype(float),
            T=np.asarray(treatment).astype(int),
            X=X[self._feature_names].values,
        )
        self._estimator = estimator
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-row CATE estimates.

        :param X: Feature frame.
        :returns: 1-D array of treatment-effect estimates.
        :raises RuntimeError: If the model has not been fitted.
        """
        if self._estimator is None or self._feature_names is None:
            raise RuntimeError("CausalForestUpliftModel has not been fitted")
        effect = self._estimator.effect(X[self._feature_names].values)
        return np.asarray(effect).ravel()
