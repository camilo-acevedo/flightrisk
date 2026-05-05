from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


@dataclass(frozen=True)
class CoxParams:
    """Hyperparameters for the Cox proportional-hazards fitter.

    :param penalizer: Elastic-net penalty strength.
    :param l1_ratio: Mix between L1 and L2 (0 = ridge, 1 = lasso).
    """

    penalizer: float = 0.01
    l1_ratio: float = 0.0


class CoxSurvivalModel:
    """Lifelines Cox PH wrapper that returns hazards on a fixed schema."""

    def __init__(self, params: CoxParams | None = None) -> None:
        """Initialise the wrapper.

        :param params: Optional hyperparameters; defaults are model-ready.
        """
        self.params = params or CoxParams()
        self._fitter: CoxPHFitter | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        durations: np.ndarray,
        events: np.ndarray,
    ) -> CoxSurvivalModel:
        """Fit the Cox model.

        :param X: Numeric feature frame; non-numeric columns must be encoded
            upstream.
        :param durations: Durations or follow-up times.
        :param events: 0/1 event indicators.
        :returns: ``self`` for chaining.
        """
        self._feature_names = X.columns.tolist()
        frame = X.copy()
        frame["__duration__"] = np.asarray(durations).astype(float)
        frame["__event__"] = np.asarray(events).astype(int)
        fitter = CoxPHFitter(penalizer=self.params.penalizer, l1_ratio=self.params.l1_ratio)
        fitter.fit(
            frame,
            duration_col="__duration__",
            event_col="__event__",
            show_progress=False,
        )
        self._fitter = fitter
        return self

    @property
    def fitter(self) -> CoxPHFitter:
        """Return the fitted lifelines fitter.

        :returns: The :class:`lifelines.CoxPHFitter` instance.
        :raises RuntimeError: If the model has not been fitted.
        """
        if self._fitter is None:
            raise RuntimeError("CoxSurvivalModel has not been fitted")
        return self._fitter

    def survival_function(
        self, X: pd.DataFrame, *, times: np.ndarray | None = None
    ) -> pd.DataFrame:
        """Return survival probabilities S(t) for each row in ``X``.

        :param X: Feature frame with the columns seen at fit time.
        :param times: Optional times at which to evaluate S(t).
        :returns: Frame indexed by time with one column per row in ``X``.
        :raises RuntimeError: If the model has not been fitted.
        """
        if self._feature_names is None:
            raise RuntimeError("CoxSurvivalModel has not been fitted")
        if times is None:
            return self.fitter.predict_survival_function(X[self._feature_names])
        return self.fitter.predict_survival_function(X[self._feature_names], times=times)

    def hazard_at_horizon(self, X: pd.DataFrame, *, horizon_days: int) -> np.ndarray:
        """Return ``1 - S(horizon)`` per row, the cumulative hazard at horizon.

        :param X: Feature frame compatible with the fit.
        :param horizon_days: Horizon at which to evaluate the hazard.
        :returns: Array of hazard probabilities in ``[0, 1]``.
        """
        survival = self.survival_function(X, times=np.array([horizon_days], dtype=float))
        return 1.0 - survival.values.ravel()
