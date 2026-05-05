from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LightGBMUpliftParams:
    """Hyperparameters for the LightGBM base learners used inside meta-learners.

    :param objective: ``"binary"`` for outcomes that are 0/1.
    :param learning_rate: Boosting learning rate.
    :param num_leaves: Maximum number of leaves per tree.
    :param min_data_in_leaf: Minimum samples per leaf.
    :param n_estimators: Boosting rounds.
    :param feature_fraction: Column subsample ratio.
    :param verbose: LightGBM verbosity flag.
    """

    objective: str = "binary"
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 200
    n_estimators: int = 600
    feature_fraction: float = 0.85
    verbose: int = -1


def _make_classifier(params: LightGBMUpliftParams) -> lgb.LGBMClassifier:
    """Build a fresh LightGBM classifier from typed params.

    :param params: Hyperparameters dataclass.
    :returns: A new :class:`lightgbm.LGBMClassifier` ready for ``fit``.
    """
    return lgb.LGBMClassifier(
        objective=params.objective,
        learning_rate=params.learning_rate,
        num_leaves=params.num_leaves,
        min_data_in_leaf=params.min_data_in_leaf,
        n_estimators=params.n_estimators,
        feature_fraction=params.feature_fraction,
        verbose=params.verbose,
    )


class TLearner:
    """T-learner: one outcome model per treatment arm; uplift is their difference."""

    def __init__(self, params: LightGBMUpliftParams | None = None) -> None:
        """Initialise the meta-learner.

        :param params: Optional base-learner hyperparameters.
        """
        self.params = params or LightGBMUpliftParams()
        self._model_treated: lgb.LGBMClassifier | None = None
        self._model_control: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> TLearner:
        """Fit the two outcome models.

        :param X: Feature frame.
        :param treatment: 0/1 treatment indicator.
        :param outcome: 0/1 outcome.
        :returns: ``self`` for chaining.
        :raises ValueError: If either arm has no rows.
        """
        treatment = np.asarray(treatment).astype(int)
        outcome = np.asarray(outcome).astype(int)
        treated_mask = treatment == 1
        control_mask = treatment == 0
        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            raise ValueError("T-learner requires non-empty treated and control arms")

        self._feature_names = X.columns.tolist()
        self._model_treated = _make_classifier(self.params).fit(
            X.loc[treated_mask, self._feature_names], outcome[treated_mask]
        )
        self._model_control = _make_classifier(self.params).fit(
            X.loc[control_mask, self._feature_names], outcome[control_mask]
        )
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-row uplift estimates ``E[Y|T=1] - E[Y|T=0]``.

        :param X: Feature frame.
        :returns: 1-D array of uplift estimates.
        :raises RuntimeError: If the model has not been fitted.
        """
        if (
            self._model_treated is None
            or self._model_control is None
            or self._feature_names is None
        ):
            raise RuntimeError("TLearner has not been fitted")
        x = X[self._feature_names]
        p_treated = self._model_treated.predict_proba(x)[:, 1]
        p_control = self._model_control.predict_proba(x)[:, 1]
        return p_treated - p_control


class XLearner:
    """X-learner: combines outcome models with treatment-imputation residuals.

    The variant implemented here uses a propensity-weighted average of the
    treated-side and control-side imputed treatment effects. The propensity is
    estimated from the data unless provided.
    """

    def __init__(self, params: LightGBMUpliftParams | None = None) -> None:
        """Initialise the meta-learner.

        :param params: Optional base-learner hyperparameters.
        """
        self.params = params or LightGBMUpliftParams()
        self._model_treated: lgb.LGBMClassifier | None = None
        self._model_control: lgb.LGBMClassifier | None = None
        self._tau_treated: lgb.LGBMRegressor | None = None
        self._tau_control: lgb.LGBMRegressor | None = None
        self._propensity: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] | None = None

    def _make_regressor(self) -> lgb.LGBMRegressor:
        """Build a fresh LightGBM regressor mirroring the classifier params.

        :returns: A new :class:`lightgbm.LGBMRegressor` for the tau models.
        """
        return lgb.LGBMRegressor(
            learning_rate=self.params.learning_rate,
            num_leaves=self.params.num_leaves,
            min_data_in_leaf=self.params.min_data_in_leaf,
            n_estimators=self.params.n_estimators,
            feature_fraction=self.params.feature_fraction,
            verbose=self.params.verbose,
        )

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> XLearner:
        """Fit the X-learner stack.

        :param X: Feature frame.
        :param treatment: 0/1 treatment indicator.
        :param outcome: 0/1 outcome.
        :returns: ``self`` for chaining.
        :raises ValueError: If either arm has no rows.
        """
        treatment = np.asarray(treatment).astype(int)
        outcome = np.asarray(outcome).astype(int)
        treated_mask = treatment == 1
        control_mask = treatment == 0
        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            raise ValueError("X-learner requires non-empty treated and control arms")

        self._feature_names = X.columns.tolist()
        self._model_treated = _make_classifier(self.params).fit(
            X.loc[treated_mask, self._feature_names], outcome[treated_mask]
        )
        self._model_control = _make_classifier(self.params).fit(
            X.loc[control_mask, self._feature_names], outcome[control_mask]
        )

        d_treated = (
            outcome[treated_mask]
            - self._model_control.predict_proba(X.loc[treated_mask, self._feature_names])[:, 1]
        )
        d_control = (
            self._model_treated.predict_proba(X.loc[control_mask, self._feature_names])[:, 1]
            - outcome[control_mask]
        )

        self._tau_treated = self._make_regressor().fit(
            X.loc[treated_mask, self._feature_names], d_treated
        )
        self._tau_control = self._make_regressor().fit(
            X.loc[control_mask, self._feature_names], d_control
        )
        self._propensity = _make_classifier(self.params).fit(X[self._feature_names], treatment)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-row uplift via propensity-weighted X-learner formula.

        :param X: Feature frame.
        :returns: 1-D array of uplift estimates.
        :raises RuntimeError: If the model has not been fitted.
        """
        if self._tau_treated is None or self._tau_control is None or self._propensity is None:
            raise RuntimeError("XLearner has not been fitted")
        if self._feature_names is None:
            raise RuntimeError("XLearner has not been fitted")
        x = X[self._feature_names]
        p = np.clip(self._propensity.predict_proba(x)[:, 1], 0.05, 0.95)
        tau_t = self._tau_treated.predict(x)
        tau_c = self._tau_control.predict(x)
        return p * tau_c + (1.0 - p) * tau_t


class DRLearner:
    """Doubly-robust learner: treatment-effect regression on AIPW pseudo-outcomes."""

    def __init__(self, params: LightGBMUpliftParams | None = None) -> None:
        """Initialise the meta-learner.

        :param params: Optional base-learner hyperparameters.
        """
        self.params = params or LightGBMUpliftParams()
        self._model_treated: lgb.LGBMClassifier | None = None
        self._model_control: lgb.LGBMClassifier | None = None
        self._propensity: lgb.LGBMClassifier | None = None
        self._tau: lgb.LGBMRegressor | None = None
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> DRLearner:
        """Fit the DR-learner stack.

        :param X: Feature frame.
        :param treatment: 0/1 treatment indicator.
        :param outcome: 0/1 outcome.
        :returns: ``self`` for chaining.
        :raises ValueError: If either arm has no rows.
        """
        treatment = np.asarray(treatment).astype(int).astype(float)
        outcome = np.asarray(outcome).astype(float)
        treated_mask = treatment == 1
        control_mask = treatment == 0
        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            raise ValueError("DR-learner requires non-empty treated and control arms")

        self._feature_names = X.columns.tolist()
        self._model_treated = _make_classifier(self.params).fit(
            X.loc[treated_mask, self._feature_names], outcome[treated_mask].astype(int)
        )
        self._model_control = _make_classifier(self.params).fit(
            X.loc[control_mask, self._feature_names], outcome[control_mask].astype(int)
        )
        self._propensity = _make_classifier(self.params).fit(
            X[self._feature_names], treatment.astype(int)
        )

        x = X[self._feature_names]
        mu1 = self._model_treated.predict_proba(x)[:, 1]
        mu0 = self._model_control.predict_proba(x)[:, 1]
        e = np.clip(self._propensity.predict_proba(x)[:, 1], 0.05, 0.95)
        pseudo = (
            mu1
            - mu0
            + treatment * (outcome - mu1) / e
            - (1 - treatment) * (outcome - mu0) / (1 - e)
        )

        self._tau = lgb.LGBMRegressor(
            learning_rate=self.params.learning_rate,
            num_leaves=self.params.num_leaves,
            min_data_in_leaf=self.params.min_data_in_leaf,
            n_estimators=self.params.n_estimators,
            feature_fraction=self.params.feature_fraction,
            verbose=self.params.verbose,
        )
        self._tau.fit(x, pseudo)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Return per-row uplift estimates.

        :param X: Feature frame.
        :returns: 1-D array of uplift estimates.
        :raises RuntimeError: If the model has not been fitted.
        """
        if self._tau is None or self._feature_names is None:
            raise RuntimeError("DRLearner has not been fitted")
        return self._tau.predict(X[self._feature_names])
