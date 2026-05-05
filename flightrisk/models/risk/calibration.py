from __future__ import annotations

from typing import Literal, Protocol

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibrationMethod = Literal["isotonic", "platt"]


class _ProbaModel(Protocol):
    """Protocol satisfied by the wrappers under :mod:`flightrisk.models.risk`."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities."""
        ...


class CalibratedRiskModel:
    """Wrap a probabilistic model with isotonic or Platt calibration.

    The wrapped model's ``predict_proba`` is fit on a held-out validation slice
    so the calibrator does not see training rows.
    """

    def __init__(self, base: _ProbaModel, *, method: CalibrationMethod = "isotonic") -> None:
        """Initialise the calibrator.

        :param base: An already-fitted probabilistic model.
        :param method: Either ``"isotonic"`` or ``"platt"``.
        :raises ValueError: If ``method`` is unknown.
        """
        if method not in ("isotonic", "platt"):
            raise ValueError(f"unknown calibration method: {method!r}")
        self.base = base
        self.method: CalibrationMethod = method
        self._iso: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, X_val: pd.DataFrame, y_val: np.ndarray) -> CalibratedRiskModel:
        """Fit the calibrator on a held-out validation slice.

        :param X_val: Validation features.
        :param y_val: Validation labels (0/1).
        :returns: ``self`` for chaining.
        """
        raw = self.base.predict_proba(X_val)
        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(raw, y_val)
        else:
            self._platt = LogisticRegression(solver="lbfgs")
            self._platt.fit(raw.reshape(-1, 1), y_val)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities for ``X``.

        :param X: Features compatible with ``self.base``.
        :returns: 1-D array of calibrated probabilities.
        :raises RuntimeError: If the calibrator has not been fitted.
        """
        raw = self.base.predict_proba(X)
        if self.method == "isotonic":
            if self._iso is None:
                raise RuntimeError("calibrator has not been fitted")
            return np.clip(self._iso.predict(raw), 0.0, 1.0)
        if self._platt is None:
            raise RuntimeError("calibrator has not been fitted")
        return self._platt.predict_proba(raw.reshape(-1, 1))[:, 1]
