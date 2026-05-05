from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from flightrisk.models.survival.labels import to_structured_array


@dataclass(frozen=True)
class SurvivalMetrics:
    """Metrics emitted by the survival track.

    :param c_index: Harrell's concordance on censored data.
    :param time_dependent_auc_mean: Mean of time-dependent AUC across horizons.
    :param integrated_brier: Integrated Brier score across horizons.
    """

    c_index: float
    time_dependent_auc_mean: float
    integrated_brier: float

    def as_dict(self) -> Mapping[str, float]:
        """Return the metrics as a plain ``dict`` for MLflow logging.

        :returns: Mapping from metric name to value.
        """
        return {
            "c_index": self.c_index,
            "time_dependent_auc_mean": self.time_dependent_auc_mean,
            "integrated_brier": self.integrated_brier,
        }


def survival_metrics(
    *,
    train_durations: np.ndarray,
    train_events: np.ndarray,
    test_durations: np.ndarray,
    test_events: np.ndarray,
    risk_scores: np.ndarray,
    survival_at_horizons: np.ndarray,
    horizons: np.ndarray,
) -> SurvivalMetrics:
    """Compute the survival metric suite.

    :param train_durations: Train follow-up times.
    :param train_events: Train 0/1 event indicators.
    :param test_durations: Test follow-up times.
    :param test_events: Test 0/1 event indicators.
    :param risk_scores: Per-row risk score on test (higher = earlier event).
    :param survival_at_horizons: ``(n_samples, len(horizons))`` matrix of S(t).
    :param horizons: 1-D array of horizons matching ``survival_at_horizons``.
    :returns: A :class:`SurvivalMetrics` bundle.
    """
    train_struct = to_structured_array(train_durations, train_events)
    test_struct = to_structured_array(test_durations, test_events)

    c_index = float(
        concordance_index_censored(test_struct["event"], test_struct["time"], risk_scores)[0]
    )
    auc_per_horizon, _ = cumulative_dynamic_auc(
        train_struct, test_struct, risk_scores, times=horizons
    )
    brier = float(
        integrated_brier_score(train_struct, test_struct, survival_at_horizons, times=horizons)
    )
    return SurvivalMetrics(
        c_index=c_index,
        time_dependent_auc_mean=float(np.mean(auc_per_horizon)),
        integrated_brier=brier,
    )
