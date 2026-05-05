from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flightrisk.features.time_helpers import yyyymmdd_to_datetime


@dataclass(frozen=True)
class SurvivalLabels:
    """Right-censored survival labels keyed by ``msno``.

    :param msno: Subscriber ID.
    :param duration_days: Time from cutoff until event or censoring, in days.
    :param event_observed: 1 if churn was observed before ``horizon_days``;
        0 if the user was still active at horizon (right-censored).
    """

    msno: pd.Series
    duration_days: pd.Series
    event_observed: pd.Series

    def to_frame(self) -> pd.DataFrame:
        """Return the labels as a tidy frame.

        :returns: Frame with ``msno``, ``duration_days``, ``event_observed``.
        """
        return pd.DataFrame(
            {
                "msno": self.msno.reset_index(drop=True),
                "duration_days": self.duration_days.reset_index(drop=True).astype("float64"),
                "event_observed": self.event_observed.reset_index(drop=True).astype("int64"),
            }
        )


def build_survival_labels(
    transactions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon_days: int = 90,
) -> SurvivalLabels:
    """Convert KKBox transactions into right-censored survival labels.

    For each user the event time is the first ``is_cancel == 1`` transaction
    after ``cutoff``; if none occurs within ``horizon_days``, the user is
    censored at the horizon. Users with no post-cutoff transactions are also
    censored at the horizon.

    :param transactions: Validated transactions frame.
    :param cutoff: Build cutoff. The clock starts here.
    :param horizon_days: Maximum follow-up window in days.
    :returns: A :class:`SurvivalLabels` bundle.
    :raises ValueError: If ``horizon_days`` is non-positive.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    df = transactions.copy()
    df["txn_date"] = yyyymmdd_to_datetime(df["transaction_date"])
    horizon_end = cutoff + pd.Timedelta(days=horizon_days)
    after = df[(df["txn_date"] > cutoff) & (df["txn_date"] <= horizon_end)].copy()
    cancels = after[after["is_cancel"] == 1].copy()
    cancels["days"] = (cancels["txn_date"] - cutoff).dt.days
    first_cancel = cancels.sort_values(["msno", "days"]).groupby("msno", as_index=False).first()

    universe = pd.DataFrame({"msno": df["msno"].drop_duplicates().tolist()})
    merged = universe.merge(
        first_cancel[["msno", "days"]].rename(columns={"days": "event_day"}),
        on="msno",
        how="left",
    )
    merged["event_observed"] = merged["event_day"].notna().astype(int)
    merged["duration_days"] = merged["event_day"].fillna(horizon_days).astype(float)
    return SurvivalLabels(
        msno=merged["msno"],
        duration_days=merged["duration_days"],
        event_observed=merged["event_observed"],
    )


def to_structured_array(durations: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Pack durations and event indicators into the structured array sksurv expects.

    :param durations: Float durations.
    :param events: Integer 0/1 event indicators.
    :returns: Structured numpy array with fields ``event`` (bool) and ``time``.
    """
    return np.array(
        list(zip(events.astype(bool), durations.astype(float))),
        dtype=[("event", "?"), ("time", "<f8")],
    )
