from __future__ import annotations

import pandas as pd


def yyyymmdd_to_datetime(values: pd.Series) -> pd.Series:
    """Convert KKBox ``YYYYMMDD`` integer dates to ``datetime64[ns]``.

    Invalid or out-of-range integers become :class:`pandas.NaT`.

    :param values: Integer-valued series in ``YYYYMMDD`` form.
    :returns: Datetime series of the same length.
    """
    return pd.to_datetime(values.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")


def days_since(reference: pd.Series, anchor: pd.Timestamp) -> pd.Series:
    """Return the number of days between each timestamp and a fixed anchor.

    Negative values mean ``reference`` lies after ``anchor``.

    :param reference: Datetime series.
    :param anchor: Reference timestamp.
    :returns: Float series of day deltas (NaN where ``reference`` is NaT).
    """
    delta = anchor - pd.to_datetime(reference)
    return delta.dt.days.astype("Float64")


def restrict_before(frame: pd.DataFrame, *, date_col: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Drop rows whose ``date_col`` is on or after ``cutoff``.

    Used to enforce no-future-leakage at feature build time.

    :param frame: Source frame.
    :param date_col: Name of the datetime column in ``frame``.
    :param cutoff: Strict upper bound (rows ``>=`` are dropped).
    :returns: A copy of ``frame`` containing only past rows.
    """
    mask = frame[date_col] < cutoff
    return frame.loc[mask].copy()
