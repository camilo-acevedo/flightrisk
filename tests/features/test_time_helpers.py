from __future__ import annotations

import pandas as pd

from flightrisk.features.time_helpers import (
    days_since,
    restrict_before,
    yyyymmdd_to_datetime,
)


def test_yyyymmdd_to_datetime_parses_valid() -> None:
    series = pd.Series([20170101, 20171231, 20180229], dtype="Int64")
    out = yyyymmdd_to_datetime(series)
    assert out.iloc[0] == pd.Timestamp("2017-01-01")
    assert out.iloc[1] == pd.Timestamp("2017-12-31")
    assert pd.isna(out.iloc[2])


def test_days_since_handles_nat() -> None:
    series = pd.to_datetime(["2017-01-01", "2017-01-31", pd.NaT])
    deltas = days_since(pd.Series(series), pd.Timestamp("2017-02-01"))
    assert deltas.iloc[0] == 31
    assert deltas.iloc[1] == 1
    assert pd.isna(deltas.iloc[2])


def test_restrict_before_drops_future_rows() -> None:
    df = pd.DataFrame(
        {
            "log_date": pd.to_datetime(["2017-01-31", "2017-02-15", "2017-03-01"]),
            "value": [1, 2, 3],
        }
    )
    out = restrict_before(df, date_col="log_date", cutoff=pd.Timestamp("2017-02-28"))
    assert out["value"].tolist() == [1, 2]
