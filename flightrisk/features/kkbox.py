from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from flightrisk.features.time_helpers import (
    days_since,
    restrict_before,
    yyyymmdd_to_datetime,
)

_LISTEN_PROPORTION_COLS = ("num_25", "num_50", "num_75", "num_985", "num_100")


@dataclass(frozen=True)
class KKBoxFeatureConfig:
    """Configuration for KKBox feature engineering.

    :param windows_days: Rolling windows used to aggregate listening behaviour.
    :param include_rolling_diversity: If true, add unique-artist diversity proxies.
    :param include_session_stats: If true, include session-length statistics.
    :param include_payment_dynamics: If true, include charge deltas and refund counts.
    """

    windows_days: tuple[int, ...] = (7, 30, 90)
    include_rolling_diversity: bool = True
    include_session_stats: bool = True
    include_payment_dynamics: bool = True


def member_features(members: pd.DataFrame, *, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Derive tenure-style features from the KKBox member master table.

    :param members: Validated member frame.
    :param cutoff: Build cutoff used to compute tenure.
    :returns: One row per ``msno`` with tenure and registration features.
    """
    out = pd.DataFrame({"msno": members["msno"].values})
    reg_dt = yyyymmdd_to_datetime(members["registration_init_time"])
    out["tenure_days"] = days_since(reg_dt, cutoff).astype("float64")
    out["registration_year"] = reg_dt.dt.year.astype("Int64")
    out["age_capped"] = members["bd"].clip(lower=10, upper=80)
    out["registered_via"] = members["registered_via"].astype("Int64")
    out["city"] = members["city"].astype("Int64")
    out["gender"] = members["gender"].fillna("unknown")
    return out


def transaction_features(
    transactions: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    include_payment_dynamics: bool = True,
) -> pd.DataFrame:
    """Aggregate transaction history up to ``cutoff`` per user.

    :param transactions: Validated transactions frame.
    :param cutoff: Build cutoff; later transactions are dropped to avoid leakage.
    :param include_payment_dynamics: Include charge deltas and refund counts.
    :returns: One row per ``msno`` with aggregated payment features.
    """
    df = transactions.copy()
    df["txn_date"] = yyyymmdd_to_datetime(df["transaction_date"])
    df["expire_date"] = yyyymmdd_to_datetime(df["membership_expire_date"])
    df = restrict_before(df, date_col="txn_date", cutoff=cutoff)
    df["charge_delta"] = df["plan_list_price"] - df["actual_amount_paid"]

    grouped = df.groupby("msno", sort=False)
    out = pd.DataFrame(
        {
            "n_transactions": grouped.size().astype("Int64"),
            "last_plan_list_price": grouped["plan_list_price"].last().astype("float64"),
            "last_actual_amount_paid": grouped["actual_amount_paid"].last().astype("float64"),
            "auto_renew_share": grouped["is_auto_renew"].mean().astype("float64"),
            "cancel_share": grouped["is_cancel"].mean().astype("float64"),
            "n_payment_methods": grouped["payment_method_id"].nunique().astype("Int64"),
            "days_since_last_txn": days_since(grouped["txn_date"].max(), cutoff).astype("float64"),
            "days_until_expire": -days_since(grouped["expire_date"].max(), cutoff).astype(
                "float64"
            ),
        }
    )
    if include_payment_dynamics:
        out["charge_delta_mean"] = grouped["charge_delta"].mean().astype("float64")
        out["charge_delta_max"] = grouped["charge_delta"].max().astype("float64")
        out["refund_count"] = (df["charge_delta"] > 0).groupby(df["msno"]).sum().astype("Int64")
    out = out.reset_index().rename(columns={"index": "msno"})
    return out


def _window_aggregate(
    logs: pd.DataFrame, *, cutoff: pd.Timestamp, window_days: int, include_session_stats: bool
) -> pd.DataFrame:
    """Aggregate listening logs over ``[cutoff - window_days, cutoff)``.

    :param logs: Logs frame restricted to past dates.
    :param cutoff: Build cutoff.
    :param window_days: Look-back window length in days.
    :param include_session_stats: Whether to compute session-second statistics.
    :returns: One row per ``msno`` with windowed counts and listen ratios.
    """
    start = cutoff - pd.Timedelta(days=window_days)
    chunk = logs[(logs["log_date"] >= start) & (logs["log_date"] < cutoff)]
    grouped = chunk.groupby("msno", sort=False)

    out = pd.DataFrame(
        {
            f"plays_{window_days}d": grouped["num_100"].sum().astype("float64"),
            f"unique_songs_{window_days}d": grouped["num_unq"].sum().astype("float64"),
            f"active_days_{window_days}d": grouped["log_date"].nunique().astype("Int64"),
        }
    )
    chunk_with_total = chunk.assign(_total=chunk[list(_LISTEN_PROPORTION_COLS)].sum(axis=1))
    safe_total = chunk_with_total.groupby("msno")["_total"].sum()
    full_plays = grouped["num_100"].sum()
    out[f"completion_ratio_{window_days}d"] = (full_plays / safe_total.replace(0, np.nan)).astype(
        "float64"
    )

    if include_session_stats:
        out[f"total_secs_{window_days}d"] = grouped["total_secs"].sum().astype("float64")
        out[f"avg_secs_per_active_day_{window_days}d"] = (
            out[f"total_secs_{window_days}d"]
            / out[f"active_days_{window_days}d"].replace(0, np.nan)
        ).astype("float64")
    return out.reset_index().rename(columns={"index": "msno"})


def listening_features(
    user_logs: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    config: KKBoxFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build rolling listening features from KKBox daily logs.

    :param user_logs: Validated daily log frame.
    :param cutoff: Build cutoff; later logs are dropped to avoid leakage.
    :param config: Optional feature configuration.
    :returns: One row per ``msno`` with windowed listening features and a
        ``days_since_last_login`` column.
    """
    cfg = config or KKBoxFeatureConfig()
    df = user_logs.copy()
    df["log_date"] = yyyymmdd_to_datetime(df["date"])
    df = restrict_before(df, date_col="log_date", cutoff=cutoff)

    base = pd.DataFrame({"msno": df["msno"].drop_duplicates().tolist()})
    out = base
    for window in cfg.windows_days:
        piece = _window_aggregate(
            df,
            cutoff=cutoff,
            window_days=window,
            include_session_stats=cfg.include_session_stats,
        )
        out = out.merge(piece, on="msno", how="left")

    count_cols = [
        c
        for c in out.columns
        if c.startswith(("plays_", "unique_songs_", "active_days_", "total_secs_"))
    ]
    out[count_cols] = out[count_cols].fillna(0)

    last_login = df.groupby("msno", sort=False)["log_date"].max()
    last_login_frame = (
        days_since(last_login, cutoff)
        .astype("float64")
        .rename("days_since_last_login")
        .reset_index()
    )
    return out.merge(last_login_frame, on="msno", how="left")


def derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interaction, ratio, and log-transformed features to the matrix.

    These are cheap, deterministic combinations of base features that tend to
    move tree-based models on subscription churn data:

    * ``log1p`` of long-tailed counters (tenure, plays, songs)
    * recent-vs-lifetime engagement ratios
    * spend efficiency (``actual / list``) and a ``charge_drop`` flag
    * recency band indicators (``never_logged_in``, ``inactive_30d``)
    * tenure × auto-renew interaction (loyalty proxy)

    :param frame: A frame produced by :func:`build_kkbox_feature_matrix`.
    :returns: A copy of ``frame`` with extra columns appended.
    """
    out = frame.copy()

    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        """Divide with a 1-floored denominator and return a numeric series."""
        return (num.astype("float64") / den.astype("float64").clip(lower=1.0)).astype("float64")

    if "tenure_days" in out.columns:
        out["tenure_days_log"] = np.log1p(out["tenure_days"].clip(lower=0)).astype("float64")
    for window in (7, 30, 90):
        plays_col = f"plays_{window}d"
        active_col = f"active_days_{window}d"
        unique_col = f"unique_songs_{window}d"
        if plays_col in out.columns:
            out[f"{plays_col}_log"] = np.log1p(out[plays_col].clip(lower=0)).astype("float64")
            if active_col in out.columns:
                out[f"plays_per_active_day_{window}d"] = _safe_div(out[plays_col], out[active_col])
            if unique_col in out.columns:
                out[f"diversity_ratio_{window}d"] = _safe_div(out[unique_col], out[plays_col])

    if "plays_7d" in out.columns and "plays_30d" in out.columns:
        out["recent_engagement_ratio_7_30"] = _safe_div(out["plays_7d"] * 30 / 7, out["plays_30d"])
    if "plays_30d" in out.columns and "plays_90d" in out.columns:
        out["recent_engagement_ratio_30_90"] = _safe_div(
            out["plays_30d"] * 90 / 30, out["plays_90d"]
        )

    if "last_actual_amount_paid" in out.columns and "last_plan_list_price" in out.columns:
        out["spend_efficiency"] = _safe_div(
            out["last_actual_amount_paid"], out["last_plan_list_price"]
        )
        out["charge_drop_flag"] = (out["spend_efficiency"] < 0.5).astype("int64")

    if "auto_renew_share" in out.columns and "tenure_days" in out.columns:
        out["loyalty_proxy"] = (
            out["auto_renew_share"].fillna(0).astype("float64")
            * np.log1p(out["tenure_days"].clip(lower=0))
        ).astype("float64")
    if "cancel_share" in out.columns and "auto_renew_share" in out.columns:
        out["churn_intent_proxy"] = (
            out["cancel_share"].fillna(0).astype("float64")
            * (1.0 - out["auto_renew_share"].fillna(0).astype("float64"))
        ).astype("float64")

    if "days_since_last_login" in out.columns:
        col = out["days_since_last_login"].astype("float64")
        out["never_logged_in"] = col.isna().astype("int64")
        out["inactive_7d"] = (col.fillna(9999) >= 7).astype("int64")
        out["inactive_30d"] = (col.fillna(9999) >= 30).astype("int64")

    return out


def build_kkbox_feature_matrix(
    members: pd.DataFrame,
    transactions: pd.DataFrame,
    user_logs: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    config: KKBoxFeatureConfig | None = None,
) -> pd.DataFrame:
    """Combine member, transaction, and listening features into one frame.

    :param members: Validated member master table.
    :param transactions: Validated transactions frame.
    :param user_logs: Validated daily log frame.
    :param cutoff: Build cutoff used to enforce no-leakage joins.
    :param config: Optional feature configuration.
    :returns: One row per ``msno`` ready to merge with labels or hazard times.
    """
    cfg = config or KKBoxFeatureConfig()
    base = member_features(members, cutoff=cutoff)
    txn = transaction_features(
        transactions,
        cutoff=cutoff,
        include_payment_dynamics=cfg.include_payment_dynamics,
    )
    logs = listening_features(user_logs, cutoff=cutoff, config=cfg)
    raw = base.merge(txn, on="msno", how="left").merge(logs, on="msno", how="left")
    return derived_features(raw)
