from __future__ import annotations

import pandas as pd
import pytest

from flightrisk.data.loaders import KKBoxArtifacts
from flightrisk.features.kkbox import (
    KKBoxFeatureConfig,
    build_kkbox_feature_matrix,
    listening_features,
    transaction_features,
)
from flightrisk.features.pipeline import build_kkbox_bundle


@pytest.fixture()
def kkbox_artifacts() -> KKBoxArtifacts:
    members = pd.DataFrame(
        {
            "msno": ["a", "b"],
            "city": [1, 2],
            "bd": [25, 75],
            "gender": ["male", None],
            "registered_via": [3, 7],
            "registration_init_time": [20160101, 20161201],
        }
    )
    transactions = pd.DataFrame(
        {
            "msno": ["a", "a", "b", "b"],
            "payment_method_id": [41, 41, 36, 36],
            "payment_plan_days": [30, 30, 30, 30],
            "plan_list_price": [149, 149, 149, 149],
            "actual_amount_paid": [149, 0, 149, 149],
            "is_auto_renew": [1, 0, 1, 1],
            "transaction_date": [20170101, 20170201, 20170105, 20170220],
            "membership_expire_date": [20170131, 20170228, 20170204, 20170322],
            "is_cancel": [0, 1, 0, 0],
        }
    )
    user_logs = pd.DataFrame(
        {
            "msno": ["a", "a", "a", "b", "b"],
            "date": [20170105, 20170120, 20170215, 20170210, 20170225],
            "num_25": [5, 3, 0, 1, 0],
            "num_50": [3, 2, 0, 0, 0],
            "num_75": [2, 1, 0, 0, 0],
            "num_985": [1, 0, 0, 0, 0],
            "num_100": [40, 30, 0, 5, 2],
            "num_unq": [30, 25, 0, 5, 2],
            "total_secs": [12000.0, 10000.0, 0.0, 800.0, 300.0],
        }
    )
    labels = pd.DataFrame({"msno": ["a", "b"], "is_churn": [0, 1]})
    return KKBoxArtifacts(
        members=members, transactions=transactions, user_logs=user_logs, labels=labels
    )


def test_transaction_features_no_future_leakage(kkbox_artifacts: KKBoxArtifacts) -> None:
    cutoff = pd.Timestamp("2017-02-15")
    out = transaction_features(kkbox_artifacts.transactions, cutoff=cutoff)
    a_row = out[out["msno"] == "a"].iloc[0]
    assert a_row["n_transactions"] == 2
    b_row = out[out["msno"] == "b"].iloc[0]
    assert b_row["n_transactions"] == 1
    assert (out["days_until_expire"].dropna() >= -180).all()


def test_listening_features_window_widens(kkbox_artifacts: KKBoxArtifacts) -> None:
    cutoff = pd.Timestamp("2017-02-28")
    out = listening_features(
        kkbox_artifacts.user_logs,
        cutoff=cutoff,
        config=KKBoxFeatureConfig(windows_days=(7, 30, 90)),
    )
    a = out[out["msno"] == "a"].iloc[0]
    assert a["plays_90d"] >= a["plays_30d"] >= a["plays_7d"]
    assert "days_since_last_login" in out.columns


def test_build_kkbox_feature_matrix_shape(kkbox_artifacts: KKBoxArtifacts) -> None:
    cutoff = pd.Timestamp("2017-02-28")
    matrix = build_kkbox_feature_matrix(
        kkbox_artifacts.members,
        kkbox_artifacts.transactions,
        kkbox_artifacts.user_logs,
        cutoff=cutoff,
    )
    assert set(matrix["msno"]) == {"a", "b"}
    assert "tenure_days" in matrix.columns
    assert "auto_renew_share" in matrix.columns
    assert any(col.startswith("plays_") for col in matrix.columns)


def test_build_kkbox_bundle_aligns_labels(kkbox_artifacts: KKBoxArtifacts) -> None:
    bundle = build_kkbox_bundle(kkbox_artifacts, cutoff="2017-02-28")
    assert (bundle.features["msno"].values == bundle.labels["msno"].values).all()
    assert bundle.cutoff == pd.Timestamp("2017-02-28")


def test_no_future_dates_in_kkbox_features(kkbox_artifacts: KKBoxArtifacts) -> None:
    cutoff = pd.Timestamp("2017-01-15")
    out = build_kkbox_feature_matrix(
        kkbox_artifacts.members,
        kkbox_artifacts.transactions,
        kkbox_artifacts.user_logs,
        cutoff=cutoff,
    )
    a = out[out["msno"] == "a"].iloc[0]
    assert a["n_transactions"] == 1
    assert a["days_since_last_login"] >= 0
