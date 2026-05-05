from __future__ import annotations

import pandas as pd

from flightrisk.models.survival.labels import build_survival_labels


def _transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "msno": ["a", "a", "b", "b", "c"],
            "payment_method_id": [41, 41, 36, 36, 41],
            "payment_plan_days": [30, 30, 30, 30, 30],
            "plan_list_price": [149, 149, 149, 149, 149],
            "actual_amount_paid": [149, 0, 149, 149, 149],
            "is_auto_renew": [1, 0, 1, 1, 1],
            "transaction_date": [20170201, 20170310, 20170215, 20170401, 20170220],
            "membership_expire_date": [20170301, 20170410, 20170315, 20170501, 20170320],
            "is_cancel": [0, 1, 0, 1, 0],
        }
    )


def test_build_survival_labels_observed_event() -> None:
    cutoff = pd.Timestamp("2017-02-28")
    labels = build_survival_labels(_transactions(), cutoff=cutoff, horizon_days=90).to_frame()
    a = labels[labels["msno"] == "a"].iloc[0]
    assert a["event_observed"] == 1
    assert a["duration_days"] > 0


def test_build_survival_labels_censored_at_horizon() -> None:
    cutoff = pd.Timestamp("2017-02-28")
    labels = build_survival_labels(_transactions(), cutoff=cutoff, horizon_days=5).to_frame()
    a = labels[labels["msno"] == "a"].iloc[0]
    assert a["event_observed"] == 0
    assert a["duration_days"] == 5


def test_build_survival_labels_no_post_cutoff_activity_is_censored() -> None:
    cutoff = pd.Timestamp("2017-02-28")
    labels = build_survival_labels(_transactions(), cutoff=cutoff, horizon_days=90).to_frame()
    c = labels[labels["msno"] == "c"].iloc[0]
    assert c["event_observed"] == 0
    assert c["duration_days"] == 90
