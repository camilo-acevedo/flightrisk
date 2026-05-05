from __future__ import annotations

import pandas as pd
import pandera as pa
import pytest

from flightrisk.data import schemas


def _valid_members() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "msno": ["a", "b"],
            "city": [1, 2],
            "bd": [25, 30],
            "gender": ["male", "female"],
            "registered_via": [3, 7],
            "registration_init_time": [20170101, 20170201],
        }
    )


def _valid_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "msno": ["a"],
            "payment_method_id": [41],
            "payment_plan_days": [30],
            "plan_list_price": [149],
            "actual_amount_paid": [149],
            "is_auto_renew": [1],
            "transaction_date": [20170101],
            "membership_expire_date": [20170131],
            "is_cancel": [0],
        }
    )


def test_members_schema_accepts_valid() -> None:
    schemas.KKBOX_MEMBERS_SCHEMA.validate(_valid_members())


def test_members_schema_rejects_negative_age() -> None:
    bad = _valid_members().copy()
    bad.loc[0, "bd"] = -50
    with pytest.raises(pa.errors.SchemaError):
        schemas.KKBOX_MEMBERS_SCHEMA.validate(bad)


def test_transactions_schema_rejects_invalid_flag() -> None:
    bad = _valid_transactions().copy()
    bad.loc[0, "is_cancel"] = 2
    with pytest.raises(pa.errors.SchemaError):
        schemas.KKBOX_TRANSACTIONS_SCHEMA.validate(bad)


def test_orange_schema_requires_binary_treatment() -> None:
    bad = pd.DataFrame({"treatment": [0, 1, 2], "outcome": [0, 0, 1]})
    with pytest.raises(pa.errors.SchemaError):
        schemas.ORANGE_UPLIFT_SCHEMA.validate(bad)


def test_labels_schema_unique_msno() -> None:
    bad = pd.DataFrame({"msno": ["a", "a"], "is_churn": [0, 1]})
    with pytest.raises(pa.errors.SchemaError):
        schemas.KKBOX_LABELS_SCHEMA.validate(bad)
