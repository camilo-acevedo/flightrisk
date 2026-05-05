from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema, Index

KKBOX_MEMBERS_SCHEMA = DataFrameSchema(
    columns={
        "msno": Column(str, nullable=False, unique=False),
        "city": Column(pa.Int64, nullable=True),
        "bd": Column(pa.Int64, nullable=True, checks=Check.in_range(-1, 120)),
        "gender": Column(str, nullable=True),
        "registered_via": Column(pa.Int64, nullable=True),
        "registration_init_time": Column(pa.Int64, nullable=False),
    },
    coerce=True,
    strict="filter",
)


KKBOX_TRANSACTIONS_SCHEMA = DataFrameSchema(
    columns={
        "msno": Column(str, nullable=False),
        "payment_method_id": Column(pa.Int64, nullable=False),
        "payment_plan_days": Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        "plan_list_price": Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        "actual_amount_paid": Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        "is_auto_renew": Column(pa.Int64, nullable=False, checks=Check.isin([0, 1])),
        "transaction_date": Column(pa.Int64, nullable=False),
        "membership_expire_date": Column(pa.Int64, nullable=False),
        "is_cancel": Column(pa.Int64, nullable=False, checks=Check.isin([0, 1])),
    },
    coerce=True,
    strict="filter",
)


KKBOX_USER_LOGS_SCHEMA = DataFrameSchema(
    columns={
        "msno": Column(str, nullable=False),
        "date": Column(pa.Int64, nullable=False),
        "num_25": Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        "num_50": Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        "num_75": Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        "num_985": Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        "num_100": Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        "num_unq": Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        "total_secs": Column(pa.Float64, nullable=False, checks=Check.ge(-1.0)),
    },
    coerce=True,
    strict="filter",
)


KKBOX_LABELS_SCHEMA = DataFrameSchema(
    columns={
        "msno": Column(str, nullable=False, unique=True),
        "is_churn": Column(pa.Int64, nullable=False, checks=Check.isin([0, 1])),
    },
    coerce=True,
    strict="filter",
)


ORANGE_UPLIFT_SCHEMA = DataFrameSchema(
    columns={
        "treatment": Column(pa.Int64, nullable=False, checks=Check.isin([0, 1])),
        "outcome": Column(pa.Int64, nullable=False, checks=Check.isin([0, 1])),
    },
    coerce=True,
    strict=False,
    index=Index(pa.Int64, nullable=False),
)
