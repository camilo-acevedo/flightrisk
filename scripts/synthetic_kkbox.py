from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from flightrisk.utils.logging import get_logger
from flightrisk.utils.paths import get_paths

_log = get_logger(__name__)


def generate(n_users: int = 50_000, *, seed: int = 1337) -> dict[str, pd.DataFrame]:
    """Generate a KKBox-shaped synthetic dataset.

    All four frames satisfy the pandera schemas defined in
    :mod:`flightrisk.data.schemas`. The label is correlated with payment and
    listening behaviour so downstream models recover non-trivial signal.

    :param n_users: Number of distinct users to simulate.
    :param seed: Random seed.
    :returns: Dict with keys ``members``, ``transactions``, ``user_logs``,
        ``labels``.
    """
    rng = np.random.default_rng(seed)
    msnos = np.array([f"u{i:07d}" for i in range(n_users)])

    members = pd.DataFrame(
        {
            "msno": msnos,
            "city": rng.integers(1, 25, n_users),
            "bd": np.clip(rng.normal(35, 12, n_users).astype(int), 12, 80),
            "gender": rng.choice(["male", "female", None], n_users, p=[0.45, 0.45, 0.10]),
            "registered_via": rng.choice([3, 4, 7, 9, 13], n_users),
            "registration_init_time": rng.integers(20140101, 20170201, n_users),
        }
    )

    txn_rows = []
    log_rows = []
    is_churn = np.zeros(n_users, dtype=int)

    txn_universe = pd.date_range("2016-09-01", "2017-05-31", freq="D")
    txn_int_universe = txn_universe.strftime("%Y%m%d").astype(int).to_numpy()
    log_universe = pd.date_range("2016-09-01", "2017-04-30", freq="D")
    log_int_universe = log_universe.strftime("%Y%m%d").astype(int).to_numpy()

    for i, msno in enumerate(msnos):
        n_txn = int(rng.integers(2, 10))
        is_paying = rng.uniform() < 0.85
        auto_renew = int(rng.uniform() < 0.65)
        plan_price = int(rng.choice([99, 129, 149, 199]))
        for _t in range(n_txn):
            txn_date = int(rng.choice(txn_int_universe))
            charge_drop = rng.uniform() < 0.05
            paid = 0 if (charge_drop or not is_paying) else plan_price
            cancel = int(rng.uniform() < 0.06 + 0.18 * (1 - auto_renew))
            txn_rows.append(
                (
                    msno,
                    int(rng.choice([36, 38, 39, 40, 41])),
                    30,
                    plan_price,
                    paid,
                    auto_renew,
                    txn_date,
                    txn_date + 30,
                    cancel,
                )
            )

        engagement = rng.uniform(0.0, 1.0)
        n_active_days = int(rng.integers(0, 90) * engagement) + 1
        active_dates = sorted(
            rng.choice(log_int_universe, size=n_active_days, replace=False).tolist()
        )
        for d in active_dates:
            plays_full = int(rng.poisson(40 * engagement))
            plays_partial = int(rng.poisson(15 * engagement))
            log_rows.append(
                (
                    msno,
                    int(d),
                    int(plays_partial * 0.5),
                    int(plays_partial * 0.3),
                    int(plays_partial * 0.15),
                    int(plays_partial * 0.05),
                    plays_full,
                    int(plays_full * 0.7) + 1,
                    float(plays_full * 240 + plays_partial * 60),
                )
            )

        churn_logit = -2.5 + 1.5 * (1 - auto_renew) + 1.5 * (1 - is_paying) - 1.2 * engagement
        is_churn[i] = int(rng.uniform() < 1 / (1 + np.exp(-churn_logit)))

    transactions = pd.DataFrame(
        txn_rows,
        columns=[
            "msno",
            "payment_method_id",
            "payment_plan_days",
            "plan_list_price",
            "actual_amount_paid",
            "is_auto_renew",
            "transaction_date",
            "membership_expire_date",
            "is_cancel",
        ],
    )
    user_logs = pd.DataFrame(
        log_rows,
        columns=[
            "msno",
            "date",
            "num_25",
            "num_50",
            "num_75",
            "num_985",
            "num_100",
            "num_unq",
            "total_secs",
        ],
    )
    labels = pd.DataFrame({"msno": msnos, "is_churn": is_churn})

    return {
        "members": members,
        "transactions": transactions,
        "user_logs": user_logs,
        "labels": labels,
    }


def write(frames: dict[str, pd.DataFrame], *, root: Path | None = None) -> Path:
    """Persist the generated frames as KKBox-style CSVs.

    :param frames: Dict from :func:`generate`.
    :param root: Optional override for the project root.
    :returns: The directory where files were written.
    """
    base = (root or get_paths().data_raw) / "kkbox"
    base.mkdir(parents=True, exist_ok=True)
    frames["members"].to_csv(base / "members_v3.csv", index=False)
    frames["transactions"].to_csv(base / "transactions.csv", index=False)
    frames["user_logs"].to_csv(base / "user_logs.csv", index=False)
    frames["labels"].to_csv(base / "train.csv", index=False)
    _log.info("synthetic kkbox written to %s", base)
    return base


def main() -> None:
    """Command-line entry point.

    :returns: ``None``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    write(generate(n_users=args.n_users, seed=args.seed))


if __name__ == "__main__":
    main()
