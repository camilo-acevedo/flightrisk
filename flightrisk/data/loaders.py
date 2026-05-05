from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from flightrisk.data import schemas
from flightrisk.utils.logging import get_logger
from flightrisk.utils.paths import get_paths

_log = get_logger(__name__)


@dataclass(frozen=True)
class KKBoxArtifacts:
    """Bundle of validated KKBox raw frames.

    :param members: User master table.
    :param transactions: Transaction history (one row per payment event).
    :param user_logs: Daily listening logs.
    :param labels: Training labels keyed by ``msno``.
    """

    members: pd.DataFrame
    transactions: pd.DataFrame
    user_logs: pd.DataFrame
    labels: pd.DataFrame


def _kkbox_dir() -> Path:
    """Return the on-disk location of the raw KKBox dataset.

    :returns: ``data/raw/kkbox`` resolved from the project root.
    """
    return get_paths().data_raw / "kkbox"


def _read_csv_validated(path: Path, schema) -> pd.DataFrame:
    """Read a CSV and validate it against a pandera schema.

    :param path: Path to the CSV file.
    :param schema: Pandera :class:`DataFrameSchema` to enforce.
    :returns: The validated frame.
    :raises FileNotFoundError: If ``path`` is missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required raw file not found: {path}. Run `flightrisk data pull` first."
        )
    _log.info("reading %s", path.name)
    frame = pd.read_csv(path)
    return schema.validate(frame, lazy=True)


def load_kkbox(*, sample_frac: float | None = None, seed: int = 1337) -> KKBoxArtifacts:
    """Load and validate the four KKBox raw tables.

    A small ``sample_frac`` is useful in tests and notebooks — sampling is
    performed on user IDs so transactions and logs stay consistent for each
    sampled user.

    :param sample_frac: Optional fraction in ``(0, 1]`` to subsample by user ID.
    :param seed: Seed used when ``sample_frac`` is provided.
    :returns: A :class:`KKBoxArtifacts` bundle ready for feature building.
    :raises FileNotFoundError: If any of the four raw files is missing.
    :raises ValueError: If ``sample_frac`` is out of range.
    """
    if sample_frac is not None and not 0 < sample_frac <= 1.0:
        raise ValueError("sample_frac must lie in (0, 1] when provided")

    base = _kkbox_dir()
    members = _read_csv_validated(base / "members_v3.csv", schemas.KKBOX_MEMBERS_SCHEMA)
    transactions = _read_csv_validated(base / "transactions.csv", schemas.KKBOX_TRANSACTIONS_SCHEMA)
    user_logs = _read_csv_validated(base / "user_logs.csv", schemas.KKBOX_USER_LOGS_SCHEMA)
    labels = _read_csv_validated(base / "train.csv", schemas.KKBOX_LABELS_SCHEMA)

    if sample_frac is not None:
        sampled_ids = members["msno"].sample(frac=sample_frac, random_state=seed)
        keep = set(sampled_ids)
        members = members[members["msno"].isin(keep)].reset_index(drop=True)
        transactions = transactions[transactions["msno"].isin(keep)].reset_index(drop=True)
        user_logs = user_logs[user_logs["msno"].isin(keep)].reset_index(drop=True)
        labels = labels[labels["msno"].isin(keep)].reset_index(drop=True)

    return KKBoxArtifacts(
        members=members,
        transactions=transactions,
        user_logs=user_logs,
        labels=labels,
    )


@dataclass(frozen=True)
class OrangeBelgiumArtifacts:
    """Bundle for the Orange Belgium uplift benchmark.

    :param features: 178 anonymised covariates.
    :param treatment: 0/1 treatment column from the original RCT.
    :param outcome: 0/1 outcome (retention indicator).
    """

    features: pd.DataFrame
    treatment: pd.Series
    outcome: pd.Series


def _orange_dir() -> Path:
    """Return the on-disk location of the raw Orange Belgium dataset.

    :returns: ``data/raw/orange-belgium`` resolved from the project root.
    """
    return get_paths().data_raw / "orange-belgium"


def load_orange_belgium(path: str | Path | None = None) -> OrangeBelgiumArtifacts:
    """Load and validate the Orange Belgium uplift benchmark.

    :param path: Optional override pointing at a single Parquet or CSV file.
        Defaults to ``data/raw/orange-belgium/orange_belgium.parquet`` and
        falls back to ``orange_belgium.csv`` if Parquet is absent.
    :returns: An :class:`OrangeBelgiumArtifacts` bundle.
    :raises FileNotFoundError: If no input file is found.
    """
    if path is None:
        base = _orange_dir()
        candidates = [base / "orange_belgium.parquet", base / "orange_belgium.csv"]
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"Orange Belgium raw file not found under {base}. "
                "Run `flightrisk data pull` first."
            )
    path = Path(path)
    _log.info("reading %s", path.name)

    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    schemas.ORANGE_UPLIFT_SCHEMA.validate(frame[["treatment", "outcome"]], lazy=True)
    treatment = frame["treatment"].astype(int)
    outcome = frame["outcome"].astype(int)
    features = frame.drop(columns=["treatment", "outcome"])
    return OrangeBelgiumArtifacts(features=features, treatment=treatment, outcome=outcome)
