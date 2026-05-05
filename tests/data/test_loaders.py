from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flightrisk.data.loaders import load_kkbox, load_orange_belgium


def _write_kkbox(tmp_path: Path) -> Path:
    base = tmp_path / "data" / "raw" / "kkbox"
    base.mkdir(parents=True)

    pd.DataFrame(
        {
            "msno": ["u1", "u2", "u3"],
            "city": [1, 2, 3],
            "bd": [25, 30, 35],
            "gender": ["male", "female", "male"],
            "registered_via": [3, 7, 9],
            "registration_init_time": [20170101, 20170201, 20170301],
        }
    ).to_csv(base / "members_v3.csv", index=False)

    pd.DataFrame(
        {
            "msno": ["u1", "u2"],
            "payment_method_id": [41, 41],
            "payment_plan_days": [30, 30],
            "plan_list_price": [149, 149],
            "actual_amount_paid": [149, 0],
            "is_auto_renew": [1, 0],
            "transaction_date": [20170101, 20170201],
            "membership_expire_date": [20170131, 20170228],
            "is_cancel": [0, 1],
        }
    ).to_csv(base / "transactions.csv", index=False)

    pd.DataFrame(
        {
            "msno": ["u1", "u2", "u3"],
            "date": [20170105, 20170210, 20170305],
            "num_25": [10, 5, 0],
            "num_50": [3, 2, 0],
            "num_75": [2, 1, 0],
            "num_985": [1, 0, 0],
            "num_100": [40, 12, 0],
            "num_unq": [30, 10, 0],
            "total_secs": [12000.0, 4000.0, 0.0],
        }
    ).to_csv(base / "user_logs.csv", index=False)

    pd.DataFrame({"msno": ["u1", "u2", "u3"], "is_churn": [0, 1, 0]}).to_csv(
        base / "train.csv", index=False
    )
    return tmp_path


def test_load_kkbox_returns_validated_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_kkbox(tmp_path)
    monkeypatch.setenv("FLIGHTRISK_ROOT", str(root))
    from flightrisk.utils.paths import get_paths

    get_paths.cache_clear()

    bundle = load_kkbox()
    assert len(bundle.members) == 3
    assert len(bundle.transactions) == 2
    assert len(bundle.user_logs) == 3
    assert set(bundle.labels["msno"]) == {"u1", "u2", "u3"}


def test_load_kkbox_sample_frac_consistent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_kkbox(tmp_path)
    monkeypatch.setenv("FLIGHTRISK_ROOT", str(root))
    from flightrisk.utils.paths import get_paths

    get_paths.cache_clear()

    bundle = load_kkbox(sample_frac=0.34, seed=0)
    sampled_ids = set(bundle.members["msno"])
    assert set(bundle.transactions["msno"]).issubset(sampled_ids)
    assert set(bundle.user_logs["msno"]).issubset(sampled_ids)
    assert set(bundle.labels["msno"]).issubset(sampled_ids)


def test_load_kkbox_sample_frac_validates_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_kkbox(tmp_path)
    monkeypatch.setenv("FLIGHTRISK_ROOT", str(root))
    from flightrisk.utils.paths import get_paths

    get_paths.cache_clear()
    with pytest.raises(ValueError):
        load_kkbox(sample_frac=1.5)


def test_load_orange_belgium_csv(tmp_path: Path) -> None:
    csv = tmp_path / "orange.csv"
    pd.DataFrame(
        {
            "f0": [0.1, 0.2, 0.3, 0.4],
            "f1": [1.0, 2.0, 3.0, 4.0],
            "treatment": [0, 1, 0, 1],
            "outcome": [0, 0, 1, 1],
        }
    ).to_csv(csv, index=False)

    bundle = load_orange_belgium(csv)
    assert bundle.features.columns.tolist() == ["f0", "f1"]
    assert bundle.treatment.tolist() == [0, 1, 0, 1]
    assert bundle.outcome.tolist() == [0, 0, 1, 1]


def test_load_orange_belgium_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLIGHTRISK_ROOT", str(tmp_path))
    from flightrisk.utils.paths import get_paths

    get_paths.cache_clear()
    with pytest.raises(FileNotFoundError):
        load_orange_belgium()
