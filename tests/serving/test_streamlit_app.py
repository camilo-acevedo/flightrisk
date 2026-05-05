from __future__ import annotations

import importlib
from pathlib import Path

import joblib
import pytest


@pytest.fixture(autouse=True)
def _reset_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FLIGHTRISK_ROOT", str(tmp_path))
    from flightrisk.utils.paths import get_paths

    get_paths.cache_clear()
    yield
    get_paths.cache_clear()


def test_streamlit_module_imports() -> None:
    module = importlib.import_module("flightrisk.serving.streamlit_app")
    assert hasattr(module, "main")


def test_list_runs_orders_by_recency(tmp_path: Path) -> None:
    from flightrisk.serving.streamlit_app import _list_runs

    base = tmp_path / "reports" / "risk"
    (base / "run-old").mkdir(parents=True)
    (base / "run-new").mkdir(parents=True)
    joblib.dump({"k": 1}, base / "run-old" / "model.joblib")
    joblib.dump({"k": 2}, base / "run-new" / "model.joblib")
    import os

    os.utime(base / "run-old" / "model.joblib", (1, 1))
    os.utime(base / "run-new" / "model.joblib", (2, 2))
    os.utime(base / "run-old", (1, 1))
    os.utime(base / "run-new", (2, 2))

    runs = _list_runs("risk")
    assert runs[0] == "run-new"
    assert "run-old" in runs


def test_list_runs_returns_empty_when_track_missing() -> None:
    from flightrisk.serving.streamlit_app import _list_runs

    assert _list_runs("survival") == []
