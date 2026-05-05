from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FLIGHTRISK_ROOT", str(tmp_path))
    monkeypatch.delenv("FLIGHTRISK_API_KEY", raising=False)
    monkeypatch.delenv("FLIGHTRISK_API_KEYS", raising=False)
    from flightrisk.utils.paths import get_paths

    get_paths.cache_clear()

    from flightrisk.serving import registry as reg_module

    reg_module._REGISTRY = reg_module.ModelRegistry()
    yield
    get_paths.cache_clear()
    reg_module._REGISTRY = reg_module.ModelRegistry()


class _StubRiskModel:
    feature_names: ClassVar[list[str]] = ["f0"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.5, dtype=float)


def _stage(tmp_path: Path) -> None:
    run_dir = tmp_path / "reports" / "risk" / "run-stub"
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(_StubRiskModel(), run_dir / "model.joblib")


def _payload() -> dict:
    return {"track": "risk", "items": [{"features": {"f0": 1.0}}]}


def test_score_passes_when_no_api_key_configured(tmp_path: Path) -> None:
    _stage(tmp_path)
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        resp = client.post("/score", json=_payload())
        assert resp.status_code == 200, resp.text


def test_score_rejects_missing_header_when_key_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    monkeypatch.setenv("FLIGHTRISK_API_KEY", "secret-1")
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        resp = client.post("/score", json=_payload())
        assert resp.status_code == 401
        assert "X-API-Key" in resp.json()["detail"] or "missing" in resp.json()["detail"]


def test_score_rejects_wrong_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stage(tmp_path)
    monkeypatch.setenv("FLIGHTRISK_API_KEY", "secret-1")
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        resp = client.post("/score", json=_payload(), headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401


def test_score_accepts_valid_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stage(tmp_path)
    monkeypatch.setenv("FLIGHTRISK_API_KEY", "secret-1")
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        resp = client.post("/score", json=_payload(), headers={"X-API-Key": "secret-1"})
        assert resp.status_code == 200, resp.text


def test_score_accepts_any_of_multiple_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    monkeypatch.setenv("FLIGHTRISK_API_KEYS", "alpha, beta , gamma")
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        for key in ("alpha", "beta", "gamma"):
            resp = client.post("/score", json=_payload(), headers={"X-API-Key": key})
            assert resp.status_code == 200, f"{key} -> {resp.status_code} {resp.text}"


def test_health_is_unauthenticated_and_unrate_limited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stage(tmp_path)
    monkeypatch.setenv("FLIGHTRISK_API_KEY", "secret-1")
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200
