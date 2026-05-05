from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _reset_paths_and_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FLIGHTRISK_ROOT", str(tmp_path))
    from flightrisk.utils.paths import get_paths

    get_paths.cache_clear()

    from flightrisk.serving import registry as reg_module

    reg_module._REGISTRY = reg_module.ModelRegistry()
    yield
    get_paths.cache_clear()
    reg_module._REGISTRY = reg_module.ModelRegistry()


class _StubRiskModel:
    feature_names: ClassVar[list[str]] = ["f0", "f1"]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(0.5 + 0.1 * X["f0"].fillna(0).values - 0.05 * X["f1"].fillna(0).values, 0, 1)


class _StubUpliftModel:
    feature_names: ClassVar[list[str]] = ["f0", "f1"]

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return 0.05 * X["f0"].fillna(0).values


class _StubSurvivalModel:
    feature_names: ClassVar[list[str]] = ["f0", "f1"]

    def hazard_at_horizon(self, X: pd.DataFrame, *, horizon_days: int) -> np.ndarray:
        scale = horizon_days / 90.0
        return np.clip(scale * (0.3 + 0.2 * X["f0"].fillna(0).values), 0.0, 1.0)


def _stage(track: str, model: object, tmp_path: Path) -> None:
    run_dir = tmp_path / "reports" / track / "run-stub"
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, run_dir / "model.joblib")


def test_health_endpoint_reports_loaded_models(tmp_path: Path) -> None:
    _stage("risk", _StubRiskModel(), tmp_path)
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["loaded"] == {}


def test_score_risk_returns_probabilities(tmp_path: Path) -> None:
    _stage("risk", _StubRiskModel(), tmp_path)
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        payload = {
            "track": "risk",
            "items": [
                {"customer_id": "u1", "features": {"f0": 1.0, "f1": 2.0}},
                {"customer_id": "u2", "features": {"f0": -1.0, "f1": 0.0}},
            ],
        }
        resp = client.post("/score", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["track"] == "risk"
        assert len(body["scores"]) == 2
        for item in body["scores"]:
            assert 0.0 <= item["score"] <= 1.0


def test_score_uplift_dispatches_correctly(tmp_path: Path) -> None:
    _stage("uplift", _StubUpliftModel(), tmp_path)
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/score",
            json={
                "track": "uplift",
                "items": [
                    {"customer_id": "u1", "features": {"f0": 4.0, "f1": 0.0}},
                ],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["scores"][0]["score"] == pytest.approx(0.2)


def test_score_survival_requires_horizon(tmp_path: Path) -> None:
    _stage("survival", _StubSurvivalModel(), tmp_path)
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/score",
            json={
                "track": "survival",
                "items": [{"customer_id": "u1", "features": {"f0": 1.0, "f1": 0.0}}],
            },
        )
        assert resp.status_code == 400


def test_score_survival_returns_horizon_hazard(tmp_path: Path) -> None:
    _stage("survival", _StubSurvivalModel(), tmp_path)
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/score",
            json={
                "track": "survival",
                "horizon_days": 90,
                "items": [{"customer_id": "u1", "features": {"f0": 1.0, "f1": 0.0}}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "1 - S(t=90d)" in body["score_meaning"]
        assert 0.0 <= body["scores"][0]["score"] <= 1.0


def test_score_returns_503_when_artifact_missing() -> None:
    from flightrisk.serving.api import app

    with TestClient(app) as client:
        resp = client.post(
            "/score",
            json={
                "track": "risk",
                "items": [{"features": {"f0": 0.0}}],
            },
        )
        assert resp.status_code == 503
