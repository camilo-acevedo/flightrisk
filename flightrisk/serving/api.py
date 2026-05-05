from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from flightrisk import __version__
from flightrisk.serving.registry import LoadedModel, ModelRegistry, get_registry
from flightrisk.serving.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ScoreResponseItem,
)
from flightrisk.utils.logging import get_logger

_log = get_logger(__name__)

app = FastAPI(
    title="flightrisk scoring API",
    description="Swappable scoring across the risk, survival, and uplift tracks.",
    version=__version__,
)


def _build_feature_frame(items: list[dict[str, Any]], expected: list[str]) -> pd.DataFrame:
    """Coerce a list of feature dicts into the schema the model expects.

    Missing columns are filled with NaN; extra columns are dropped. The order
    matches ``expected`` so model wrappers that use position-based access still
    work.

    :param items: List of feature dicts (one per customer).
    :param expected: Ordered feature names used at fit time.
    :returns: DataFrame ready for the model's ``predict_proba`` /
        ``predict_uplift`` / hazard interface.
    """
    frame = pd.DataFrame(items)
    if expected:
        for col in expected:
            if col not in frame.columns:
                frame[col] = np.nan
        frame = frame[expected]
    return frame


def _score(loaded: LoadedModel, request: BatchScoreRequest) -> tuple[np.ndarray, str]:
    """Dispatch to the right scoring entry point for the loaded track.

    :param loaded: A :class:`LoadedModel` returned by the registry.
    :param request: The incoming batch request.
    :returns: ``(scores, score_meaning)`` tuple.
    :raises HTTPException: If the track is unknown or required parameters are
        missing.
    """
    feature_dicts = [item.features for item in request.items]
    frame = _build_feature_frame(feature_dicts, loaded.feature_names)

    if request.track == "risk":
        proba = loaded.model.predict_proba(frame)
        return np.asarray(proba).ravel(), "P(churn) within model horizon"
    if request.track == "uplift":
        uplift = loaded.model.predict_uplift(frame)
        return np.asarray(uplift).ravel(), "estimated retention uplift if treated"
    if request.track == "survival":
        if request.horizon_days is None:
            raise HTTPException(
                status_code=400,
                detail="horizon_days is required for survival scoring",
            )
        hazard = loaded.model.hazard_at_horizon(frame, horizon_days=request.horizon_days)
        return np.asarray(hazard).ravel(), f"1 - S(t={request.horizon_days}d)"
    raise HTTPException(status_code=400, detail=f"unknown track: {request.track}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return liveness information and the currently cached models.

    :returns: A :class:`HealthResponse` payload.
    """
    return HealthResponse(loaded=get_registry().loaded_tracks())


@app.post("/score", response_model=BatchScoreResponse)
def score(request: BatchScoreRequest) -> BatchScoreResponse:
    """Score a batch of customers with the requested track.

    :param request: A :class:`BatchScoreRequest` payload.
    :returns: A :class:`BatchScoreResponse` with per-customer scores.
    :raises HTTPException: When the model artifact is missing or invalid.
    """
    registry: ModelRegistry = get_registry()
    try:
        loaded = registry.get(request.track, run_id=request.run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"model artifact unavailable: {exc}") from exc

    scores, meaning = _score(loaded, request)
    items = [
        ScoreResponseItem(customer_id=req.customer_id, score=float(score_value))
        for req, score_value in zip(request.items, scores, strict=True)
    ]
    return BatchScoreResponse(
        track=request.track,
        run_id=loaded.run_id,
        score_meaning=meaning,
        scores=items,
    )
