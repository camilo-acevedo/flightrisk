from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from flightrisk import __version__
from flightrisk.serving.registry import LoadedModel, ModelRegistry, get_registry
from flightrisk.serving.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    HealthResponse,
    ScoreResponseItem,
)
from flightrisk.serving.security import require_api_key
from flightrisk.utils.logging import get_logger

_log = get_logger(__name__)

_DEFAULT_LIMIT = os.environ.get("FLIGHTRISK_RATE_LIMIT", "60/minute")

limiter = Limiter(key_func=get_remote_address, default_limits=[_DEFAULT_LIMIT])

app = FastAPI(
    title="flightrisk scoring API",
    description="Swappable scoring across the risk, survival, and uplift tracks.",
    version=__version__,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


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
@limiter.exempt
def health(request: Request) -> HealthResponse:
    """Return liveness information and the currently cached models.

    Excluded from rate-limiting so liveness checks always succeed.

    :param request: The incoming request, required by slowapi's signature.
    :returns: A :class:`HealthResponse` payload.
    """
    return HealthResponse(loaded=get_registry().loaded_tracks())


@app.post("/score", response_model=BatchScoreResponse)
@limiter.limit(_DEFAULT_LIMIT)
def score(
    request: Request,
    payload: BatchScoreRequest,
    api_key: str = Depends(require_api_key),
) -> BatchScoreResponse:
    """Score a batch of customers with the requested track.

    Protected by the ``X-API-Key`` header (when ``FLIGHTRISK_API_KEY`` /
    ``FLIGHTRISK_API_KEYS`` is set) and rate-limited per client IP.

    :param request: The incoming request, required by slowapi.
    :param payload: A :class:`BatchScoreRequest` payload.
    :param api_key: The validated API key (or sentinel when auth is disabled).
    :returns: A :class:`BatchScoreResponse` with per-customer scores.
    :raises HTTPException: When the model artifact is missing or invalid.
    """
    registry: ModelRegistry = get_registry()
    try:
        loaded = registry.get(payload.track, run_id=payload.run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"model artifact unavailable: {exc}") from exc

    _log.info("score track=%s caller=%s n=%d", payload.track, api_key, len(payload.items))
    scores, meaning = _score(loaded, payload)
    items = [
        ScoreResponseItem(customer_id=req.customer_id, score=float(score_value))
        for req, score_value in zip(payload.items, scores, strict=True)
    ]
    return BatchScoreResponse(
        track=payload.track,
        run_id=loaded.run_id,
        score_meaning=meaning,
        scores=items,
    )
