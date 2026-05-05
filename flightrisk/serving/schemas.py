from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """Per-customer feature payload submitted for scoring.

    :param customer_id: Optional opaque identifier echoed in the response.
    :param features: Mapping from feature name to numeric or string value.
    """

    customer_id: str | None = Field(default=None)
    features: dict[str, float | int | str | None]


class BatchScoreRequest(BaseModel):
    """Batch payload of multiple :class:`ScoreRequest` items.

    :param track: Which track to score with.
    :param run_id: Optional run id to pin a specific model version.
    :param horizon_days: Required when ``track == "survival"``.
    :param items: Per-customer feature payloads.
    """

    track: Literal["risk", "survival", "uplift"]
    run_id: str | None = None
    horizon_days: int | None = None
    items: list[ScoreRequest]


class ScoreResponseItem(BaseModel):
    """One row of the scoring response.

    :param customer_id: Mirrors the incoming ``customer_id`` if any.
    :param score: Per-row score interpreted per ``track``.
    """

    customer_id: str | None
    score: float


class BatchScoreResponse(BaseModel):
    """Batch scoring response.

    :param track: Track that produced the scores.
    :param run_id: Run id of the model that served the request.
    :param score_meaning: Human-readable description of what ``score`` means.
    :param scores: One :class:`ScoreResponseItem` per request item.
    """

    track: Literal["risk", "survival", "uplift"]
    run_id: str
    score_meaning: str
    scores: list[ScoreResponseItem]


class HealthResponse(BaseModel):
    """Liveness check response.

    :param status: Always ``"ok"`` when the service is reachable.
    :param loaded: Mapping from track name to loaded run id.
    """

    status: Literal["ok"] = "ok"
    loaded: dict[str, str]
