from __future__ import annotations

import os

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

API_KEY_HEADER = "X-API-Key"
_HEADER_SCHEME = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


def _allowed_keys() -> set[str]:
    """Return the set of accepted API keys.

    Reads ``FLIGHTRISK_API_KEYS`` (comma-separated) and ``FLIGHTRISK_API_KEY``
    (single value); both are honoured so deployments can rotate easily.

    :returns: A set of allowed key strings; empty when no key is configured
        (auth then becomes a no-op for local development).
    """
    raw_multi = os.environ.get("FLIGHTRISK_API_KEYS", "")
    raw_single = os.environ.get("FLIGHTRISK_API_KEY", "")
    keys = {k.strip() for k in raw_multi.split(",") if k.strip()}
    if raw_single.strip():
        keys.add(raw_single.strip())
    return keys


def require_api_key(
    request: Request,
    api_key: str | None = Security(_HEADER_SCHEME),
) -> str:
    """FastAPI dependency that enforces the ``X-API-Key`` header.

    When no keys are configured (no ``FLIGHTRISK_API_KEY*`` env var present),
    auth is treated as disabled and the dependency returns the literal
    ``"<unauthenticated>"`` so route handlers can still log a caller id.

    :param request: The incoming request; included for completeness so callers
        can inject this dependency without manually wiring the header.
    :param api_key: The value of the ``X-API-Key`` header, populated by
        FastAPI's :class:`APIKeyHeader` security scheme.
    :returns: The validated key (or ``"<unauthenticated>"`` when auth is off).
    :raises HTTPException: 401 if a key is required and missing or invalid.
    """
    allowed = _allowed_keys()
    if not allowed:
        return "<unauthenticated>"
    if api_key is None:
        raise HTTPException(status_code=401, detail="missing X-API-Key header")
    if api_key not in allowed:
        raise HTTPException(status_code=401, detail="invalid API key")
    return api_key
