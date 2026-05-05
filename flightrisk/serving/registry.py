from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib

from flightrisk.utils.logging import get_logger
from flightrisk.utils.paths import get_paths

_log = get_logger(__name__)

TrackName = Literal["risk", "survival", "uplift"]


@dataclass(frozen=True)
class LoadedModel:
    """Metadata for a model loaded into the in-process registry.

    :param track: Track identifier (``"risk"``, ``"survival"`` or ``"uplift"``).
    :param run_id: MLflow run identifier (or directory name on disk).
    :param model: The deserialised estimator bundle.
    :param feature_names: Feature ordering required by the model.
    """

    track: TrackName
    run_id: str
    model: object
    feature_names: list[str]


class ModelRegistry:
    """Thread-safe in-memory registry that lazily loads track artifacts.

    The registry resolves paths under ``reports/<track>/`` and caches the most
    recently loaded artifact for each track. Concurrent requests reuse the same
    instance so reloads only happen when the chosen run id changes.
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._lock = threading.RLock()
        self._cache: dict[TrackName, LoadedModel] = {}

    def _resolve(self, track: TrackName, run_id: str | None) -> Path:
        """Return the path of the model artifact for ``track`` and ``run_id``.

        :param track: Track identifier.
        :param run_id: Optional run id; when ``None`` the most recently
            modified subdirectory is used.
        :returns: Path to the ``model.joblib`` file.
        :raises FileNotFoundError: If no artifact can be located.
        """
        base = get_paths().reports / track
        if run_id is not None:
            candidate = base / run_id / "model.joblib"
            if candidate.exists():
                return candidate
            raise FileNotFoundError(candidate)
        if not base.exists():
            raise FileNotFoundError(base)
        runs = sorted(base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for run in runs:
            candidate = run / "model.joblib"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"no model.joblib under {base}")

    def get(self, track: TrackName, run_id: str | None = None) -> LoadedModel:
        """Return a loaded model for ``track``, loading from disk on first use.

        :param track: Track identifier.
        :param run_id: Optional run id; ``None`` selects the latest run.
        :returns: A :class:`LoadedModel` ready for inference.
        :raises FileNotFoundError: If the artifact is missing.
        """
        with self._lock:
            cached = self._cache.get(track)
            if cached is not None and (run_id is None or cached.run_id == run_id):
                return cached
            path = self._resolve(track, run_id)
            run_dir = path.parent
            _log.info("loading %s model from %s", track, path)
            model = joblib.load(path)
            feature_names = getattr(model, "feature_names", None) or []
            loaded = LoadedModel(
                track=track,
                run_id=run_dir.name,
                model=model,
                feature_names=list(feature_names),
            )
            self._cache[track] = loaded
            return loaded

    def loaded_tracks(self) -> dict[TrackName, str]:
        """Return a snapshot of the currently cached run ids per track.

        :returns: Mapping from track name to run id.
        """
        with self._lock:
            return {track: model.run_id for track, model in self._cache.items()}


_REGISTRY = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Return the process-wide :class:`ModelRegistry` instance.

    :returns: The shared registry singleton.
    """
    return _REGISTRY
