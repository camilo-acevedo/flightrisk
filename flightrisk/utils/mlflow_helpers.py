from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping

import mlflow

from flightrisk.config import get_settings
from flightrisk.utils.logging import get_logger

_log = get_logger(__name__)


def configure_mlflow(experiment: str | None = None) -> str:
    """Configure MLflow tracking and ensure the target experiment exists.

    :param experiment: Optional experiment name; defaults to the project setting.
    :returns: The resolved experiment name.
    """
    settings = get_settings()
    uri = settings.resolved_tracking_uri()
    mlflow.set_tracking_uri(uri)
    name = experiment or settings.mlflow_experiment
    mlflow.set_experiment(name)
    _log.info("mlflow tracking_uri=%s experiment=%s", uri, name)
    return name


@contextmanager
def start_run(run_name: str, tags: Mapping[str, str] | None = None) -> Iterator[mlflow.ActiveRun]:
    """Context manager that starts a tagged MLflow run.

    :param run_name: Human-readable run name.
    :param tags: Optional tag mapping (e.g. data hashes, git SHAs).
    :yields: The active MLflow run, for direct ``mlflow.log_*`` calls.
    """
    with mlflow.start_run(run_name=run_name, tags=dict(tags or {})) as run:
        yield run
