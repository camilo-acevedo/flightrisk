from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from flightrisk.utils.paths import get_paths


class Settings(BaseSettings):
    """Process-wide settings resolved from environment variables.

    :param mlflow_tracking_uri: MLflow tracking URI; defaults to a local store.
    :param mlflow_experiment: Default experiment name for runs.
    :param random_seed: Global seed for reproducibility.
    :param kaggle_username: Kaggle credential, used only by data ingestion.
    :param kaggle_key: Kaggle API key, used only by data ingestion.
    """

    model_config = SettingsConfigDict(
        env_prefix="FLIGHTRISK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str | None = Field(default=None)
    mlflow_experiment: str = Field(default="flightrisk")
    random_seed: int = Field(default=1337)
    kaggle_username: str | None = Field(default=None)
    kaggle_key: str | None = Field(default=None)

    def resolved_tracking_uri(self) -> str:
        """Return a concrete MLflow tracking URI.

        Falls back to a ``file://`` URI rooted at ``mlruns/`` inside the repo.

        :returns: A URI string MLflow can consume.
        """
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        mlruns: Path = get_paths().mlruns
        return mlruns.resolve().as_uri()


def get_settings() -> Settings:
    """Build a fresh :class:`Settings` instance from the environment.

    :returns: Settings populated from env vars and ``.env`` if present.
    """
    return Settings()
