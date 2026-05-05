from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Container with all resolved project paths.

    :param root: Repository root.
    :param data_raw: Immutable raw datasets, DVC-tracked.
    :param data_interim: Cleaned, time-aligned intermediate tables.
    :param data_features: Final feature parquet partitions.
    :param mlruns: MLflow tracking store.
    :param reports: Generated figures and HTML reports.
    :param configs: Hydra configuration tree.
    """

    root: Path
    data_raw: Path
    data_interim: Path
    data_features: Path
    mlruns: Path
    reports: Path
    configs: Path

    def ensure(self) -> "ProjectPaths":
        """Create every directory that does not yet exist.

        :returns: The same instance, for fluent chaining.
        """
        for path in (
            self.data_raw,
            self.data_interim,
            self.data_features,
            self.mlruns,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_paths() -> ProjectPaths:
    """Resolve project paths from the environment or defaults.

    Honours ``FLIGHTRISK_ROOT`` if set, otherwise walks up from this module.

    :returns: A frozen :class:`ProjectPaths` instance.
    """
    env_root = os.environ.get("FLIGHTRISK_ROOT")
    if env_root:
        root = Path(env_root).resolve()
    else:
        root = Path(__file__).resolve().parents[2]

    return ProjectPaths(
        root=root,
        data_raw=root / "data" / "raw",
        data_interim=root / "data" / "interim",
        data_features=root / "data" / "features",
        mlruns=root / "mlruns",
        reports=root / "reports",
        configs=root / "configs",
    )
