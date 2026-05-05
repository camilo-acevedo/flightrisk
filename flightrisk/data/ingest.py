from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from flightrisk.utils.logging import get_logger
from flightrisk.utils.paths import get_paths

_log = get_logger(__name__)


def _has_dvc() -> bool:
    """Return whether the ``dvc`` executable is on PATH.

    :returns: ``True`` if DVC is installed, ``False`` otherwise.
    """
    return shutil.which("dvc") is not None


def dvc_pull(targets: list[str] | None = None) -> int:
    """Invoke ``dvc pull`` to materialise raw datasets.

    :param targets: Optional list of DVC targets to pull. If ``None``, pull
        everything tracked by the project.
    :returns: The exit code of the DVC subprocess.
    :raises RuntimeError: If DVC is not installed.
    """
    if not _has_dvc():
        raise RuntimeError(
            "dvc is not installed. `pip install -e \".[data]\"` or install DVC manually."
        )
    cmd = ["dvc", "pull"]
    if targets:
        cmd.extend(targets)
    _log.info("running %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=get_paths().root, check=False)
    return proc.returncode


def kaggle_download_kkbox() -> Path:
    """Download the KKBox raw bundle via the Kaggle CLI as a fallback.

    Requires ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` (or
    ``FLIGHTRISK_KAGGLE_USERNAME`` / ``FLIGHTRISK_KAGGLE_KEY``) and the user to
    have accepted the competition rules.

    :returns: The directory where files were extracted.
    :raises RuntimeError: If Kaggle credentials are missing or the CLI fails.
    """
    user = os.environ.get("KAGGLE_USERNAME") or os.environ.get("FLIGHTRISK_KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or os.environ.get("FLIGHTRISK_KAGGLE_KEY")
    if not (user and key):
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY in env."
        )
    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"] = key

    target = get_paths().data_raw / "kkbox"
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        "kkbox-churn-prediction-challenge",
        "-p",
        str(target),
    ]
    _log.info("running %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"kaggle CLI failed with code {proc.returncode}")
    return target
