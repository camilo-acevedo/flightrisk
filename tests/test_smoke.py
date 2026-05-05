from __future__ import annotations

import flightrisk
from flightrisk.config import Settings, get_settings
from flightrisk.utils import get_logger, get_paths, seed_everything


def test_package_has_version() -> None:
    assert isinstance(flightrisk.__version__, str)
    assert flightrisk.__version__


def test_paths_resolve_to_repo_root() -> None:
    paths = get_paths()
    assert paths.root.exists()
    assert paths.configs.is_dir()


def test_logger_is_singleton_per_name() -> None:
    a = get_logger("flightrisk.smoke")
    b = get_logger("flightrisk.smoke")
    assert a is b


def test_settings_defaults() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.random_seed >= 0
    assert settings.resolved_tracking_uri().startswith(("file:", "http", "sqlite", "databricks"))


def test_seed_everything_returns_same_seed() -> None:
    assert seed_everything(7) == 7
