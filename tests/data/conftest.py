from __future__ import annotations

import pytest

from flightrisk.utils.paths import get_paths


@pytest.fixture(autouse=True)
def _reset_paths_cache():
    """Clear the cached :func:`get_paths` result around every data test.

    Tests in this module use ``monkeypatch.setenv("FLIGHTRISK_ROOT", ...)`` to
    point at a temporary tree. Without clearing the cache the result leaks into
    later tests (including the smoke suite) and reports the wrong root.
    """
    get_paths.cache_clear()
    yield
    get_paths.cache_clear()
