from __future__ import annotations

import logging
import os
import sys
from functools import lru_cache

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%dT%H:%M:%S"


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger named ``name``.

    Reads ``FLIGHTRISK_LOG_LEVEL`` (default ``INFO``). Subsequent calls with the
    same name are cached so handlers are never duplicated.

    :param name: Module-qualified logger name, typically ``__name__``.
    :returns: A :class:`logging.Logger` writing to stderr.
    """
    level = os.environ.get("FLIGHTRISK_LOG_LEVEL", "INFO").upper()
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(fmt=_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
