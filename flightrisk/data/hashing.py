from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 of a single file, streaming in 1 MiB chunks.

    :param path: Path to the file.
    :returns: Hex digest of the file contents.
    :raises FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths: Iterable[str | Path]) -> str:
    """Hash a deterministic ordering of files into one digest.

    Each file is hashed individually and the resulting ``name:digest`` lines are
    fed into a final SHA-256 in sorted order, so the result depends on file
    contents and names only, not directory ordering.

    :param paths: Iterable of file paths under a common root.
    :returns: Combined hex digest.
    """
    entries = []
    for raw in paths:
        path = Path(raw)
        entries.append((path.name, sha256_file(path)))
    entries.sort()
    digest = hashlib.sha256()
    for name, file_digest in entries:
        digest.update(f"{name}:{file_digest}\n".encode("utf-8"))
    return digest.hexdigest()
