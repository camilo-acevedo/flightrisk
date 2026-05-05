from __future__ import annotations

from pathlib import Path

import pytest

from flightrisk.data.hashing import sha256_file, sha256_paths


def _write(tmp_path: Path, name: str, content: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    p = _write(tmp_path, "x.bin", b"hello flightrisk")
    digest = sha256_file(p)
    assert len(digest) == 64
    assert digest == sha256_file(p)


def test_sha256_file_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        sha256_file(tmp_path / "missing.bin")


def test_sha256_paths_is_order_invariant(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.bin", b"first")
    b = _write(tmp_path, "b.bin", b"second")
    assert sha256_paths([a, b]) == sha256_paths([b, a])


def test_sha256_paths_changes_when_content_changes(tmp_path: Path) -> None:
    a = _write(tmp_path, "a.bin", b"v1")
    b = _write(tmp_path, "b.bin", b"v2")
    digest = sha256_paths([a, b])
    a.write_bytes(b"v1-updated")
    assert sha256_paths([a, b]) != digest
