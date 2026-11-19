"""Shared exact-file integrity helpers for licensed raw inputs."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw_hash_manifest(path: Path, expected_count: int = 14) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rel = row["relative_path"]
            if rel in expected:
                raise AssertionError(f"Duplicate raw-input hash record: {rel}")
            expected[rel] = {"size_bytes": int(row["size_bytes"]), "sha256": row["sha256"]}
    if len(expected) != expected_count:
        raise AssertionError(f"Expected {expected_count} raw-input hash records, found {len(expected)}")
    return expected
