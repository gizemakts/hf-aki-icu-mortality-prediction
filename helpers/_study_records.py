"""File integrity checks for study inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from helpers._raw_integrity import sha256_file


OUTCOME_MANIFEST = Path(
    "quality_control/eicu_external_validation/eicu_outcome_attachment/"
    "EICU_OUTCOME_EVALUABLE_LOCK_MANIFEST.json"
)
PERFORMANCE_MANIFEST = Path(
    "quality_control/eicu_external_validation/eicu_locked_performance/"
    "primary_performance_lock/EICU_LOCKED_PRIMARY_PERFORMANCE_LOCK_MANIFEST.json"
)
OUTCOME_MANIFEST_SHA256 = "460fba8e83d2acdd7b8316d4679bff5d9338027cd558d073e196ce23e8d89766"
PERFORMANCE_MANIFEST_SHA256 = "cee01e97040fa27fe794890774cca3030a6aa0645b12b480009be03c0ecdd0c8"


def verify_study_file(path: Path, expected_sha256: str) -> Path:
    """Verify the contents of the specified study file."""
    if not path.is_file():
        raise FileNotFoundError(f"Required study file is missing: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}")
    return path


def add_manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outcome-manifest", type=Path, default=OUTCOME_MANIFEST)
    parser.add_argument("--performance-manifest", type=Path, default=PERFORMANCE_MANIFEST)


def manifest_paths(args: argparse.Namespace) -> dict[Path, Path]:
    return {
        OUTCOME_MANIFEST: args.outcome_manifest,
        PERFORMANCE_MANIFEST: args.performance_manifest,
    }
