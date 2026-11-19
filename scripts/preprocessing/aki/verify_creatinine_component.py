#!/usr/bin/env python3
"""Verify the generated KDIGO creatinine component before downstream use."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from helpers._runtime import RAW_DATA_ROOT as _RAW_DATA_ROOT


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    os.environ.setdefault("HF_AKI_PROJECT_ROOT", str(_REPOSITORY_ROOT))
    os.environ.setdefault("MIMIC_ROOT", str(_RAW_DATA_ROOT))

    root = _REPOSITORY_ROOT
    creatinine_dir = root / "outputs" / "02_phenotype" / "creatinine"
    qc_dir = root / "quality_control" / "phenotype"
    qc_dir.mkdir(parents=True, exist_ok=True)

    path = creatinine_dir / "kdigo_creatinine.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    prohibited = ("expire", "mortality", "death", "outcome")
    bad = [c for c in df.columns if any(token in c.lower() for token in prohibited)]
    if bad:
        raise AssertionError(f"Creatinine component is not outcome-blind: {bad}")

    required = ["stay_id", "kdigo_creatinine_stage", "creatinine_evaluable"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Required creatinine columns missing: {missing}")
    if df["stay_id"].duplicated().any():
        raise AssertionError("Creatinine component is not unique by stay_id.")

    manifest = {
        "status": "PASS",
        "source": str(path.relative_to(root)),
        "rows": int(len(df)),
        "sha256": sha256_file(path),
        "outcome_blind": True,
    }
    (qc_dir / "creatinine_verification_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("CREATININE COMPONENT VERIFICATION: PASS")


if __name__ == "__main__":
    main()
