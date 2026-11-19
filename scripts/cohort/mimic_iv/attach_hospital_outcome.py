#!/usr/bin/env python3
"""Attach in-hospital mortality after reference-cohort membership is fixed."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

EXPECTED_REFERENCE_N = 3251
EXPECTED_REFERENCE_EVENTS = 695


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    root = Path(os.environ.get("HF_AKI_PROJECT_ROOT", str(_REPOSITORY_ROOT))).resolve()
    cohort_dir = root / "outputs" / "01_cohort"
    validation_dir = root / "outputs" / "05_validation" / "reference_cohort"
    qc_dir = root / "quality_control" / "cohort"
    for d in (cohort_dir, validation_dir, qc_dir):
        d.mkdir(parents=True, exist_ok=True)

    preoutcome_path = require_file(cohort_dir / "reference_cohort_preoutcome.parquet")
    landmark_path = require_file(cohort_dir / "landmark_population.parquet")
    integrity_path = require_file(validation_dir / "preoutcome_integrity_manifest.json")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("status") != "PASS" or integrity.get("outcome_blind") is not True:
        raise AssertionError("Reference-cohort pre-outcome integrity has not passed")

    idx = pd.read_parquet(preoutcome_path)
    if len(idx) != EXPECTED_REFERENCE_N:
        raise AssertionError(f"Reference cohort N={len(idx)}, expected {EXPECTED_REFERENCE_N}")
    outcome = pd.read_parquet(
        landmark_path,
        columns=["stay_id", "outtime", "dischtime", "deathtime", "hospital_expire_flag", "gender", "age"],
    )
    cohort = idx.merge(outcome, on="stay_id", how="left", validate="1:1")
    flag = pd.to_numeric(cohort["hospital_expire_flag"], errors="coerce")
    if flag.isna().any() or not flag.isin([0, 1]).all():
        raise AssertionError("Invalid hospital_expire_flag after outcome attachment")
    cohort["hospital_mortality_after_landmark"] = flag.astype("int8")
    deaths = int(cohort["hospital_mortality_after_landmark"].sum())
    if deaths != EXPECTED_REFERENCE_EVENTS:
        raise AssertionError(f"Reference-cohort deaths={deaths}, expected {EXPECTED_REFERENCE_EVENTS}")

    out = cohort_dir / "reference_cohort_with_outcome.parquet"
    cohort.to_parquet(out, index=False)
    qc = pd.DataFrame([{
        "n": int(len(cohort)),
        "deaths": deaths,
        "mortality": float(deaths / len(cohort)),
        "missing_hospital_expire_flag": int(flag.isna().sum()),
    }])
    qc.to_csv(qc_dir / "reference_outcome_summary.csv", index=False)
    manifest = {
        "status": "PASS",
        "reference_cohort_n": int(len(cohort)),
        "deaths": deaths,
        "in_hospital_mortality": float(deaths / len(cohort)),
        "outcome_attached_after_preoutcome_integrity": True,
    }
    (validation_dir / "outcome_attachment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("REFERENCE COHORT OUTCOME ATTACHMENT: PASS")


if __name__ == "__main__":
    main()
