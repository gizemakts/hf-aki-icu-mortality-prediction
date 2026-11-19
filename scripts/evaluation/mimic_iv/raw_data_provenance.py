#!/usr/bin/env python3
"""Record current raw-input and generated-cohort provenance."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from helpers._runtime import RAW_DATA_ROOT
from helpers._raw_integrity import sha256_file

RAW_DEPENDENCIES = [
    "hosp/admissions.csv.gz",
    "hosp/d_labitems.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/labevents.csv.gz",
    "hosp/omr.csv.gz",
    "hosp/patients.csv.gz",
    "hosp/procedures_icd.csv.gz",
    "hosp/transfers.csv.gz",
    "icu/chartevents.csv.gz",
    "icu/d_items.csv.gz",
    "icu/icustays.csv.gz",
    "icu/inputevents.csv.gz",
    "icu/outputevents.csv.gz",
    "icu/procedureevents.csv.gz",
]


def main() -> None:
    raw_root = Path(RAW_DATA_ROOT).resolve()
    cohort_dir = _REPOSITORY_ROOT / "outputs" / "01_cohort"
    phenotype_dir = _REPOSITORY_ROOT / "outputs" / "02_phenotype"
    qc_dir = _REPOSITORY_ROOT / "quality_control"
    qc_dir.mkdir(parents=True, exist_ok=True)

    cohort_path = cohort_dir / "primary_cohort_preoutcome.parquet"
    target_path = phenotype_dir / "aki_target_preoutcome.parquet"
    for path in (cohort_path, target_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    cohort = pd.read_parquet(cohort_path, columns=["subject_id", "stay_id"])
    target = pd.read_parquet(target_path, columns=["stay_id"])
    if len(cohort) != 3457 or cohort["subject_id"].nunique() != 3457 or cohort["stay_id"].nunique() != 3457:
        raise AssertionError("Primary cohort identity check failed before provenance recording.")
    if len(target) != 10083 or target["stay_id"].nunique() != 10083:
        raise AssertionError("AKI target identity check failed before provenance recording.")

    rows: list[dict[str, object]] = []
    for rel in RAW_DEPENDENCIES:
        path = raw_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        header = pd.read_csv(path, nrows=0)
        rows.append(
            {
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "columns": "|".join(map(str, header.columns)),
                "sha256": sha256_file(path),
            }
        )
    raw = pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)
    raw.to_csv(qc_dir / "raw_input_provenance.csv", index=False)

    generated = pd.DataFrame(
        [
            {
                "artifact": "primary_cohort_preoutcome.parquet",
                "path": str(cohort_path.relative_to(_REPOSITORY_ROOT)),
                "rows": len(cohort),
                "logical_identity": "3457 unique subject_id/stay_id pairs; checked above",
            },
            {
                "artifact": "aki_target_preoutcome.parquet",
                "path": str(target_path.relative_to(_REPOSITORY_ROOT)),
                "rows": len(target),
                "logical_identity": "10083 unique stay_id values; checked above",
            },
        ]
    )
    generated.to_csv(qc_dir / "generated_input_provenance.csv", index=False)

    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dependency_count": len(raw),
        "primary_cohort_n": len(cohort),
        "aki_target_stays": len(target),
        "raw_input_hashes_recorded": True,
        "derived_hashes_recorded_before_run_completion": False,
        "historical_generated_hash_gate_used": False,
    }
    (qc_dir / "raw_data_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("RAW DATA PROVENANCE: PASS")


if __name__ == "__main__":
    main()
