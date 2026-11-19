#!/usr/bin/env python3
"""Assemble the primary analysis cohort after outcome-blind AKI adjudication."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from helpers._runtime import RAW_DATA_ROOT as _RAW_DATA_ROOT
from helpers._runtime import display_table

EXPECTED_PRIMARY_N = 3457
EXPECTED_PRIMARY_EVENTS = 707
EXPECTED_CONSECUTIVE_UO_N = 2972
EXPECTED_TARGET_N = 10083

OUTCOME_TOKENS = (
    "death",
    "mortality",
    "expire",
    "outcome",
    "deathtime",
    "hospital_expire",
    "survival",
)


def _require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _outcome_like(columns) -> list[str]:
    return [
        str(c)
        for c in columns
        if any(token in str(c).lower() for token in OUTCOME_TOKENS)
    ]


def _assert_outcome_blind(df: pd.DataFrame, label: str) -> None:
    bad = _outcome_like(df.columns)
    if bad:
        raise AssertionError(f"{label} contains outcome-like fields: {bad}")


def _assert_unique(df: pd.DataFrame, column: str, label: str) -> None:
    if df[column].isna().any():
        raise AssertionError(f"{label}: null {column} values")
    if df[column].duplicated().any():
        raise AssertionError(f"{label}: duplicate {column} values")


def main() -> None:
    """Validate the adjudicated phenotype, persist the primary cohort, then attach outcome."""

    os.environ.setdefault("HF_AKI_PROJECT_ROOT", str(_REPOSITORY_ROOT))
    os.environ.setdefault("MIMIC_ROOT", str(_RAW_DATA_ROOT))

    root = _REPOSITORY_ROOT
    cohort_dir = root / "outputs" / "01_cohort"
    phenotype_dir = root / "outputs" / "02_phenotype"
    adjudication_dir = phenotype_dir / "adjudication"
    qc_dir = root / "quality_control" / "cohort" / "primary"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    phenotype_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    target_path = _require(adjudication_dir / "adjudicated_aki_target_preoutcome.parquet")
    index_path = _require(adjudication_dir / "adjudicated_primary_cohort_preoutcome.parquet")
    consecutive_path = _require(adjudication_dir / "consecutive_urine_output_cohort_preoutcome.parquet")
    adjudication_manifest_path = _require(adjudication_dir / "adjudication_manifest.json")
    landmark_path = _require(cohort_dir / "landmark_population.parquet")

    adjudication_manifest = json.loads(adjudication_manifest_path.read_text(encoding="utf-8"))
    if adjudication_manifest.get("status") != "PASS":
        raise AssertionError("Adjudication manifest did not pass.")
    if adjudication_manifest.get("outcome_blind") is not True:
        raise AssertionError("Adjudication manifest is not marked outcome-blind.")
    if int(adjudication_manifest.get("expanded_target_stays", -1)) != EXPECTED_TARGET_N:
        raise AssertionError("Adjudication target count mismatch.")

    target = pd.read_parquet(target_path)
    index_preoutcome = pd.read_parquet(index_path)
    consecutive = pd.read_parquet(consecutive_path)

    _assert_outcome_blind(target, "AKI target")
    _assert_outcome_blind(index_preoutcome, "primary preoutcome cohort")
    _assert_outcome_blind(consecutive, "consecutive-urine-output sensitivity cohort")

    if len(target) != EXPECTED_TARGET_N:
        raise AssertionError(f"AKI target N={len(target)}, expected {EXPECTED_TARGET_N}.")
    if len(index_preoutcome) != EXPECTED_PRIMARY_N:
        raise AssertionError(
            f"Primary preoutcome cohort N={len(index_preoutcome)}, expected {EXPECTED_PRIMARY_N}."
        )
    if len(consecutive) != EXPECTED_CONSECUTIVE_UO_N:
        raise AssertionError(
            f"Consecutive-urine-output sensitivity N={len(consecutive)}, "
            f"expected {EXPECTED_CONSECUTIVE_UO_N}."
        )

    _assert_unique(index_preoutcome, "subject_id", "primary preoutcome cohort")
    _assert_unique(index_preoutcome, "stay_id", "primary preoutcome cohort")

    aki_positive = index_preoutcome["aki_positive"].fillna(False).astype(bool)
    if not aki_positive.all():
        raise AssertionError("Primary cohort contains a stay without an AKI-positive phenotype.")

    kdigo_stage = pd.to_numeric(index_preoutcome["kdigo_stage"], errors="coerce")
    if kdigo_stage.isna().any() or not kdigo_stage.isin([1, 2, 3]).all():
        raise AssertionError("Primary cohort contains an invalid KDIGO stage.")

    # Persist the outcome-blind scientific sources before outcome attachment.
    primary_preoutcome_path = cohort_dir / "primary_cohort_preoutcome.parquet"
    aki_target_path = phenotype_dir / "aki_target_preoutcome.parquet"
    index_preoutcome.to_parquet(primary_preoutcome_path, index=False)
    target.to_parquet(aki_target_path, index=False)

    persisted = pd.read_parquet(primary_preoutcome_path)
    _assert_outcome_blind(persisted, "persisted primary preoutcome cohort")
    _assert_unique(persisted, "subject_id", "persisted primary preoutcome cohort")
    _assert_unique(persisted, "stay_id", "persisted primary preoutcome cohort")
    if len(persisted) != EXPECTED_PRIMARY_N:
        raise AssertionError("Primary cohort changed during persistence.")

    landmark_columns = [
        "stay_id",
        "outtime",
        "dischtime",
        "deathtime",
        "hospital_expire_flag",
        "gender",
        "age",
    ]
    landmark = pd.read_parquet(landmark_path, columns=landmark_columns)
    primary_with_outcome = persisted.merge(landmark, on="stay_id", how="left", validate="1:1")

    flag = pd.to_numeric(primary_with_outcome["hospital_expire_flag"], errors="coerce")
    if flag.isna().any() or not flag.isin([0, 1]).all():
        raise AssertionError("Invalid hospital_expire_flag after outcome attachment.")
    events = int(flag.sum())
    if events != EXPECTED_PRIMARY_EVENTS:
        raise AssertionError(
            f"Primary cohort events={events}, expected {EXPECTED_PRIMARY_EVENTS}."
        )

    mortality = float(events / len(primary_with_outcome))
    primary_outcome_path = cohort_dir / "primary_cohort_with_outcome.parquet"
    primary_with_outcome.to_parquet(primary_outcome_path, index=False)

    stage_distribution = {
        str(int(stage)): int(count)
        for stage, count in kdigo_stage.value_counts().sort_index().items()
    }
    summary = {
        "primary_n": int(len(primary_with_outcome)),
        "events": events,
        "in_hospital_mortality": mortality,
        "kdigo_stage_distribution": stage_distribution,
        "consecutive_urine_output_sensitivity_n": int(len(consecutive)),
    }
    (cohort_dir / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    manifest = {
        "status": "PASS",
        "outcome_attached_after_preoutcome_persistence": True,
        "primary_n": int(len(primary_with_outcome)),
        "events": events,
        "aki_target_stays": int(len(target)),
        "consecutive_urine_output_sensitivity_n": int(len(consecutive)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (cohort_dir / "primary_cohort_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    qc = pd.DataFrame(
        [
            {
                "primary_n": len(primary_with_outcome),
                "events": events,
                "mortality_percent": 100 * mortality,
                "kdigo_stage_1_n": stage_distribution.get("1", 0),
                "kdigo_stage_2_n": stage_distribution.get("2", 0),
                "kdigo_stage_3_n": stage_distribution.get("3", 0),
                "consecutive_urine_output_sensitivity_n": len(consecutive),
            }
        ]
    )
    qc.to_csv(qc_dir / "primary_cohort_summary.csv", index=False)
    display_table(qc)
    print("PRIMARY COHORT ASSEMBLY: PASS")


if __name__ == "__main__":
    main()
