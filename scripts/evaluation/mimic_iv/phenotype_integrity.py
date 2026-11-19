#!/usr/bin/env python3
"""Validate the current AKI phenotype, cohort, and prespecified sensitivity populations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


EXPECTED_COUNTS = {
    "01_primary.parquet": 3457,
    "02_uo_consecutive_ibw_elapsed.parquet": 2972,
    "03_hf_wide.parquet": 7006,
    "04_prior_only_krt.parquet": 3583,
    "05_rolling_only_creatinine.parquet": 3279,
    "06_hierarchical_creatinine.parquet": 3405,
    "07_hf_recency_3y.parquet": 3174,
    "08_hf_recency_1y.parquet": 2710,
    "09_uo_mean_abw_elapsed.parquet": 4019,
    "10_fixed_first_eligible.parquet": 2741,
    "11_uo_source_p95_cap.parquet": 3365,
    "12_uo_no_interval_cap.parquet": 3698,
}


def _outcome_like(columns: list[str] | pd.Index) -> list[str]:
    tokens = ("death", "mortality", "expire", "outcome", "deathtime", "survival")
    return [c for c in columns if any(token in c.lower() for token in tokens)]


def _identity(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return set(
        zip(
            pd.to_numeric(frame["subject_id"], errors="raise").astype("int64"),
            pd.to_numeric(frame["stay_id"], errors="raise").astype("int64"),
        )
    )


def main() -> None:
    outputs = _REPOSITORY_ROOT / "outputs"
    cohort_dir = outputs / "01_cohort"
    phenotype_dir = outputs / "02_phenotype"
    population_dir = outputs / "06_sensitivity" / "populations"
    qc_dir = _REPOSITORY_ROOT / "quality_control"
    qc_dir.mkdir(parents=True, exist_ok=True)

    cohort_pre = cohort_dir / "primary_cohort_preoutcome.parquet"
    cohort_outcome = cohort_dir / "primary_cohort_with_outcome.parquet"
    cohort_manifest = cohort_dir / "primary_cohort_manifest.json"
    target_path = phenotype_dir / "aki_target_preoutcome.parquet"
    for path in [cohort_pre, cohort_outcome, cohort_manifest, target_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    pre = pd.read_parquet(cohort_pre)
    outcome = pd.read_parquet(cohort_outcome)
    target = pd.read_parquet(target_path)
    manifest = json.loads(cohort_manifest.read_text(encoding="utf-8"))

    if len(pre) != 3457 or pre["subject_id"].nunique() != 3457 or pre["stay_id"].nunique() != 3457:
        raise AssertionError("Primary pre-outcome cohort must contain 3,457 unique subjects and stays.")
    if _outcome_like(pre.columns):
        raise AssertionError(f"Primary pre-outcome cohort contains outcome-like fields: {_outcome_like(pre.columns)}")
    if len(outcome) != 3457 or int(pd.to_numeric(outcome["hospital_expire_flag"], errors="raise").sum()) != 707:
        raise AssertionError("Primary outcome cohort must contain 3,457 subjects and 707 deaths.")
    if _identity(pre) != _identity(outcome):
        raise AssertionError("Outcome attachment changed primary cohort membership.")
    if len(target) != 10083 or target["stay_id"].nunique() != 10083:
        raise AssertionError("AKI target must contain 10,083 unique renal-eligible stays.")
    required_target = {
        "creatinine_stage", "urine_output_stage", "active_rrt_any_by_landmark",
        "aki_positive", "kdigo_stage", "urine_output_ascertainable",
    }
    missing = sorted(required_target - set(target.columns))
    if missing:
        raise AssertionError(f"AKI target is missing required scientific fields: {missing}")
    stage = pd.to_numeric(pre["kdigo_stage"], errors="raise")
    if not set(stage.unique()).issubset({1, 2, 3, 1.0, 2.0, 3.0}):
        raise AssertionError("Primary cohort contains an invalid KDIGO stage.")
    if manifest.get("status") != "PASS" or manifest.get("primary_n") != 3457 or manifest.get("events") != 707:
        raise AssertionError("Primary cohort manifest does not match the reproduced cohort.")

    population_rows = []
    primary_population = None
    for filename, expected_n in EXPECTED_COUNTS.items():
        path = population_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if len(frame) != expected_n or frame["subject_id"].nunique() != expected_n or frame["stay_id"].nunique() != expected_n:
            raise AssertionError(f"{filename}: expected {expected_n} unique subjects/stays, found {len(frame)} rows.")
        if _outcome_like(frame.columns):
            raise AssertionError(f"{filename} contains outcome-like fields before model fitting.")
        population_rows.append({"population": path.stem, "subjects": expected_n})
        if filename == "01_primary.parquet":
            primary_population = frame

    if primary_population is None or _identity(primary_population) != _identity(pre):
        raise AssertionError("The primary sensitivity population does not reproduce primary cohort membership.")

    verification = {
        "status": "PASS",
        "primary_n": 3457,
        "events": 707,
        "aki_target_stays": 10083,
        "phenotype_population_count": 12,
        "consecutive_urine_output_population_n": 2972,
        "outcome_blind_preoutcome_inputs": True,
    }
    (qc_dir / "semantic_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    pd.DataFrame(population_rows).to_csv(qc_dir / "phenotype_population_counts.csv", index=False)
    print("PHENOTYPE INTEGRITY: PASS")


if __name__ == "__main__":
    main()
