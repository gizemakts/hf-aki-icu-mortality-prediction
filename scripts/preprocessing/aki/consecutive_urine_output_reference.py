#!/usr/bin/env python3
"""13 Consecutive Urine-Output Reference.

Verify the outcome-blind consecutive-urine-output reference phenotype, persist
the reference cohort, and attach hospital outcome only after the pre-outcome
membership has passed all guards.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from helpers._runtime import RAW_DATA_ROOT as _RAW_DATA_ROOT
from helpers._runtime import display_table

EXPECTED_N = 2972
EXPECTED_EVENTS = 677
EXPECTED_TARGET_N = 10083


def _outcome_like_columns(df: pd.DataFrame) -> list[str]:
    tokens = ("expire", "mortality", "death", "outcome", "deathtime", "hospital_expire")
    return [c for c in df.columns if any(token in c.lower() for token in tokens)]


def _assert_outcome_blind(df: pd.DataFrame, label: str) -> None:
    bad = _outcome_like_columns(df)
    if bad:
        raise AssertionError(f"{label} contains outcome-like columns: {bad}")


def main() -> None:
    os.environ.setdefault("HF_AKI_PROJECT_ROOT", str(_REPOSITORY_ROOT))
    os.environ.setdefault("MIMIC_ROOT", str(_RAW_DATA_ROOT))

    project_root = Path(os.environ["HF_AKI_PROJECT_ROOT"]).resolve()
    phenotype_dir = project_root / "outputs" / "02_phenotype"
    cohort_dir = project_root / "outputs" / "01_cohort"
    source_dir = phenotype_dir / "consecutive_urine_output"
    out_dir = phenotype_dir / "consecutive_urine_output_reference"
    qc_dir = project_root / "quality_control" / "phenotype" / "consecutive_urine_output_reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    source_target_path = source_dir / "consecutive_urine_output_target.parquet"
    source_index_path = source_dir / "consecutive_urine_output_cohort_preoutcome.parquet"
    source_manifest_path = source_dir / "consecutive_urine_output_manifest.json"
    landmark_path = cohort_dir / "landmark_population.parquet"
    for path in (source_target_path, source_index_path, source_manifest_path, landmark_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "PASS" or source_manifest.get("outcome_blind") is not True:
        raise AssertionError("Consecutive-urine-output phenotype did not pass its outcome-blind manifest guard.")
    if int(source_manifest.get("consecutive_urine_output_cohort_n", -1)) != EXPECTED_N:
        raise AssertionError("Consecutive-urine-output manifest cohort N mismatch.")

    target = pd.read_parquet(source_target_path)
    index = pd.read_parquet(source_index_path)
    _assert_outcome_blind(target, "consecutive-urine-output target")
    _assert_outcome_blind(index, "consecutive-urine-output cohort")

    if len(target) != EXPECTED_TARGET_N:
        raise AssertionError(f"Target N={len(target)}, expected {EXPECTED_TARGET_N}.")
    if len(index) != EXPECTED_N:
        raise AssertionError(f"Cohort N={len(index)}, expected {EXPECTED_N}.")
    if index.subject_id.isna().any() or index.stay_id.isna().any():
        raise AssertionError("Null subject_id/stay_id in reference cohort.")
    if index.subject_id.duplicated().any() or index.stay_id.duplicated().any():
        raise AssertionError("Reference cohort is not unique by subject and stay.")
    if target.stay_id.duplicated().any():
        raise AssertionError("Reference target is not unique by stay.")

    cr = pd.to_numeric(target["kdigo_creatinine_stage_strict_outpatient"], errors="coerce")
    uo = pd.to_numeric(target["urine_output_stage_consecutive"], errors="coerce")
    rrt = pd.to_numeric(target["active_rrt_any_by_landmark"], errors="coerce").fillna(0).eq(1)
    expected_positive = cr.fillna(0).ge(1) | uo.fillna(0).ge(1) | rrt
    expected_stage = pd.concat([cr.fillna(-1), uo.fillna(-1)], axis=1).max(axis=1)
    expected_stage = expected_stage.where(expected_stage.ge(1), np.nan).mask(rrt, 3.0)

    saved_positive = target["aki_positive_consecutive_urine_output"].fillna(False).astype(bool)
    saved_stage = pd.to_numeric(target["kdigo_stage_consecutive_urine_output"], errors="coerce")
    positive_mismatch = int((saved_positive != expected_positive).sum())
    stage_equal = np.isclose(
        saved_stage.fillna(-999).to_numpy(dtype=float),
        expected_stage.fillna(-999).to_numpy(dtype=float),
        atol=1e-12,
        rtol=0,
    )
    stage_mismatch = int((~stage_equal).sum())
    if positive_mismatch:
        raise AssertionError(f"Positive flag mismatch: {positive_mismatch}")
    if stage_mismatch:
        raise AssertionError(f"KDIGO stage mismatch: {stage_mismatch}")

    reselected = (
        target.loc[expected_positive]
        .sort_values(["subject_id", "intime", "stay_id"], kind="mergesort")
        .drop_duplicates("subject_id", keep="first")
    )
    expected_membership = index[["subject_id", "stay_id"]].sort_values(["subject_id", "stay_id"]).reset_index(drop=True)
    reselected_membership = reselected[["subject_id", "stay_id"]].sort_values(["subject_id", "stay_id"]).reset_index(drop=True)
    if not expected_membership.equals(reselected_membership):
        mismatch = expected_membership.merge(reselected_membership, how="outer", indicator=True)
        mismatch.to_csv(qc_dir / "first_positive_membership_mismatch.csv", index=False)
        raise AssertionError("First-positive reselection does not reproduce the reference membership.")

    hard_guard = pd.DataFrame([{
        "cohort_n": len(index),
        "target_stays": len(target),
        "positive_flag_mismatch": positive_mismatch,
        "kdigo_stage_mismatch": stage_mismatch,
        "outcome_like_columns_index": len(_outcome_like_columns(index)),
        "outcome_like_columns_target": len(_outcome_like_columns(target)),
        "duplicate_subject_id_index": int(index.subject_id.duplicated().sum()),
        "duplicate_stay_id_index": int(index.stay_id.duplicated().sum()),
    }])
    display_table(hard_guard)
    hard_guard.to_csv(qc_dir / "reference_hard_guards.csv", index=False)

    pre_target_path = out_dir / "reference_target.parquet"
    pre_index_path = out_dir / "reference_cohort_preoutcome.parquet"
    target.to_parquet(pre_target_path, index=False)
    index.to_parquet(pre_index_path, index=False)

    persisted_index = pd.read_parquet(pre_index_path)
    _assert_outcome_blind(persisted_index, "persisted reference cohort")
    if len(persisted_index) != EXPECTED_N:
        raise AssertionError("Persisted reference cohort N mismatch.")
    persisted_membership = persisted_index[["subject_id", "stay_id"]].sort_values(["subject_id", "stay_id"]).reset_index(drop=True)
    if not persisted_membership.equals(expected_membership):
        raise AssertionError("Reference membership changed during persistence.")

    landmark = pd.read_parquet(landmark_path)
    outcome_columns = ["stay_id", "outtime", "dischtime", "deathtime", "hospital_expire_flag", "gender", "age"]
    missing = [c for c in outcome_columns if c not in landmark.columns]
    if missing:
        raise KeyError(f"Landmark table missing required outcome/descriptor fields: {missing}")
    if landmark.stay_id.duplicated().any():
        raise AssertionError("Landmark table is not unique by stay_id.")

    cohort_with_outcome = persisted_index.merge(landmark[outcome_columns], on="stay_id", how="left", validate="1:1")
    flag = pd.to_numeric(cohort_with_outcome.hospital_expire_flag, errors="coerce")
    if flag.isna().any() or not flag.isin([0, 1]).all():
        raise AssertionError("Invalid hospital_expire_flag after reference outcome attachment.")
    cohort_with_outcome["hospital_expire_flag"] = flag.astype("int8")
    death_time = pd.to_datetime(cohort_with_outcome.deathtime, errors="coerce")
    landmark_time = pd.to_datetime(cohort_with_outcome.landmark_time, errors="coerce")
    death_at_or_before_landmark = death_time.notna() & landmark_time.notna() & death_time.le(landmark_time)
    if death_at_or_before_landmark.any():
        cohort_with_outcome.loc[death_at_or_before_landmark].to_csv(qc_dir / "death_at_or_before_landmark.csv", index=False)
        raise AssertionError("Selected reference patients include death at/before landmark.")

    deaths = int(cohort_with_outcome.hospital_expire_flag.sum())
    if deaths != EXPECTED_EVENTS:
        raise AssertionError(f"Reference event count={deaths}, expected {EXPECTED_EVENTS}.")
    mortality = deaths / len(cohort_with_outcome)
    outcome_path = out_dir / "reference_cohort_with_outcome.parquet"
    cohort_with_outcome.to_parquet(outcome_path, index=False)

    outcome_summary = pd.DataFrame([{
        "n": len(cohort_with_outcome),
        "hospital_deaths": deaths,
        "hospital_mortality": mortality,
        "death_at_or_before_landmark": int(death_at_or_before_landmark.sum()),
    }])
    display_table(outcome_summary)
    outcome_summary.to_csv(qc_dir / "reference_outcome_summary.csv", index=False)

    manifest = {
        "status": "PASS",
        "outcome_blind_construction": True,
        "cohort_n": int(len(persisted_index)),
        "hospital_deaths_after_attachment": deaths,
        "hospital_mortality_after_attachment": float(mortality),
        "hard_guards": {
            "outcome_like_columns_in_preoutcome_zero": True,
            "one_selected_icu_stay_per_subject": True,
            "combined_stage_exact_reproduction": True,
            "first_positive_membership_exact_reproduction": True,
            "death_at_or_before_landmark_zero": True,
        },
        "definition": {
            "landmark": "+24 h ICU",
            "heart_failure": "prior completed-admission direct heart-failure documentation",
            "preexisting_kidney_state_exclusion": "temporal chronic kidney-replacement-therapy state with post-transplant handling",
            "creatinine": "maximum of rolling KDIGO stage and strict outpatient-proxy historical median creatinine stage",
            "urine_output": "fixed6, ideal-body-weight-normalized persistent/consecutive-hour urine-output interpretation",
            "urine_output_reliability": "ureteral-stent or GU-irrigation evidence by landmark makes urine-output criterion unevaluable",
            "rrt": "active acute RRT by landmark contributes KDIGO Stage 3 after chronic-state exclusion",
            "episode_selection": "first qualifying positive ICU stay per subject",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "reference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("REFERENCE PHENOTYPE: PASS")
    print("N:", len(persisted_index))
    print("Deaths after outcome attachment:", deaths)


if __name__ == "__main__":
    main()
