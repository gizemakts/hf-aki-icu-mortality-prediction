#!/usr/bin/env python3
"""Validate the outcome-blind reference cohort before outcome attachment."""
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
PRIMARY_UO_VARIANT = "event_gap6_full"


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def outcome_like(columns) -> list[str]:
    tokens = ("death", "mortality", "expire", "outcome", "deathtime", "hospital_expire", "survival")
    return [c for c in columns if any(t in c.lower() for t in tokens)]


def main() -> None:
    root = Path(os.environ.get("HF_AKI_PROJECT_ROOT", str(_REPOSITORY_ROOT))).resolve()
    cohort_dir = root / "outputs" / "01_cohort"
    phenotype_dir = root / "outputs" / "02_phenotype"
    validation_dir = root / "outputs" / "05_validation" / "reference_cohort"
    qc_dir = root / "quality_control" / "cohort"
    validation_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    index_path = require_file(cohort_dir / "reference_cohort_preoutcome.parquet")
    combined_path = require_file(phenotype_dir / "combined" / "combined_kdigo_variants.parquet")
    uo_path = require_file(phenotype_dir / "urine_output" / "urine_output_event_time_variants.parquet")
    source_definition_path = require_file(root / "quality_control" / "phenotype" / "urine_output_source_definition.json")
    creatinine_manifest_path = require_file(root / "quality_control" / "phenotype" / "creatinine_verification_manifest.json")

    idx = pd.read_parquet(index_path)
    combined = pd.read_parquet(combined_path)
    uo = pd.read_parquet(uo_path)
    source_definition = json.loads(source_definition_path.read_text(encoding="utf-8"))
    creatinine_manifest = json.loads(creatinine_manifest_path.read_text(encoding="utf-8"))

    bad = outcome_like(idx.columns)
    if bad:
        raise AssertionError(f"Reference cohort contains outcome-like fields: {bad}")
    if len(idx) != EXPECTED_REFERENCE_N:
        raise AssertionError(f"Reference cohort N={len(idx)}, expected {EXPECTED_REFERENCE_N}")
    if idx["subject_id"].duplicated().any() or idx["stay_id"].duplicated().any():
        raise AssertionError("Reference cohort must be unique by subject_id and stay_id")
    if not idx["prior_documented_hf"].eq(1).all():
        raise AssertionError("Reference cohort contains stays without prior documented HF")
    if int(idx["chronic_krt_definite"].fillna(0).sum()) != 0:
        raise AssertionError("Reference cohort contains definite chronic KRT")
    if int(idx["current_chronic_krt_state_code"].fillna(0).sum()) != 0:
        raise AssertionError("Reference cohort contains current chronic-KRT state coding")
    if not idx["combined_status_ibw_anyrrt"].eq("positive").all():
        raise AssertionError("Reference cohort contains non-positive combined KDIGO status")
    if not pd.to_numeric(idx["combined_stage_ibw_anyrrt"], errors="coerce").between(1, 3).all():
        raise AssertionError("Reference cohort contains invalid KDIGO stage")
    if not idx["variant_id"].eq(PRIMARY_UO_VARIANT).all():
        raise AssertionError("Reference cohort contains a non-primary urine-output reconstruction")

    if "primary_direct_itemids" not in source_definition:
        raise AssertionError("Urine-output source definition is missing primary_direct_itemids")
    defined_items = set(map(int, source_definition["primary_direct_itemids"]))
    required_direct = {226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567, 226557, 226558}
    if "gu_irrigation_itemids" not in source_definition:
        raise AssertionError("Urine-output source definition is missing gu_irrigation_itemids")
    forbidden_primary = set(map(int, source_definition["gu_irrigation_itemids"]))
    if defined_items != required_direct:
        raise AssertionError(f"Primary urine-output source set differs from the prespecified set: {sorted(defined_items)}")
    if defined_items & forbidden_primary:
        raise AssertionError("GU irrigation appears in the primary urine-output source set")
    if source_definition.get("gu_irrigation_in_primary") is not False:
        raise AssertionError("Urine-output source definition does not explicitly exclude GU irrigation")
    creatinine_verified = (
        creatinine_manifest.get("status") == "PASS"
        and creatinine_manifest.get("outcome_blind") is True
    )
    if not creatinine_verified:
        raise AssertionError("Creatinine component verification did not pass")

    primary_uo = uo.loc[uo["variant_id"].eq(PRIMARY_UO_VARIANT)].copy()
    observability = {
        "full6_n": int(primary_uo.get("any_full6_window", pd.Series(dtype=float)).fillna(0).sum()),
        "full12_n": int(primary_uo.get("any_full12_window", pd.Series(dtype=float)).fillna(0).sum()),
        "full24_n": int(primary_uo.get("any_full24_window", pd.Series(dtype=float)).fillna(0).sum()),
    }
    if "max_coverage24_hours" in primary_uo:
        observability["median_max_coverage24_h"] = float(pd.to_numeric(primary_uo["max_coverage24_hours"], errors="coerce").median())

    gates = pd.DataFrame([
        {"check": "outcome_blind", "passed": True},
        {"check": "expected_reference_n", "passed": len(idx) == EXPECTED_REFERENCE_N},
        {"check": "unique_subjects", "passed": not idx["subject_id"].duplicated().any()},
        {"check": "unique_stays", "passed": not idx["stay_id"].duplicated().any()},
        {"check": "prior_hf_required", "passed": bool(idx["prior_documented_hf"].eq(1).all())},
        {"check": "chronic_krt_excluded", "passed": int(idx["chronic_krt_definite"].fillna(0).sum()) == 0},
        {"check": "primary_uo_reconstruction", "passed": bool(idx["variant_id"].eq(PRIMARY_UO_VARIANT).all())},
        {"check": "direct_uo_sources", "passed": defined_items == required_direct},
        {"check": "gu_irrigation_excluded", "passed": not bool(defined_items & forbidden_primary)},
        {"check": "creatinine_component_verified", "passed": creatinine_verified},
    ])
    if not gates["passed"].all():
        raise AssertionError("Reference-cohort pre-outcome integrity check failed")
    gates.to_csv(qc_dir / "reference_cohort_preoutcome_checks.csv", index=False)

    manifest = {
        "status": "PASS",
        "outcome_blind": True,
        "reference_cohort_n": int(len(idx)),
        "primary_uo_variant": PRIMARY_UO_VARIANT,
        "uo_observability": observability,
        "primary_uo_itemids": sorted(defined_items),
        "gu_irrigation_excluded": True,
    }
    (validation_dir / "preoutcome_integrity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("REFERENCE COHORT PRE-OUTCOME INTEGRITY: PASS")


if __name__ == "__main__":
    main()
