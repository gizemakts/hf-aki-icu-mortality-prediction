#!/usr/bin/env python3
"""Validate phenotype and provenance inputs before predictor extraction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def main() -> None:
    cohort_path = _REPOSITORY_ROOT / "outputs" / "01_cohort" / "primary_cohort_preoutcome.parquet"
    cohort_manifest_path = _REPOSITORY_ROOT / "outputs" / "01_cohort" / "primary_cohort_manifest.json"
    phenotype_check_path = _REPOSITORY_ROOT / "quality_control" / "semantic_verification.json"
    provenance_path = _REPOSITORY_ROOT / "quality_control" / "raw_data_provenance.json"
    external_path = _REPOSITORY_ROOT / "quality_control" / "external_validation_status.json"
    expanded_hf_path = _REPOSITORY_ROOT / "outputs" / "06_sensitivity" / "expanded_heart_failure" / "expanded_heart_failure_manifest.json"
    for path in (cohort_path, cohort_manifest_path, phenotype_check_path, provenance_path, external_path, expanded_hf_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    cohort = pd.read_parquet(cohort_path, columns=["subject_id", "stay_id"])
    cohort_manifest = json.loads(cohort_manifest_path.read_text(encoding="utf-8"))
    phenotype_check = json.loads(phenotype_check_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    external = json.loads(external_path.read_text(encoding="utf-8"))
    expanded_hf = json.loads(expanded_hf_path.read_text(encoding="utf-8"))

    if len(cohort) != 3457 or cohort["subject_id"].nunique() != 3457 or cohort["stay_id"].nunique() != 3457:
        raise AssertionError("Primary cohort must contain 3,457 unique subjects and stays.")
    if cohort_manifest.get("status") != "PASS" or cohort_manifest.get("primary_n") != 3457:
        raise AssertionError("Primary cohort manifest is not valid.")
    if phenotype_check.get("status") != "PASS" or phenotype_check.get("phenotype_population_count") != 12:
        raise AssertionError("Phenotype integrity checks are incomplete.")
    if provenance.get("status") != "PASS" or provenance.get("raw_dependency_count") != 14:
        raise AssertionError("Raw data provenance is incomplete.")
    if external.get("status") not in {"DATA_UNAVAILABLE", "FEASIBLE", "BLOCKED"}:
        raise AssertionError("External-validation status is not valid.")
    if external.get("status") == "BLOCKED":
        raise AssertionError("The configured eICU feasibility branch is blocked and must be resolved.")
    if external.get("status") == "FEASIBLE":
        if external.get("eicu_data_available") is not True or external.get("feasibility_assessment_completed") is not True:
            raise AssertionError("The eICU-available branch did not complete its feasibility contract.")
        if external.get("external_validation_performed") is not False:
            raise AssertionError("Feasibility must not be represented as completed external validation.")
    if external.get("status") == "DATA_UNAVAILABLE" and external.get("eicu_data_available") is not False:
        raise AssertionError("The eICU-unavailable branch has an inconsistent status contract.")
    if expanded_hf.get("status") != "PASS" or expanded_hf.get("outcome_blind") is not True:
        raise AssertionError("Expanded-HF measurement/QC sensitivity did not complete outcome-blindly.")

    qc_dir = _REPOSITORY_ROOT / "quality_control"
    checks = pd.DataFrame(
        [
            {"check": "primary_cohort", "status": "PASS"},
            {"check": "phenotype_integrity", "status": "PASS"},
            {"check": "raw_data_provenance", "status": "PASS"},
            {"check": "external_validation_status", "status": str(external.get("status"))},
            {"check": "expanded_hf_measurement_qc", "status": "PASS"},
        ]
    )
    checks.to_csv(qc_dir / "model_input_validation.csv", index=False)
    manifest = {
        "status": "PASS",
        "primary_n": 3457,
        "phenotype_populations": 12,
        "predictor_extraction_allowed": True,
        "historical_generated_hash_gate_used": False,
        "external_validation_branch_status": str(external.get("status")),
        "expanded_hf_role": "outcome-blind measurement/QC sensitivity; not an additional modeled population",
    }
    (qc_dir / "model_input_validation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("MODEL INPUT VALIDATION: PASS")


if __name__ == "__main__":
    main()
