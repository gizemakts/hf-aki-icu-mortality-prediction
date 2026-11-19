#!/usr/bin/env python3
"""Verify cohort, model, validation, and sensitivity analysis invariants.

Checks require the authorized scientific-invariant contract and the corresponding
study outputs. Each reported invariant is checked against its accepted value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def membership_sha(df: pd.DataFrame) -> str:
    ids = df[["subject_id", "stay_id"]].copy()
    ids["subject_id"] = pd.to_numeric(ids["subject_id"], errors="raise").astype("int64")
    ids["stay_id"] = pd.to_numeric(ids["stay_id"], errors="raise").astype("int64")
    ids = ids.sort_values(["subject_id", "stay_id"]).reset_index(drop=True)
    payload = "subject_id,stay_id\n" + "".join(
        f"{int(r.subject_id)},{int(r.stay_id)}\n" for r in ids.itertuples(index=False)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify cross-analysis scientific invariants")
    parser.add_argument(
        "--invariants", type=Path,
        default=Path("config/accepted_scientific_invariants.json"),
        help="Path to the authorized scientific-invariant contract (JSON)",
    )
    args = parser.parse_args(argv)
    root = _REPOSITORY_ROOT
    invariant_contract_path = args.invariants
    if not invariant_contract_path.is_absolute():
        invariant_contract_path = root / invariant_contract_path
    if not invariant_contract_path.is_file():
        parser.error(
            "Authorized scientific-invariant contract is required: "
            f"{invariant_contract_path}. Provide it with --invariants."
        )
    accepted = json.loads(invariant_contract_path.read_text(encoding="utf-8"))
    cohort_dir = root / "outputs" / "01_cohort"
    model_dir = root / "outputs" / "04_modeling"
    validation_dir = root / "outputs" / "05_validation" / "internal_validation"
    sensitivity_dir = root / "outputs" / "06_sensitivity" / "phenotype"
    uf_dir = root / "outputs" / "06_sensitivity" / "isolated_uf"
    out = root / "outputs" / "05_validation" / "scientific_integrity"
    out.mkdir(parents=True, exist_ok=True)

    cohort_path = require(cohort_dir / "primary_cohort_preoutcome.parquet")
    cohort_summary_path = require(cohort_dir / "cohort_summary.json")
    model_manifest_path = require(model_dir / "model_fit_manifest.json")
    validation_manifest_path = require(validation_dir / "internal_validation_manifest.json")
    validation_summary_path = require(validation_dir / "internal_validation_summary.csv")
    sensitivity_manifest_path = require(sensitivity_dir / "phenotype_sensitivity_manifest.json")
    sensitivity_overview_path = require(sensitivity_dir / "sensitivity_arm_overview.csv")
    uf_manifest_path = require(uf_dir / "isolated_uf_sensitivity_manifest.json")
    uf_affected_path = require(uf_dir / "affected_stays.csv")

    cohort = pd.read_parquet(cohort_path)
    cohort_summary = json.loads(cohort_summary_path.read_text(encoding="utf-8"))
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    sensitivity_manifest = json.loads(sensitivity_manifest_path.read_text(encoding="utf-8"))
    uf_manifest = json.loads(uf_manifest_path.read_text(encoding="utf-8"))
    validation_summary = pd.read_csv(validation_summary_path)
    sensitivity_overview = pd.read_csv(sensitivity_overview_path)
    uf_affected = pd.read_csv(uf_affected_path)

    adjudication_dir = root / "outputs" / "02_phenotype" / "adjudication"
    variant_summary = pd.read_csv(require(adjudication_dir / "adjudication_variant_summary.csv"))
    elapsed_stage = pd.read_csv(require(adjudication_dir / "elapsed_duration_stage_effect.csv"))
    elapsed_membership = pd.read_csv(require(adjudication_dir / "elapsed_duration_membership_effect.csv"))
    union_stage = pd.read_csv(require(adjudication_dir / "interval_union_stage_effect.csv"))
    creatinine_aggregation = pd.read_csv(require(adjudication_dir / "creatinine_aggregation_effect.csv"))
    transplant = pd.read_csv(require(adjudication_dir / "transplant_override_summary.csv"))
    rrt_policy = pd.read_csv(require(adjudication_dir / "rrt_policy_regression_summary.csv")).iloc[0]
    expanded_hf_summary = pd.read_csv(require(
        root / "quality_control" / "sensitivity" / "expanded_heart_failure" / "expanded_heart_failure_summary.csv"
    ))
    primary_population = pd.read_parquet(require(root / "outputs" / "06_sensitivity" / "populations" / "01_primary.parquet"), columns=["subject_id", "stay_id"])
    hf_wide_population = pd.read_parquet(require(root / "outputs" / "06_sensitivity" / "populations" / "03_hf_wide.parquet"), columns=["subject_id", "stay_id"])

    accepted_checks: list[dict[str, object]] = []

    def check(name: str, observed: object, expected: object) -> None:
        passed = observed == expected
        accepted_checks.append({"check": name, "expected": expected, "observed": observed, "pass": passed})
        if not passed:
            raise AssertionError(f"Accepted scientific invariant failed: {name}: {observed!r} != {expected!r}")

    expected_n = int(accepted["primary_n"])
    expected_events = int(accepted["primary_events"])
    expected_arms = int(accepted["modeled_populations"])
    expected_alternative_refits = int(accepted["alternative_complete_refits"])
    expected_bootstrap = int(accepted["bootstrap_replicates"])
    expected_uf_stays = int(accepted["isolated_uf_affected_stays"])

    if len(cohort) != expected_n or cohort["stay_id"].nunique() != expected_n:
        raise AssertionError("Primary cohort must contain 3,457 unique stays.")
    if cohort["subject_id"].nunique() != expected_n:
        raise AssertionError("Primary cohort must contain one selected stay per subject.")

    if int(cohort_summary.get("primary_n", cohort_summary.get("n", -1))) != expected_n:
        raise AssertionError("Cohort summary N does not match 3,457.")
    if int(cohort_summary.get("deaths", cohort_summary.get("events", -1))) != expected_events:
        raise AssertionError("Cohort summary event count does not match 707.")

    if model_manifest.get("status") != "PASS":
        raise AssertionError("Model fit did not report PASS.")
    if int(model_manifest.get("n", -1)) != expected_n or int(model_manifest.get("events", -1)) != expected_events:
        raise AssertionError("Model-fit cohort identity disagrees with the primary cohort.")

    if validation_manifest.get("status") != "PASS":
        raise AssertionError("Internal validation did not report PASS.")
    if int(validation_manifest.get("n", -1)) != expected_n or int(validation_manifest.get("events", -1)) != expected_events:
        raise AssertionError("Internal-validation cohort identity disagrees with the primary cohort.")
    if int(validation_manifest.get("bootstrap_resamples_primary", -1)) != expected_bootstrap:
        raise AssertionError("Primary internal validation must contain 500 bootstrap resamples.")
    if int(validation_manifest.get("bootstrap_resamples_secondary", -1)) != expected_bootstrap:
        raise AssertionError("Secondary internal validation must contain 500 bootstrap resamples.")

    if sensitivity_manifest.get("status") != "PASS":
        raise AssertionError("Phenotype sensitivity analysis did not report PASS.")
    if int(sensitivity_manifest.get("arm_count", -1)) != expected_arms:
        raise AssertionError("Phenotype sensitivity analysis must contain 12 populations.")
    if int(sensitivity_manifest.get("alternative_arms_completely_refit", -1)) != expected_alternative_refits:
        raise AssertionError("Exactly 11 alternative phenotype populations must receive complete refits.")
    if int(sensitivity_manifest.get("bootstrap_resamples_each", -1)) != expected_bootstrap:
        raise AssertionError("Each phenotype sensitivity refit must use 500 bootstrap resamples.")
    if len(sensitivity_overview) != expected_arms:
        raise AssertionError("Phenotype sensitivity overview must contain 12 rows.")

    if uf_manifest.get("status") != "PASS":
        raise AssertionError("Isolated-UF sensitivity analysis did not report PASS.")
    if not 0 < expected_uf_stays < 11:
        raise AssertionError("Restricted isolated-UF count violates the public small-cell contract.")
    for key, expected_value in (("source_verified_affected_stays", expected_uf_stays),
                                ("selected_kdigo_stage_changes", int(accepted["isolated_uf_stage_changes"]))):
        value = uf_manifest.get(key)
        # Both numeric restricted history and publication-masked source are supported.
        if value != "<11" and value != expected_value:
            raise AssertionError(f"Isolated-UF manifest value disagrees with restricted invariant for {key}.")
    uf_identity_changes = int(bool(uf_manifest.get("membership_changed", True))) + int(bool(uf_manifest.get("timezero_changed", True)))
    if uf_identity_changes != int(accepted["isolated_uf_membership_or_timezero_changes"]):
        raise AssertionError("Isolated-UF suppression must not change cohort membership or time zero.")
    if len(uf_affected) != expected_uf_stays:
        raise AssertionError("Isolated-UF affected-stay table differs from restricted invariant.")

    check("primary_n", len(cohort), accepted["primary_n"])
    check("primary_events", int(cohort_summary.get("events", -1)), accepted["primary_events"])
    check("primary_kdigo_stage_counts", cohort_summary.get("kdigo_stage_distribution"), accepted["primary_kdigo_stage_counts"])

    unguarded = variant_summary.loc[variant_summary["variant"].eq("calendar_hour_mean_ibw_post_transplant")]
    check("unguarded_mean_uo_n", int(unguarded.iloc[0]["subjects"]), accepted["unguarded_mean_uo_n"])
    elapsed_mean = elapsed_stage.loc[elapsed_stage["comparison"].eq("cal_mean_ibw -> elapsed_mean_ibw")].iloc[0]
    check("elapsed_duration_target_stage_changes", int(elapsed_mean["target_stage_changed"]), accepted["elapsed_duration_target_stage_changes"])
    check("elapsed_duration_positives_removed", int(elapsed_mean["positive_removed"]), accepted["elapsed_duration_positives_removed"])
    elapsed_selected = elapsed_membership.loc[elapsed_membership["variant"].eq("elapsed_guard_mean_ibw")].iloc[0]
    check("elapsed_duration_selected_subjects_dropped", int(elapsed_selected["subject_dropped_vs_reference"]), accepted["elapsed_duration_selected_subjects_dropped"])
    check("span_versus_union_stage_changes", int(union_stage["target_stage_changed"].sum()), accepted["span_versus_union_stage_changes"])
    for variant, key in (("MAX", "same_charttime_max_stage_changes"), ("MIN", "same_charttime_min_stage_changes")):
        row = creatinine_aggregation.loc[creatinine_aggregation["variant"].eq(variant)].iloc[0]
        check(key, int(row["target_stage_changed"]), accepted[key])

    primary_by_subject = primary_population.set_index("subject_id")["stay_id"]
    hf_wide_by_subject = hf_wide_population.set_index("subject_id")["stay_id"]
    common_subjects = primary_by_subject.index.intersection(hf_wide_by_subject.index)
    check("hf_wide_n", len(hf_wide_population), accepted["hf_wide_n"])
    check("hf_wide_subject_additions", len(hf_wide_by_subject.index.difference(primary_by_subject.index)), accepted["hf_wide_subject_additions"])
    check("hf_wide_replaced_stays", int((primary_by_subject.loc[common_subjects] != hf_wide_by_subject.loc[common_subjects]).sum()), accepted["hf_wide_replaced_stays"])
    hf_row = expanded_hf_summary.iloc[0]
    check("remote_prior_hf_over_3_years", int(hf_row["primary_selected_latest_prior_hf_gt3y_n"]), accepted["remote_prior_hf_over_3_years"])
    check("remote_prior_hf_over_5_years", int(hf_row["primary_selected_latest_prior_hf_gt5y_n"]), accepted["remote_prior_hf_over_5_years"])

    for key in ("active_rrt_target_stays", "direct_scuf_target_stays", "itemid_226499_target_stays", "unknown_weak_rrt_patterns"):
        check(key, int(rrt_policy[key]), accepted[key])
    check("post_transplant_hard_contradictions", int(transplant.iloc[0]["hard_contradictions"]), accepted["post_transplant_hard_contradictions"])

    metrics = {"auroc", "brier", "calibration_intercept", "calibration_slope", "observed_expected_ratio"}
    if set(validation_summary["metric"].astype(str)) != metrics:
        raise AssertionError("Internal-validation summary has an unexpected metric set.")

    current_membership_sha = membership_sha(cohort)

    completion = pd.DataFrame([
        {"analysis": "primary_cohort", "status": "PASS", "subjects": expected_n, "events": expected_events},
        {"analysis": "model_fit", "status": "PASS", "subjects": expected_n, "events": expected_events},
        {"analysis": "internal_validation", "status": "PASS", "subjects": expected_n, "events": expected_events},
        {"analysis": "phenotype_sensitivity", "status": "PASS", "subjects": expected_n, "events": expected_events},
        {"analysis": "isolated_uf_sensitivity", "status": "PASS", "subjects": expected_n, "events": expected_events},
    ])
    completion.to_csv(out / "model_completion_summary.csv", index=False)
    validation_summary.to_csv(out / "primary_secondary_internal_validation.csv", index=False)
    sensitivity_overview.to_csv(out / "phenotype_sensitivity_overview.csv", index=False)
    uf_affected.to_csv(out / "isolated_uf_affected_stays.csv", index=False)
    pd.DataFrame(accepted_checks).to_csv(out / "accepted_scientific_invariant_checks.csv", index=False)

    manifest = {
        "status": "PASS",
        "primary_n": expected_n,
        "primary_events": expected_events,
        "primary_membership_sha256": current_membership_sha,
        "phenotype_sensitivity_arms": expected_arms,
        "alternative_complete_refits": expected_alternative_refits,
        "bootstrap_resamples": expected_bootstrap,
        "isolated_uf_affected_stays": expected_uf_stays,
        "accepted_scientific_invariant_contract": invariant_contract_path.name,
        "accepted_scientific_invariant_contract_sha256": hashlib.sha256(
            invariant_contract_path.read_bytes()
        ).hexdigest(),
        "accepted_scientific_invariant_checks": len(accepted_checks),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "scientific_integrity_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("SCIENTIFIC INTEGRITY: PASS")


if __name__ == "__main__":
    main()
