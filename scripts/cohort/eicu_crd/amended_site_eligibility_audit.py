#!/usr/bin/env python3
"""
Outcome-blind eICU amended site-eligibility audit.

Purpose
-------
Apply the PRE-OUTCOME amended site rule:

SOURCE-COMPLETE
AND
EXACT-ROUTINE-PREDICTOR-TARGET-COMPLETE

This script deterministically derives the amended hospital set and its
adult +24 h / fixed-HF +24 h audit-stage consequences.

It does NOT:
- access outcomes or discharge status/location;
- fit/evaluate a model;
- perform imputation;
- apply final renal/chronic-KRT eligibility;
- perform final episode selection;
- lock final cohort membership.

Expected audit-stage consequence from prior nonbinding audit:
- 145 eligible hospitals
- 98,298 adult +24 h stays retained
- 15,034 fixed-HF +24 h stays retained
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HF_VALUES = (
    "CHF",
    "CHF - severity unknown",
    "CHF - class I",
    "CHF - class II",
    "CHF - class III",
    "CHF - class IV",
)
HF_PATH_PREFIX = (
    "notes/Progress Notes/Past History/Organ Systems/"
    "Cardiovascular (R)/Congestive Heart Failure/"
)

PROHIBITED_TOKENS = (
    "hospitaldischargestatus",
    "unitdischargestatus",
    "hospitaldischargelocation",
    "unitdischargelocation",
    "mortality",
    "death",
    "survival",
    "auroc",
    "auc",
    "brier",
    "calibration",
)

ROUTINE_TARGETS = [
    "heart_rate_max_per10",
    "map_min_per10",
    "resp_rate_max_per5",
    "temperature_max_abs_deviation_from_37c",
    "spo2_deficit_per5",
    "gcs_deficit_from15",
    "hemoglobin_min_gdl",
    "wbc_max_log2",
    "platelet_min_log2",
    "bun_max_log2",
    "bicarbonate_min_per5",
    "sodium_max_abs_deviation_from140_per5",
    "potassium_max_mmol_l",
    "glucose_max_log2",
]

LAB_SPECS = {
    "hemoglobin_min_gdl": ("Hgb", 3, 0.0, 50.0),
    "wbc_max_log2": ("WBC x 1000", 3, 0.0, 1000.0),
    "platelet_min_log2": ("platelets x 1000", 3, 0.0, 10000.0),
    "bun_max_log2": ("BUN", 1, 0.0, 300.0),
    "bicarbonate_min_per5": ("bicarbonate", 1, 0.0, 10000.0),
    "sodium_max_abs_deviation_from140_per5": ("sodium", 1, 0.0, 200.0),
    "potassium_max_mmol_l": ("potassium", 1, 0.0, 30.0),
    "glucose_max_log2": ("glucose", 1, 0.0, 10000.0),
}

EXPECTED = {
    "eligible_hospitals": 145,
    "adult_plus24h_retained": 98298,
    "hf_plus24h_retained": 15034,
}

def discover(root: Path, stem: str) -> Path:
    wanted = {f"{stem}.csv".lower(), f"{stem}.csv.gz".lower()}
    matches = sorted(
        [p.resolve() for p in root.rglob("*")
         if p.is_file() and p.name.lower() in wanted],
        key=lambda p: p.as_posix().lower(),
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"{stem}: expected exactly one {stem}.csv(.gz), found {len(matches)}"
        )
    return matches[0]

def qpath(p: Path) -> str:
    return str(p.resolve()).replace("'", "''")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_lines(values) -> str:
    payload = "\n".join(str(x) for x in sorted(values)) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def assert_safe_columns(columns, label: str) -> None:
    bad = [
        str(c) for c in columns
        if any(tok in str(c).lower() for tok in PROHIBITED_TOKENS)
    ]
    if bad:
        raise AssertionError(f"{label}: prohibited columns detected: {bad}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import duckdb
    import pandas as pd

    root = Path(args.eicu_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    PATIENT = discover(root, "patient")
    PH = discover(root, "pastHistory")
    NURSE = discover(root, "nurseCharting")
    LAB = discover(root, "lab")
    INF = discover(root, "infusionDrug")
    TREAT = discover(root, "treatment")
    IO = discover(root, "intakeOutput")
    DX = discover(root, "diagnosis")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    patient = con.execute(f"""
        SELECT
            patientUnitStayID AS stay_id,
            hospitalID AS hospital_id,
            age,
            unitDischargeOffset AS unit_discharge_offset,
            hospitalDischargeOffset AS hospital_discharge_offset
        FROM read_csv_auto(
            '{qpath(PATIENT)}',
            header=true,
            sample_size=200000
        )
    """).df()
    assert_safe_columns(patient.columns, "patient_safe")

    if patient["stay_id"].duplicated().any():
        raise AssertionError("Duplicate patientUnitStayID")

    def age_num(x):
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        if s.startswith(">"):
            return 90.0
        try:
            return float(s)
        except Exception:
            return None

    patient["age_num"] = patient["age"].map(age_num)
    unit_off = pd.to_numeric(patient["unit_discharge_offset"], errors="coerce")
    hosp_off = pd.to_numeric(patient["hospital_discharge_offset"], errors="coerce")

    adult = patient.loc[
        patient["age_num"].fillna(-1).ge(18)
        & unit_off.gt(1440)
        & hosp_off.gt(1440),
        ["stay_id", "hospital_id"],
    ].copy()

    con.register("patient_ids_df", patient[["stay_id", "hospital_id"]])
    con.execute("CREATE TEMP TABLE patient_ids AS SELECT * FROM patient_ids_df")
    con.register("adult_df", adult)
    con.execute("CREATE TEMP TABLE adult AS SELECT * FROM adult_df")

    source_sql = {
        "pastHistory": f"""
            SELECT DISTINCT p.hospital_id
            FROM read_csv_auto('{qpath(PH)}', header=true, sample_size=200000) x
            JOIN patient_ids p ON x.patientUnitStayID=p.stay_id
        """,
        "nurseCharting": f"""
            SELECT DISTINCT p.hospital_id
            FROM read_csv_auto('{qpath(NURSE)}', header=true, sample_size=200000) x
            JOIN patient_ids p ON x.patientUnitStayID=p.stay_id
        """,
        "lab": f"""
            SELECT DISTINCT p.hospital_id
            FROM read_csv_auto('{qpath(LAB)}', header=true, sample_size=200000) x
            JOIN patient_ids p ON x.patientUnitStayID=p.stay_id
        """,
        "creatinine": f"""
            SELECT DISTINCT p.hospital_id
            FROM read_csv_auto('{qpath(LAB)}', header=true, sample_size=200000) x
            JOIN patient_ids p ON x.patientUnitStayID=p.stay_id
            WHERE lower(x.labName)='creatinine'
              AND x.labTypeID=1
              AND TRY_CAST(x.labResult AS DOUBLE)>0
              AND TRY_CAST(x.labResult AS DOUBLE)<=150
        """,
        "infusionDrug": f"""
            SELECT DISTINCT p.hospital_id
            FROM read_csv_auto('{qpath(INF)}', header=true, sample_size=200000) x
            JOIN patient_ids p ON x.patientUnitStayID=p.stay_id
        """,
        "treatment": f"""
            SELECT DISTINCT p.hospital_id
            FROM read_csv_auto('{qpath(TREAT)}', header=true, sample_size=200000) x
            JOIN patient_ids p ON x.patientUnitStayID=p.stay_id
        """,
        "intakeOutput": f"""
            SELECT DISTINCT p.hospital_id
            FROM read_csv_auto('{qpath(IO)}', header=true, sample_size=200000) x
            JOIN patient_ids p ON x.patientUnitStayID=p.stay_id
        """,
        "diagnosis": f"""
            SELECT DISTINCT p.hospital_id
            FROM read_csv_auto('{qpath(DX)}', header=true, sample_size=200000) x
            JOIN patient_ids p ON x.patientUnitStayID=p.stay_id
        """,
    }

    source_sets = {
        name: set(con.execute(sql).df()["hospital_id"].unique())
        for name, sql in source_sql.items()
    }
    source_complete = set.intersection(*source_sets.values())

    nurse = con.execute(f"""
        SELECT
            p.hospital_id,
            n.nursingChartCellTypeValLabel AS vallabel,
            n.nursingChartCellTypeValName AS valname,
            TRY_CAST(n.nursingChartValue AS DOUBLE) AS v
        FROM read_csv_auto(
            '{qpath(NURSE)}',
            header=true,
            sample_size=200000
        ) n
        JOIN patient_ids p ON n.patientUnitStayID=p.stay_id
        WHERE n.nursingChartCellTypeCat IN
            ('Vital Signs','Scores','Other Vital Signs and Infusions')
    """).df()

    nurse_masks = {
        "heart_rate_max_per10": (
            nurse["vallabel"].eq("Heart Rate")
            & nurse["valname"].eq("Heart Rate")
            & nurse["v"].gt(0) & nurse["v"].lt(300)
        ),
        "map_min_per10": (
            (
                nurse["vallabel"].eq("Non-Invasive BP")
                & nurse["valname"].eq("Non-Invasive BP Mean")
            )
            | (
                nurse["vallabel"].eq("Invasive BP")
                & nurse["valname"].eq("Invasive BP Mean")
            )
            | (
                nurse["vallabel"].eq("MAP (mmHg)")
                & nurse["valname"].eq("Value")
            )
            | (
                nurse["vallabel"].eq("Arterial Line MAP (mmHg)")
                & nurse["valname"].eq("Value")
            )
        ) & nurse["v"].gt(0) & nurse["v"].lt(300),
        "resp_rate_max_per5": (
            nurse["vallabel"].eq("Respiratory Rate")
            & nurse["valname"].eq("Respiratory Rate")
            & nurse["v"].gt(0) & nurse["v"].lt(70)
        ),
        "temperature_max_abs_deviation_from_37c": (
            (
                nurse["vallabel"].eq("Temperature")
                & nurse["valname"].eq("Temperature (C)")
                & nurse["v"].gt(10) & nurse["v"].lt(50)
            )
            | (
                nurse["vallabel"].eq("Temperature")
                & nurse["valname"].eq("Temperature (F)")
                & nurse["v"].gt(70) & nurse["v"].lt(120)
            )
        ),
        "spo2_deficit_per5": (
            (
                nurse["vallabel"].eq("O2 Saturation")
                & nurse["valname"].eq("O2 Saturation")
            )
            | (
                nurse["vallabel"].eq("SpO2")
                & nurse["valname"].eq("Value")
            )
        ) & nurse["v"].gt(0) & nurse["v"].le(100),
        "gcs_deficit_from15": (
            (
                nurse["vallabel"].eq("Glasgow coma score")
                & nurse["valname"].eq("GCS Total")
            )
            | (
                nurse["vallabel"].eq("Score (Glasgow Coma Scale)")
                & nurse["valname"].eq("Value")
            )
        ) & nurse["v"].between(3, 15, inclusive="both"),
    }

    target_sets = {}
    for target, mask in nurse_masks.items():
        target_sets[target] = set(nurse.loc[mask, "hospital_id"].unique())

    lab_names = sorted({v[0] for v in LAB_SPECS.values()})
    names_sql = ",".join("'" + x.replace("'", "''") + "'" for x in lab_names)

    lab = con.execute(f"""
        SELECT
            p.hospital_id,
            l.labName AS lab_name,
            l.labTypeID AS lab_type_id,
            TRY_CAST(l.labResult AS DOUBLE) AS v
        FROM read_csv_auto(
            '{qpath(LAB)}',
            header=true,
            sample_size=200000
        ) l
        JOIN patient_ids p ON l.patientUnitStayID=p.stay_id
        WHERE l.labName IN ({names_sql})
    """).df()

    for target, (name, lab_type, lo, hi) in LAB_SPECS.items():
        mask = (
            lab["lab_name"].eq(name)
            & pd.to_numeric(lab["lab_type_id"], errors="coerce").eq(lab_type)
            & lab["v"].gt(lo)
            & lab["v"].le(hi)
        )
        target_sets[target] = set(lab.loc[mask, "hospital_id"].unique())

    target_complete = set.intersection(*[target_sets[t] for t in ROUTINE_TARGETS])
    amended_eligible = source_complete & target_complete

    values_sql = ",".join("'" + v.replace("'", "''") + "'" for v in HF_VALUES)
    hf_ids = con.execute(f"""
        SELECT DISTINCT ph.patientUnitStayID AS stay_id
        FROM read_csv_auto(
            '{qpath(PH)}',
            header=true,
            sample_size=200000
        ) ph
        JOIN adult a ON ph.patientUnitStayID=a.stay_id
        WHERE ph.pastHistoryOffset<=1440
          AND ph.pastHistoryEnteredOffset<=1440
          AND ph.pastHistoryValue IN ({values_sql})
          AND ph.pastHistoryValueText=ph.pastHistoryValue
          AND ph.pastHistoryPath=
              '{HF_PATH_PREFIX.replace("'", "''")}' || ph.pastHistoryValue
    """).df()

    hf = adult.merge(hf_ids.assign(hf_proxy=1), on="stay_id", how="inner")

    adult_retained = adult.loc[adult["hospital_id"].isin(amended_eligible)].copy()
    hf_retained = hf.loc[hf["hospital_id"].isin(amended_eligible)].copy()

    observed = {
        "eligible_hospitals": len(amended_eligible),
        "adult_plus24h_retained": len(adult_retained),
        "hf_plus24h_retained": len(hf_retained),
    }
    for k, expected in EXPECTED.items():
        if observed[k] != expected:
            raise AssertionError(
                f"{k}: expected {expected:,}, observed {observed[k]:,}"
            )

    pd.DataFrame({"hospital_id": sorted(amended_eligible)}).to_csv(
        out / "AMENDED_SITE_ELIGIBLE_HOSPITALS.csv",
        index=False,
    )

    excluded_source = sorted(set(patient["hospital_id"].unique()) - source_complete)
    excluded_target = sorted(source_complete - amended_eligible)

    pd.DataFrame([{
        "all_released_hospitals": int(patient["hospital_id"].nunique()),
        "source_complete_hospitals": int(len(source_complete)),
        "target_complete_within_source_complete_hospitals": int(len(amended_eligible)),
        "source_incomplete_hospitals": int(len(excluded_source)),
        "additional_exact_target_incomplete_hospitals": int(len(excluded_target)),
        "adult_plus24h_stays_total": int(len(adult)),
        "adult_plus24h_stays_retained_amended_rule": int(len(adult_retained)),
        "fixed_hf_plus24h_stays_total": int(len(hf)),
        "fixed_hf_plus24h_stays_retained_amended_rule": int(len(hf_retained)),
    }]).to_csv(out / "AMENDED_SITE_ELIGIBILITY_FLOW.csv", index=False)

    rows = []
    for target in ROUTINE_TARGETS:
        absent = source_complete - target_sets[target]
        rows.append({
            "model_term": target,
            "source_complete_hospitals": int(len(source_complete)),
            "hospital_wide_target_absent_hospitals": int(len(absent)),
            "adult_plus24h_stays_in_target_absent_hospitals": int(
                adult.loc[adult["hospital_id"].isin(absent)].shape[0]
            ),
            "hf_plus24h_stays_in_target_absent_hospitals": int(
                hf.loc[hf["hospital_id"].isin(absent)].shape[0]
            ),
        })
    pd.DataFrame(rows).to_csv(
        out / "AMENDED_TARGET_STRUCTURAL_ABSENCE_SUMMARY.csv",
        index=False,
    )

    add_excl = pd.DataFrame({"hospital_id": excluded_target})
    add_excl["missing_exact_targets"] = [
        ";".join([t for t in ROUTINE_TARGETS if hid not in target_sets[t]])
        for hid in excluded_target
    ]
    add_excl.to_csv(
        out / "ADDITIONAL_TARGET_INSUFFICIENT_DATA_HOSPITALS.csv",
        index=False,
    )

    pd.DataFrame([{
        "set_name": "source_complete_hospitals_152",
        "n": len(source_complete),
        "sha256_sorted_ids_newline": sha256_lines(source_complete),
    }, {
        "set_name": "amended_site_eligible_hospitals_145",
        "n": len(amended_eligible),
        "sha256_sorted_ids_newline": sha256_lines(amended_eligible),
    }, {
        "set_name": "additional_target_incomplete_hospitals_7",
        "n": len(excluded_target),
        "sha256_sorted_ids_newline": sha256_lines(excluded_target),
    }]).to_csv(out / "SITE_ID_SET_HASHES.csv", index=False)

    for path in out.glob("*.csv"):
        df0 = pd.read_csv(path, nrows=5)
        assert_safe_columns(df0.columns, path.name)

    hashes = {
        p.name: sha256_file(p)
        for p in sorted(out.glob("*.csv"), key=lambda x: x.name)
    }

    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "outcome-blind deterministic audit of amended "
            "source-complete + exact-target-complete site rule"
        ),
        "site_gate_reopened_preoutcome": True,
        "source_complete_hospitals": int(len(source_complete)),
        "amended_site_eligible_hospitals": int(len(amended_eligible)),
        "additional_target_incomplete_hospitals": int(len(excluded_target)),
        "adult_plus24h_stays_retained": int(len(adult_retained)),
        "fixed_hf_plus24h_stays_retained": int(len(hf_retained)),
        "required_source_interfaces": list(source_sql.keys()),
        "required_exact_routine_targets": ROUTINE_TARGETS,
        "positive_event_required_for_vasoactive": False,
        "positive_event_required_for_rrt": False,
        "positive_event_required_for_chronic_krt": False,
        "age_sex_additional_site_gate": False,
        "percentage_capture_threshold": None,
        "monitor_spo2_primary_fallback": False,
        "bicarbonate_hco3_totalco2_pooling": False,
        "imputation_performed": False,
        "outcome_or_discharge_status_selected": False,
        "discharge_location_selected": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "final_renal_chronic_krt_eligibility_applied": False,
        "final_episode_selection_applied": False,
        "final_cohort_membership_fixed": False,
        "expected_count_check_passed": True,
        "generated_sha256": hashes,
    }

    (out / "AMENDED_SITE_ELIGIBILITY_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("EICU AMENDED SITE-ELIGIBILITY AUDIT: PASS")
    print(f"Source-complete hospitals: {len(source_complete):,}")
    print(f"Amended source+target-complete hospitals: {len(amended_eligible):,}")
    print(f"Additional exact-target exclusions: {len(excluded_target):,}")
    print(f"Adult +24 h stays retained: {len(adult_retained):,}")
    print(f"Fixed HF +24 h stays retained: {len(hf_retained):,}")
    print("Expected-count check: PASS")
    print("Outcome/discharge-status values selected: NO")
    print("Imputation performed: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("Final cohort membership fixed: NO")
    print("=" * 78)

if __name__ == "__main__":
    main()
