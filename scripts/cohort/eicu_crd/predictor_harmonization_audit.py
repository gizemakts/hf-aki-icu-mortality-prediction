#!/usr/bin/env python3
"""Outcome-blind eICU predictor harmonization audit for the fixed MIMIC HF/AKI model.

This script is intentionally READ-ONLY with respect to raw eICU data and does not
select hospitalDischargeStatus, unitDischargeStatus, mortality, death, survival,
or any model-performance field. It does not fit a model.

Purpose: gather schema, label, unit, timing, and hospital-coverage evidence needed
to lock the eICU mapping for the 19 transformed primary-model columns.

Run example (PowerShell):
    python predictor_harmonization_audit.py `
      --eicu-root "C:\\path\\to\\eicu-crd" `
      --out "C:\\path\\to\\EICU_PREDICTOR_HARMONIZATION_AUDIT"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROHIBITED_OUTPUT_TOKENS = (
    "mortality", "expire", "death", "survival", "auc", "auroc", "brier",
    "calibration", "hospitaldischargestatus", "unitdischargestatus",
)

REQUIRED_TABLES = [
    "patient", "vitalPeriodic", "vitalAperiodic", "nurseCharting", "lab", "infusionDrug"
]


def discover_table(root: Path, stem: str) -> Path:
    wanted = {f"{stem}.csv".lower(), f"{stem}.csv.gz".lower()}
    matches = sorted(
        [p.resolve() for p in root.rglob("*") if p.is_file() and p.name.lower() in wanted],
        key=lambda p: p.as_posix().lower(),
    )
    if not matches:
        raise FileNotFoundError(f"Required eICU table not found: {stem}.csv(.gz) under {root}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple matches for {stem}:\n" + "\n".join(map(str, matches)))
    return matches[0]


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def assert_safe_columns(columns, label: str) -> None:
    bad = [c for c in columns if any(tok in str(c).lower() for tok in PROHIBITED_OUTPUT_TOKENS)]
    if bad:
        raise AssertionError(f"{label}: prohibited result columns detected: {bad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-root", required=True, help="Root containing extracted eICU CSV/CSV.GZ tables")
    ap.add_argument("--out", required=True, help="New/empty output directory for audit summaries")
    args = ap.parse_args()

    root = Path(args.eicu_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    try:
        import duckdb
        import pandas as pd
    except Exception as exc:
        raise ImportError("This audit requires duckdb and pandas.") from exc

    tables = {stem: discover_table(root, stem) for stem in REQUIRED_TABLES}
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    # ------------------------------------------------------------------
    # 1) Source inventory and schema only. Do not read outcome fields.
    # ------------------------------------------------------------------
    inventory = []
    schemas = []
    for stem, path in tables.items():
        inventory.append({"table": stem, "file_name": path.name, "size_bytes": path.stat().st_size})
        desc = con.execute(
            f"DESCRIBE SELECT * FROM read_csv_auto('{qpath(path)}', header=true, sample_size=200000)"
        ).df()
        for _, r in desc.iterrows():
            col = str(r.get("column_name", ""))
            schemas.append({"table": stem, "column": col, "type": str(r.get("column_type", ""))})
    pd.DataFrame(inventory).to_csv(out / "SOURCE_TABLE_INVENTORY.csv", index=False)
    pd.DataFrame(schemas).to_csv(out / "SOURCE_SCHEMA_AUDIT.csv", index=False)

    # ------------------------------------------------------------------
    # 2) Outcome-blind landmark denominator.
    # hospitalDischargeOffset is used only to establish availability at +24 h;
    # hospitalDischargeStatus is NEVER selected.
    # ------------------------------------------------------------------
    P = tables["patient"]
    patient = con.execute(f"""
        SELECT
            patientUnitStayID AS "patientUnitStayID",
            hospitalID AS "hospitalID",
            age AS "age",
            gender AS "gender",
            unitDischargeOffset AS "unitDischargeOffset",
            hospitalDischargeOffset AS "hospitalDischargeOffset"
        FROM read_csv_auto('{qpath(P)}', header=true, sample_size=200000)
    """).df()
    patient.columns = [str(c) for c in patient.columns]
    assert_safe_columns(patient.columns, "patient_safe")

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

    patient["age_numeric_for_eligibility"] = patient["age"].map(age_num)
    patient["adult_plus24h"] = (
        patient["age_numeric_for_eligibility"].fillna(-1).ge(18)
        & pd.to_numeric(patient["unitDischargeOffset"], errors="coerce").gt(1440)
        & pd.to_numeric(patient["hospitalDischargeOffset"], errors="coerce").gt(1440)
    )
    base = patient.loc[patient["adult_plus24h"], ["patientUnitStayID", "hospitalID"]].copy()
    if base["patientUnitStayID"].duplicated().any():
        raise AssertionError("patientUnitStayID is not unique in patient table")
    con.register("base_df", base)
    con.execute("CREATE TEMP TABLE base AS SELECT * FROM base_df")

    gender = (
        patient.loc[patient["adult_plus24h"], "gender"]
        .fillna("<NULL>")
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("gender")
        .reset_index(name="stays")
    )
    gender.to_csv(out / "DEMOGRAPHIC_CATEGORY_AUDIT.csv", index=False)

    age_summary = pd.DataFrame([{
        "adult_plus24h_stays": int(len(base)),
        "hospitals": int(base["hospitalID"].nunique()),
        "age_gt89_rows": int(patient.loc[patient["adult_plus24h"], "age"].astype(str).str.strip().str.startswith(">").sum()),
        "age_missing_or_unparseable": int(patient.loc[patient["adult_plus24h"], "age_numeric_for_eligibility"].isna().sum()),
    }])
    age_summary.to_csv(out / "AGE_TOPCODING_AUDIT.csv", index=False)

    # ------------------------------------------------------------------
    # 3) VitalPeriodic: label-free structured columns, 0..1440 min only.
    # Important: periodic values are high-frequency monitor-derived values.
    # ------------------------------------------------------------------
    VP = tables["vitalPeriodic"]
    vp = con.execute(f"""
        SELECT
            v.patientUnitStayID AS "patientUnitStayID",
            b.hospitalID AS "hospitalID",
            COUNT(*) AS rows_0_24h,
            COUNT(v.heartRate) AS hr_rows,
            COUNT(v.respiration) AS rr_rows,
            COUNT(v.temperature) AS temp_rows,
            COUNT(v.saO2) AS spo2_rows,
            COUNT(v.systemicMean) AS invasive_map_rows,
            MIN(v.observationOffset) AS min_offset,
            MAX(v.observationOffset) AS max_offset
        FROM read_csv_auto('{qpath(VP)}', header=true, sample_size=200000) v
        JOIN base b USING (patientUnitStayID)
        WHERE v.observationOffset BETWEEN 0 AND 1440
        GROUP BY 1,2
    """).df()
    assert_safe_columns(vp.columns, "vital_periodic")
    vp.to_csv(out / "VITAL_PERIODIC_STAY_COVERAGE.csv", index=False)

    # ------------------------------------------------------------------
    # 4) VitalAperiodic: non-invasive MAP, 0..1440 min only.
    # ------------------------------------------------------------------
    VA = tables["vitalAperiodic"]
    va = con.execute(f"""
        SELECT
            v.patientUnitStayID AS "patientUnitStayID",
            b.hospitalID AS "hospitalID",
            COUNT(*) AS rows_0_24h,
            COUNT(v.nonInvasiveMean) AS noninvasive_map_rows,
            MIN(v.observationOffset) AS min_offset,
            MAX(v.observationOffset) AS max_offset
        FROM read_csv_auto('{qpath(VA)}', header=true, sample_size=200000) v
        JOIN base b USING (patientUnitStayID)
        WHERE v.observationOffset BETWEEN 0 AND 1440
        GROUP BY 1,2
    """).df()
    assert_safe_columns(va.columns, "vital_aperiodic")
    va.to_csv(out / "VITAL_APERIODIC_STAY_COVERAGE.csv", index=False)

    # ------------------------------------------------------------------
    # 5) NurseCharting label inventory. Requires BOTH chart and entry offset
    # <= landmark to protect against retrospective entry.
    # Collect GCS and common vital/temperature labels only.
    # ------------------------------------------------------------------
    NC = tables["nurseCharting"]
    nc_labels = con.execute(f"""
        SELECT
            nursingChartCellTypeCat,
            nursingChartCellTypeValLabel,
            nursingChartCellTypeValName,
            COUNT(*) AS rows,
            COUNT(DISTINCT n.patientUnitStayID) AS stays,
            COUNT(DISTINCT b.hospitalID) AS hospitals,
            MIN(nursingChartOffset) AS min_chart_offset,
            MAX(nursingChartOffset) AS max_chart_offset,
            MIN(nursingChartEntryOffset) AS min_entry_offset,
            MAX(nursingChartEntryOffset) AS max_entry_offset
        FROM read_csv_auto('{qpath(NC)}', header=true, sample_size=200000) n
        JOIN base b USING (patientUnitStayID)
        WHERE nursingChartOffset BETWEEN 0 AND 1440
          AND nursingChartEntryOffset <= 1440
          AND regexp_matches(
                lower(coalesce(nursingChartCellTypeValLabel,'') || ' ' ||
                      coalesce(nursingChartCellTypeValName,'') || ' ' ||
                      coalesce(nursingChartCellTypeCat,'')),
                'glasgow|gcs|temperature|heart rate|respiratory rate|o2 saturation|spo2|sao2|mean arterial|mean bp|blood pressure'
              )
        GROUP BY 1,2,3
        ORDER BY stays DESC, rows DESC
    """).df()
    assert_safe_columns(nc_labels.columns, "nurse_charting_labels")
    nc_labels.to_csv(out / "NURSE_CHARTING_CANDIDATE_LABELS.csv", index=False)

    # ------------------------------------------------------------------
    # 6) Lab mapping inventory. labResultOffset = draw/event time;
    # labResultRevisedOffset acts as entry/revision guard.
    # Restrict to candidate primary-model labs + creatinine for KDIGO dependency.
    # ------------------------------------------------------------------
    L = tables["lab"]
    lab_labels = con.execute(f"""
        SELECT
            labName,
            labMeasureNameSystem,
            labMeasureNameInterface,
            COUNT(*) AS rows,
            COUNT(DISTINCT l.patientUnitStayID) AS stays,
            COUNT(DISTINCT b.hospitalID) AS hospitals,
            MIN(labResultOffset) AS min_result_offset,
            MAX(labResultOffset) AS max_result_offset,
            MIN(labResultRevisedOffset) AS min_revised_offset,
            MAX(labResultRevisedOffset) AS max_revised_offset
        FROM read_csv_auto('{qpath(L)}', header=true, sample_size=200000) l
        JOIN base b USING (patientUnitStayID)
        WHERE labResultOffset BETWEEN 0 AND 1440
          AND labResultRevisedOffset <= 1440
          AND regexp_matches(
                lower(coalesce(labName,'')),
                'hemoglobin|hgb|wbc|white blood|platelet|bun|urea nitrogen|bicarbonate|hco3|total co2|sodium|potassium|glucose|creatinine'
              )
        GROUP BY 1,2,3
        ORDER BY stays DESC, rows DESC
    """).df()
    assert_safe_columns(lab_labels.columns, "lab_labels")
    lab_labels.to_csv(out / "LAB_CANDIDATE_LABELS_AND_UNITS.csv", index=False)

    # Per-hospital any-lab-interface availability (all labs, not target-specific)
    lab_hosp = con.execute(f"""
        SELECT
            b.hospitalID AS "hospitalID",
            COUNT(DISTINCT b.patientUnitStayID) AS landmark_stays,
            COUNT(DISTINCT CASE WHEN l.patientUnitStayID IS NOT NULL THEN b.patientUnitStayID END) AS stays_with_any_lab_0_24h
        FROM base b
        LEFT JOIN read_csv_auto('{qpath(L)}', header=true, sample_size=200000) l
          ON l.patientUnitStayID=b.patientUnitStayID
         AND l.labResultOffset BETWEEN 0 AND 1440
         AND l.labResultRevisedOffset <= 1440
        GROUP BY 1
        ORDER BY landmark_stays DESC
    """).df()
    lab_hosp["any_lab_pct"] = 100.0 * lab_hosp["stays_with_any_lab_0_24h"] / lab_hosp["landmark_stays"]
    lab_hosp.to_csv(out / "LAB_INTERFACE_COVERAGE_BY_HOSPITAL.csv", index=False)

    # ------------------------------------------------------------------
    # 7) InfusionDrug candidate vasoactive labels and hospital capture.
    # No assumption that 'no row' means 'no drug'. This output is specifically
    # to assess structural/site missingness before defining absence=0.
    # ------------------------------------------------------------------
    ID = tables["infusionDrug"]
    inf_labels = con.execute(f"""
        SELECT
            drugName,
            COUNT(*) AS rows,
            COUNT(DISTINCT i.patientUnitStayID) AS stays,
            COUNT(DISTINCT b.hospitalID) AS hospitals,
            MIN(infusionOffset) AS min_offset,
            MAX(infusionOffset) AS max_offset
        FROM read_csv_auto('{qpath(ID)}', header=true, sample_size=200000) i
        JOIN base b USING (patientUnitStayID)
        WHERE infusionOffset BETWEEN 0 AND 1440
          AND regexp_matches(
                lower(coalesce(drugName,'')),
                'norepinephrine|noradrenaline|epinephrine|adrenaline|phenylephrine|vasopressin|dopamine|dobutamine|milrinone'
              )
        GROUP BY 1
        ORDER BY stays DESC, rows DESC
    """).df()
    assert_safe_columns(inf_labels.columns, "infusion_labels")
    inf_labels.to_csv(out / "INFUSION_VASOACTIVE_CANDIDATE_LABELS.csv", index=False)

    inf_hosp = con.execute(f"""
        WITH any_inf AS (
            SELECT DISTINCT patientUnitStayID
            FROM read_csv_auto('{qpath(ID)}', header=true, sample_size=200000)
            WHERE infusionOffset BETWEEN 0 AND 1440
        ), target_inf AS (
            SELECT DISTINCT patientUnitStayID
            FROM read_csv_auto('{qpath(ID)}', header=true, sample_size=200000)
            WHERE infusionOffset BETWEEN 0 AND 1440
              AND regexp_matches(
                    lower(coalesce(drugName,'')),
                    'norepinephrine|noradrenaline|epinephrine|adrenaline|phenylephrine|vasopressin|dopamine|dobutamine|milrinone'
                  )
        )
        SELECT
            b.hospitalID AS "hospitalID",
            COUNT(*) AS landmark_stays,
            SUM(CASE WHEN a.patientUnitStayID IS NOT NULL THEN 1 ELSE 0 END) AS stays_with_any_infusion_row,
            SUM(CASE WHEN t.patientUnitStayID IS NOT NULL THEN 1 ELSE 0 END) AS stays_with_target_vasoactive_row
        FROM base b
        LEFT JOIN any_inf a USING (patientUnitStayID)
        LEFT JOIN target_inf t USING (patientUnitStayID)
        GROUP BY 1
        ORDER BY landmark_stays DESC
    """).df()
    inf_hosp["any_infusion_row_pct"] = 100.0 * inf_hosp["stays_with_any_infusion_row"] / inf_hosp["landmark_stays"]
    inf_hosp["target_vasoactive_row_pct"] = 100.0 * inf_hosp["stays_with_target_vasoactive_row"] / inf_hosp["landmark_stays"]
    inf_hosp.to_csv(out / "INFUSION_INTERFACE_COVERAGE_BY_HOSPITAL.csv", index=False)

    # ------------------------------------------------------------------
    # 8) Hospital-level vital capture. Absence remains structural-unknown until
    # explicitly adjudicated; these are coverage diagnostics only.
    # ------------------------------------------------------------------
    con.register("vp_cov_df", vp)
    con.register("va_cov_df", va)
    con.execute("CREATE TEMP TABLE vp_cov AS SELECT * FROM vp_cov_df")
    con.execute("CREATE TEMP TABLE va_cov AS SELECT * FROM va_cov_df")
    vital_hosp = con.execute("""
        SELECT
            b.hospitalID AS "hospitalID",
            COUNT(*) AS landmark_stays,
            100.0 * AVG(CASE WHEN coalesce(v.hr_rows,0)>0 THEN 1 ELSE 0 END) AS hr_periodic_pct,
            100.0 * AVG(CASE WHEN coalesce(v.rr_rows,0)>0 THEN 1 ELSE 0 END) AS rr_periodic_pct,
            100.0 * AVG(CASE WHEN coalesce(v.temp_rows,0)>0 THEN 1 ELSE 0 END) AS temp_periodic_pct,
            100.0 * AVG(CASE WHEN coalesce(v.spo2_rows,0)>0 THEN 1 ELSE 0 END) AS spo2_periodic_pct,
            100.0 * AVG(CASE WHEN coalesce(v.invasive_map_rows,0)>0 OR coalesce(a.noninvasive_map_rows,0)>0 THEN 1 ELSE 0 END) AS any_map_pct
        FROM base b
        LEFT JOIN vp_cov v USING (patientUnitStayID, hospitalID)
        LEFT JOIN va_cov a USING (patientUnitStayID, hospitalID)
        GROUP BY 1
        ORDER BY landmark_stays DESC
    """).df()
    vital_hosp.to_csv(out / "VITAL_COVERAGE_BY_HOSPITAL.csv", index=False)

    # ------------------------------------------------------------------
    # 9) Fixed-model input dictionary (19 transformed columns) for traceability.
    # Note: 19 transformed columns correspond to 18 clinical constructs because
    # kdigo_stage contributes two indicator columns with Stage 1 as reference.
    # ------------------------------------------------------------------
    model_inputs = [
        ("age_per10y", "age", "age capped at 91 years, divided by 10"),
        ("sex_female", "sex", "female indicator"),
        ("heart_rate_max_per10", "heart rate", "maximum 0-24 h, divided by 10"),
        ("map_min_per10", "MAP", "minimum 0-24 h, divided by 10"),
        ("resp_rate_max_per5", "respiratory rate", "maximum 0-24 h, divided by 5"),
        ("temperature_max_abs_deviation_from_37c", "temperature", "max(|min_temp-37|, |max_temp-37|)"),
        ("spo2_deficit_per5", "SpO2", "(100 - minimum SpO2 0-24 h) / 5"),
        ("gcs_deficit_from15", "GCS", "15 - minimum GCS 0-24 h"),
        ("hemoglobin_min_gdl", "hemoglobin", "minimum 0-24 h, g/dL"),
        ("wbc_max_log2", "WBC", "log2(maximum 0-24 h)"),
        ("platelet_min_log2", "platelets", "log2(minimum 0-24 h)"),
        ("bun_max_log2", "BUN", "log2(maximum 0-24 h)"),
        ("bicarbonate_min_per5", "bicarbonate", "minimum 0-24 h divided by 5"),
        ("sodium_max_abs_deviation_from140_per5", "sodium", "max(|min-140|, |max-140|) / 5"),
        ("potassium_max_mmol_l", "potassium", "maximum 0-24 h, mmol/L"),
        ("glucose_max_log2", "glucose", "log2(maximum 0-24 h, mg/dL)"),
        ("vasoactive_any_by_landmark", "vasoactive support", "any prespecified vasopressor or inotrope by +24 h"),
        ("kdigo_stage_2_indicator", "KDIGO stage", "1 if Stage 2; Stage 1 reference"),
        ("kdigo_stage_3_indicator", "KDIGO stage", "1 if Stage 3; Stage 1 reference"),
    ]
    pd.DataFrame(model_inputs, columns=["model_column", "clinical_construct", "fixed_mimic_transform"]).to_csv(
        out / "FIXED_MODEL_INPUT_DICTIONARY.csv", index=False
    )

    # ------------------------------------------------------------------
    # 10) Manifest. Hash only generated audit outputs, not huge raw inputs.
    # ------------------------------------------------------------------
    for path in out.glob("*.csv"):
        # Safety check against accidentally generated prohibited columns.
        df0 = pd.read_csv(path, nrows=5)
        assert_safe_columns(df0.columns, path.name)

    output_hashes = {
        p.name: sha256_file(p)
        for p in sorted(out.glob("*.csv"), key=lambda x: x.name)
    }
    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "outcome-blind evidence collection for eICU predictor harmonization",
        "raw_data_modified": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "hospitalDischargeStatus_selected": False,
        "unitDischargeStatus_selected": False,
        "adult_plus24h_stays": int(len(base)),
        "hospitals": int(base["hospitalID"].nunique()),
        "required_tables": REQUIRED_TABLES,
        "source_files": {k: v.name for k, v in tables.items()},
        "generated_sha256": output_hashes,
        "interpretation": (
            "Audit summaries only. No harmonization rule is fixed by this script. "
            "Absence of infusion or other site-dependent rows must not be interpreted as clinical absence."
        ),
    }
    (out / "AUDIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 78)
    print("EICU PREDICTOR HARMONIZATION AUDIT: PASS")
    print(f"Adult +24 h stays: {len(base):,}")
    print(f"Hospitals: {base['hospitalID'].nunique():,}")
    print(f"Output: {out}")
    print("Outcome/discharge-status values selected: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("=" * 78)


if __name__ == "__main__":
    main()
