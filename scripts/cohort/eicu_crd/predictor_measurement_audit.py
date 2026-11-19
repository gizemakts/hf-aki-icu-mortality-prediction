#!/usr/bin/env python3
"""
Outcome-blind eICU predictor measurement audit.

DuckDB field names are normalized before pandas joins, and hospital coverage
uses explicitly named aggregations.

Purpose
-------
Resolve the remaining measurement questions needed to lock the
EICU_PREDICTOR_HARMONIZATION_MATRIX without using any mortality outcome.

This audit is read-only with respect to raw eICU data. It does not select
hospitalDischargeStatus, unitDischargeStatus, mortality, death, survival, or
performance fields. It does not fit or evaluate a model.

It focuses on:
  * candidate source comparison for first-24h vital-sign extremes,
  * GCS source/representation audit,
  * target-lab label/unit/coverage/duplicate/revision audit,
  * vasoactive label/rate/site-capture audit,
  * hospital-level structural availability.

The denominator is the same outcome-blind adult +24 h availability population
used by the prior harmonization audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

PROHIBITED_OUTPUT_TOKENS = (
    "mortality", "expire", "death", "survival", "auc", "auroc", "brier",
    "calibration", "hospitaldischargestatus", "unitdischargestatus",
)

REQUIRED_TABLES = [
    "patient",
    "vitalPeriodic",
    "vitalAperiodic",
    "nurseCharting",
    "lab",
    "infusionDrug",
    "treatment",
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
    bad = [
        c for c in columns
        if any(tok in str(c).lower() for tok in PROHIBITED_OUTPUT_TOKENS)
    ]
    if bad:
        raise AssertionError(f"{label}: prohibited result columns detected: {bad}")



def normalize_id_columns(df):
    """Normalize eICU/DuckDB identifier/name casing used by downstream pandas code."""
    canonical = {
        "patientunitstayid": "patientUnitStayID",
        "hospitalid": "hospitalID",
        "labname": "labName",
        "labtypeid": "labTypeID",
        "labmeasurenamesystem": "labMeasureNameSystem",
        "labmeasurenameinterface": "labMeasureNameInterface",
        "infusionoffset": "infusionOffset",
        "drugname": "drugName",
    }
    rename = {}
    for c in df.columns:
        target = canonical.get(str(c).lower())
        if target is not None and c != target:
            rename[c] = target
    return df.rename(columns=rename) if rename else df

def quantiles(series):
    import pandas as pd
    x = pd.to_numeric(series, errors="coerce").dropna()
    if x.empty:
        return {
            "n": 0, "p01": None, "p05": None, "p25": None,
            "p50": None, "p75": None, "p95": None, "p99": None,
            "min": None, "max": None,
        }
    q = x.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(x.size),
        "p01": float(q.loc[0.01]),
        "p05": float(q.loc[0.05]),
        "p25": float(q.loc[0.25]),
        "p50": float(q.loc[0.50]),
        "p75": float(q.loc[0.75]),
        "p95": float(q.loc[0.95]),
        "p99": float(q.loc[0.99]),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-root", required=True)
    ap.add_argument("--out", required=True)
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
        import numpy as np
        import pandas as pd
    except Exception as exc:
        raise ImportError("This audit requires duckdb, numpy, and pandas.") from exc

    tables = {stem: discover_table(root, stem) for stem in REQUIRED_TABLES}
    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    # ------------------------------------------------------------------
    # 1) Outcome-blind adult +24 h denominator.
    # Only discharge OFFSETS are used for landmark availability.
    # No discharge STATUS, location, death, or outcome field is selected.
    # ------------------------------------------------------------------
    P = tables["patient"]
    patient = con.execute(f"""
        SELECT
            patientUnitStayID AS patientUnitStayID,
            hospitalID AS hospitalID,
            age AS age,
            gender AS gender,
            unitDischargeOffset AS unitDischargeOffset,
            hospitalDischargeOffset AS hospitalDischargeOffset
        FROM read_csv_auto('{qpath(P)}', header=true, sample_size=200000)
    """).df()
    patient = normalize_id_columns(patient)
    assert_safe_columns(patient.columns, "patient_safe")

    def age_num(x):
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        if s.startswith(">"):
            return 90.0  # eligibility only; NOT a fixed model-mapping rule
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
    base = patient.loc[
        patient["adult_plus24h"], ["patientUnitStayID", "hospitalID"]
    ].copy()
    if base["patientUnitStayID"].duplicated().any():
        raise AssertionError("patientUnitStayID is not unique in patient table")

    con.register("base_df", base)
    con.execute("CREATE TEMP TABLE base AS SELECT * FROM base_df")
    n_base = int(len(base))

    # ------------------------------------------------------------------
    # 2) Structured monitor vitals: MIMIC-like plausibility rules.
    # ------------------------------------------------------------------
    VP = tables["vitalPeriodic"]
    vp = con.execute(f"""
        SELECT
            v.patientUnitStayID,
            b.hospitalID,
            MAX(CASE WHEN v.heartRate > 0 AND v.heartRate < 300
                     THEN v.heartRate END) AS hr_max,
            MAX(CASE WHEN v.respiration > 0 AND v.respiration < 70
                     THEN v.respiration END) AS rr_max,
            MIN(CASE WHEN v.temperature > 10 AND v.temperature < 50
                     THEN v.temperature END) AS temp_min_c,
            MAX(CASE WHEN v.temperature > 10 AND v.temperature < 50
                     THEN v.temperature END) AS temp_max_c,
            MIN(CASE WHEN v.saO2 > 0 AND v.saO2 <= 100
                     THEN v.saO2 END) AS spo2_min,
            MIN(CASE WHEN v.systemicMean > 0 AND v.systemicMean < 300
                     THEN v.systemicMean END) AS invasive_map_min
        FROM read_csv_auto('{qpath(VP)}', header=true, sample_size=200000) v
        JOIN base b USING (patientUnitStayID)
        WHERE v.observationOffset BETWEEN 0 AND 1440
        GROUP BY 1,2
    """).df()
    vp = normalize_id_columns(vp)

    VA = tables["vitalAperiodic"]
    va = con.execute(f"""
        SELECT
            v.patientUnitStayID,
            b.hospitalID,
            MIN(CASE WHEN v.nonInvasiveMean > 0 AND v.nonInvasiveMean < 300
                     THEN v.nonInvasiveMean END) AS noninvasive_map_min
        FROM read_csv_auto('{qpath(VA)}', header=true, sample_size=200000) v
        JOIN base b USING (patientUnitStayID)
        WHERE v.observationOffset BETWEEN 0 AND 1440
        GROUP BY 1,2
    """).df()
    va = normalize_id_columns(va)

    monitor = base.merge(vp, on=["patientUnitStayID", "hospitalID"], how="left")
    monitor = monitor.merge(va, on=["patientUnitStayID", "hospitalID"], how="left")
    monitor["map_min"] = monitor[
        ["invasive_map_min", "noninvasive_map_min"]
    ].min(axis=1, skipna=True)
    monitor.loc[
        monitor[["invasive_map_min", "noninvasive_map_min"]].isna().all(axis=1),
        "map_min",
    ] = np.nan

    # ------------------------------------------------------------------
    # 3) Nurse-charted vitals with chart-time + entry-time guard.
    # Exact labels follow the eICU structured nurseCharting taxonomy and
    # MIT-LCP pivoted-vital conventions, while retaining both C and F temp.
    # ------------------------------------------------------------------
    NC = tables["nurseCharting"]
    nc_vitals = con.execute(f"""
        WITH x AS (
            SELECT
                n.patientUnitStayID,
                b.hospitalID,
                n.nursingChartOffset AS chartoffset,
                n.nursingChartEntryOffset AS entryoffset,
                n.nursingChartCellTypeValLabel AS vallabel,
                n.nursingChartCellTypeValName AS valname,
                TRY_CAST(n.nursingChartValue AS DOUBLE) AS v
            FROM read_csv_auto('{qpath(NC)}', header=true, sample_size=200000) n
            JOIN base b USING (patientUnitStayID)
            WHERE n.nursingChartOffset BETWEEN 0 AND 1440
              AND n.nursingChartEntryOffset <= 1440
              AND n.nursingChartCellTypeCat IN
                    ('Vital Signs','Scores','Other Vital Signs and Infusions')
        )
        SELECT
            patientUnitStayID,
            hospitalID,
            MAX(CASE WHEN vallabel='Heart Rate'
                       AND valname='Heart Rate'
                       AND v>0 AND v<300
                     THEN v END) AS hr_max,
            MAX(CASE WHEN vallabel='Respiratory Rate'
                       AND valname='Respiratory Rate'
                       AND v>0 AND v<70
                     THEN v END) AS rr_max,
            MIN(CASE
                  WHEN vallabel='Temperature' AND valname='Temperature (C)'
                       AND v>10 AND v<50 THEN v
                  WHEN vallabel='Temperature' AND valname='Temperature (F)'
                       AND v>70 AND v<120 THEN (v-32.0)/1.8
                END) AS temp_min_c,
            MAX(CASE
                  WHEN vallabel='Temperature' AND valname='Temperature (C)'
                       AND v>10 AND v<50 THEN v
                  WHEN vallabel='Temperature' AND valname='Temperature (F)'
                       AND v>70 AND v<120 THEN (v-32.0)/1.8
                END) AS temp_max_c,
            MIN(CASE
                  WHEN vallabel='O2 Saturation'
                       AND valname='O2 Saturation'
                       AND v>0 AND v<=100 THEN v
                  WHEN vallabel='SpO2'
                       AND valname='Value'
                       AND v>0 AND v<=100 THEN v
                END) AS spo2_min,
            MIN(CASE
                  WHEN vallabel='Non-Invasive BP'
                       AND valname='Non-Invasive BP Mean'
                       AND v>0 AND v<300 THEN v
                  WHEN vallabel='Invasive BP'
                       AND valname='Invasive BP Mean'
                       AND v>0 AND v<300 THEN v
                  WHEN vallabel='MAP (mmHg)'
                       AND valname='Value'
                       AND v>0 AND v<300 THEN v
                  WHEN vallabel='Arterial Line MAP (mmHg)'
                       AND valname='Value'
                       AND v>0 AND v<300 THEN v
                END) AS map_min
        FROM x
        GROUP BY 1,2
    """).df()
    nc_vitals = normalize_id_columns(nc_vitals)

    nurse = base.merge(
        nc_vitals, on=["patientUnitStayID", "hospitalID"], how="left"
    )

    # Candidate label inventory specifically for BP/MAP and GCS labels.
    nc_labels = con.execute(f"""
        SELECT
            n.nursingChartCellTypeCat,
            n.nursingChartCellTypeValLabel,
            n.nursingChartCellTypeValName,
            COUNT(*) AS rows,
            COUNT(DISTINCT n.patientUnitStayID) AS stays,
            COUNT(DISTINCT b.hospitalID) AS hospitals,
            MIN(n.nursingChartOffset) AS min_chart_offset,
            MAX(n.nursingChartOffset) AS max_chart_offset,
            MIN(n.nursingChartEntryOffset) AS min_entry_offset,
            MAX(n.nursingChartEntryOffset) AS max_entry_offset
        FROM read_csv_auto('{qpath(NC)}', header=true, sample_size=200000) n
        JOIN base b USING (patientUnitStayID)
        WHERE n.nursingChartOffset BETWEEN 0 AND 1440
          AND n.nursingChartEntryOffset <= 1440
          AND regexp_matches(
                lower(
                    coalesce(n.nursingChartCellTypeValLabel,'') || ' ' ||
                    coalesce(n.nursingChartCellTypeValName,'')
                ),
                'glasgow|gcs|blood pressure|non-invasive bp|invasive bp|map|arterial line'
              )
        GROUP BY 1,2,3
        ORDER BY stays DESC, rows DESC
    """).df()
    nc_labels = normalize_id_columns(nc_labels)
    assert_safe_columns(nc_labels.columns, "nurse_label_inventory")
    nc_labels.to_csv(out / "NURSE_BP_GCS_LABEL_INVENTORY.csv", index=False)

    # Same-chartoffset duplicate/conflict audit for exact nurse vital labels.
    nc_dups = con.execute(f"""
        WITH x AS (
            SELECT
                n.patientUnitStayID,
                n.nursingChartOffset AS chartoffset,
                n.nursingChartCellTypeValLabel AS vallabel,
                n.nursingChartCellTypeValName AS valname,
                TRY_CAST(n.nursingChartValue AS DOUBLE) AS v
            FROM read_csv_auto('{qpath(NC)}', header=true, sample_size=200000) n
            JOIN base b USING (patientUnitStayID)
            WHERE n.nursingChartOffset BETWEEN 0 AND 1440
              AND n.nursingChartEntryOffset <= 1440
              AND TRY_CAST(n.nursingChartValue AS DOUBLE) IS NOT NULL
              AND (
                    (n.nursingChartCellTypeValLabel='Heart Rate'
                     AND n.nursingChartCellTypeValName='Heart Rate')
                 OR (n.nursingChartCellTypeValLabel='Respiratory Rate'
                     AND n.nursingChartCellTypeValName='Respiratory Rate')
                 OR (n.nursingChartCellTypeValLabel='O2 Saturation'
                     AND n.nursingChartCellTypeValName='O2 Saturation')
                 OR (n.nursingChartCellTypeValLabel='Temperature'
                     AND n.nursingChartCellTypeValName IN ('Temperature (C)','Temperature (F)'))
                 OR (n.nursingChartCellTypeValLabel='Non-Invasive BP'
                     AND n.nursingChartCellTypeValName='Non-Invasive BP Mean')
                 OR (n.nursingChartCellTypeValLabel='Invasive BP'
                     AND n.nursingChartCellTypeValName='Invasive BP Mean')
                 OR (n.nursingChartCellTypeValLabel IN ('MAP (mmHg)','Arterial Line MAP (mmHg)')
                     AND n.nursingChartCellTypeValName='Value')
              )
        ),
        g AS (
            SELECT
                vallabel, valname,
                patientUnitStayID, chartoffset,
                COUNT(*) AS n_rows,
                MIN(v) AS min_v,
                MAX(v) AS max_v
            FROM x
            GROUP BY 1,2,3,4
        )
        SELECT
            vallabel,
            valname,
            COUNT(*) AS same_time_groups,
            SUM(CASE WHEN n_rows>1 THEN 1 ELSE 0 END) AS duplicate_groups,
            SUM(CASE WHEN n_rows>1 AND min_v<>max_v THEN 1 ELSE 0 END) AS conflicting_duplicate_groups,
            MAX(n_rows) AS max_rows_same_time
        FROM g
        GROUP BY 1,2
        ORDER BY duplicate_groups DESC, same_time_groups DESC
    """).df()
    nc_dups = normalize_id_columns(nc_dups)
    nc_dups.to_csv(out / "NURSE_SAME_TIME_DUPLICATE_AUDIT.csv", index=False)

    # ------------------------------------------------------------------
    # 4) Outcome-blind monitor-vs-nurse source comparison.
    # ------------------------------------------------------------------
    vital_variables = ["hr_max", "rr_max", "map_min", "temp_min_c", "temp_max_c", "spo2_min"]
    dist_rows = []
    compare_rows = []
    hosp_rows = []

    for var in vital_variables:
        for source_name, df in [("monitor", monitor), ("nurseCharting", nurse)]:
            q = quantiles(df[var])
            dist_rows.append({
                "variable": var,
                "source": source_name,
                "available_stays": int(pd.to_numeric(df[var], errors="coerce").notna().sum()),
                "available_pct": float(pd.to_numeric(df[var], errors="coerce").notna().mean() * 100.0),
                **q,
            })

            tmp = df[["hospitalID", var]].copy()
            tmp["available"] = pd.to_numeric(tmp[var], errors="coerce").notna().astype(int)
            h = (
                tmp.groupby("hospitalID", as_index=False)
                .agg(
                    available_stays=("available", "sum"),
                    landmark_stays=("available", "size"),
                )
            )
            h["available_pct"] = 100.0 * h["available_stays"] / h["landmark_stays"]
            h["variable"] = var
            h["source"] = source_name
            hosp_rows.append(h)

        pair = monitor[["patientUnitStayID", var]].merge(
            nurse[["patientUnitStayID", var]],
            on="patientUnitStayID",
            how="inner",
            suffixes=("_monitor", "_nurse"),
        )
        a = pd.to_numeric(pair[f"{var}_monitor"], errors="coerce")
        b = pd.to_numeric(pair[f"{var}_nurse"], errors="coerce")
        ok = a.notna() & b.notna()
        d = (a[ok] - b[ok]).astype(float)
        compare_rows.append({
            "variable": var,
            "paired_stays": int(ok.sum()),
            "paired_pct_of_landmark": float(ok.sum() / n_base * 100.0),
            "median_monitor_minus_nurse": float(d.median()) if len(d) else None,
            "p05_monitor_minus_nurse": float(d.quantile(0.05)) if len(d) else None,
            "p95_monitor_minus_nurse": float(d.quantile(0.95)) if len(d) else None,
            "mean_abs_difference": float(d.abs().mean()) if len(d) else None,
            "exact_agreement_pct": float((d.abs() < 1e-12).mean() * 100.0) if len(d) else None,
            "within_1_unit_pct": float((d.abs() <= 1.0).mean() * 100.0) if len(d) else None,
            "pearson_r": float(a[ok].corr(b[ok])) if ok.sum() >= 2 else None,
        })

    pd.DataFrame(dist_rows).to_csv(out / "VITAL_SOURCE_DISTRIBUTION_SUMMARY.csv", index=False)
    pd.DataFrame(compare_rows).to_csv(out / "VITAL_SOURCE_PAIRED_COMPARISON.csv", index=False)
    pd.concat(hosp_rows, ignore_index=True).to_csv(
        out / "VITAL_SOURCE_COVERAGE_BY_HOSPITAL.csv", index=False
    )

    # ------------------------------------------------------------------
    # 5) GCS mapping audit.
    # Direct total and component representations are audited separately.
    # No mortality/outcome is used to choose between them.
    # ------------------------------------------------------------------
    gcs_events = con.execute(f"""
        WITH x AS (
            SELECT
                n.patientUnitStayID,
                b.hospitalID,
                n.nursingChartOffset AS chartoffset,
                n.nursingChartEntryOffset AS entryoffset,
                n.nursingChartCellTypeCat AS cat,
                n.nursingChartCellTypeValLabel AS vallabel,
                n.nursingChartCellTypeValName AS valname,
                n.nursingChartValue AS raw_value,
                TRY_CAST(n.nursingChartValue AS DOUBLE) AS v
            FROM read_csv_auto('{qpath(NC)}', header=true, sample_size=200000) n
            JOIN base b USING (patientUnitStayID)
            WHERE n.nursingChartOffset BETWEEN 0 AND 1440
              AND n.nursingChartEntryOffset <= 1440
              AND n.nursingChartCellTypeCat IN ('Scores','Other Vital Signs and Infusions')
              AND (
                    lower(coalesce(n.nursingChartCellTypeValLabel,'')) LIKE '%glasgow%'
                 OR lower(coalesce(n.nursingChartCellTypeValLabel,'')) LIKE '%gcs%'
                 OR lower(coalesce(n.nursingChartCellTypeValName,'')) LIKE '%verbal%'
                 OR lower(coalesce(n.nursingChartCellTypeValName,'')) LIKE '%motor%'
                 OR lower(coalesce(n.nursingChartCellTypeValName,'')) LIKE '%eye%'
              )
        ),
        y AS (
            SELECT
                patientUnitStayID,
                hospitalID,
                chartoffset,
                MIN(CASE
                    WHEN vallabel='Glasgow coma score' AND valname='GCS Total'
                         AND v BETWEEN 3 AND 15 THEN v
                    WHEN vallabel='Score (Glasgow Coma Scale)' AND valname='Value'
                         AND v BETWEEN 3 AND 15 THEN v
                END) AS direct_total,
                MAX(CASE
                    WHEN vallabel='Glasgow coma score' AND valname='Motor'
                         AND v BETWEEN 1 AND 6 THEN v
                END) AS motor,
                MAX(CASE
                    WHEN vallabel='Glasgow coma score' AND valname='Verbal'
                         AND v BETWEEN 1 AND 5 THEN v
                END) AS verbal,
                MAX(CASE
                    WHEN vallabel='Glasgow coma score' AND valname='Eyes'
                         AND v BETWEEN 1 AND 4 THEN v
                END) AS eyes,
                MAX(CASE
                    WHEN lower(raw_value) LIKE '%intubat%'
                      OR lower(raw_value) LIKE '%trach%'
                    THEN 1 ELSE 0 END) AS intub_or_trach_text,
                MAX(CASE
                    WHEN lower(raw_value) LIKE '%unable to score%'
                    THEN 1 ELSE 0 END) AS unable_to_score_text
            FROM x
            GROUP BY 1,2,3
        )
        SELECT
            *,
            CASE WHEN motor IS NOT NULL AND verbal IS NOT NULL AND eyes IS NOT NULL
                 THEN motor + verbal + eyes END AS component_sum
        FROM y
    """).df()
    gcs_events = normalize_id_columns(gcs_events)

    gcs_summary_rows = []
    for col in ["direct_total", "component_sum"]:
        q = quantiles(gcs_events.groupby("patientUnitStayID")[col].min())
        stay_vals = gcs_events.groupby("patientUnitStayID")[col].min()
        gcs_summary_rows.append({
            "representation": col,
            "available_stays": int(stay_vals.notna().sum()),
            "available_pct": float(stay_vals.notna().sum() / n_base * 100.0),
            **q,
        })

    paired_gcs = gcs_events.dropna(subset=["direct_total", "component_sum"]).copy()
    gcs_event_agreement = {
        "representation": "same_offset_direct_vs_component",
        "available_stays": int(paired_gcs["patientUnitStayID"].nunique()),
        "available_pct": float(paired_gcs["patientUnitStayID"].nunique() / n_base * 100.0),
        "n": int(len(paired_gcs)),
        "p01": None, "p05": None, "p25": None, "p50": None,
        "p75": None, "p95": None, "p99": None, "min": None, "max": None,
        "exact_agreement_pct": float(
            (paired_gcs["direct_total"] == paired_gcs["component_sum"]).mean() * 100.0
        ) if len(paired_gcs) else None,
        "median_direct_minus_component": float(
            (paired_gcs["direct_total"] - paired_gcs["component_sum"]).median()
        ) if len(paired_gcs) else None,
    }
    gcs_df = pd.DataFrame(gcs_summary_rows)
    gcs_df["exact_agreement_pct"] = np.nan
    gcs_df["median_direct_minus_component"] = np.nan
    gcs_df = pd.concat([gcs_df, pd.DataFrame([gcs_event_agreement])], ignore_index=True)
    gcs_df.to_csv(out / "GCS_REPRESENTATION_AUDIT.csv", index=False)

    pd.DataFrame([{
        "landmark_stays": n_base,
        "stays_with_any_intub_or_trach_text": int(
            gcs_events.loc[gcs_events["intub_or_trach_text"].eq(1), "patientUnitStayID"].nunique()
        ),
        "stays_with_unable_to_score_text": int(
            gcs_events.loc[gcs_events["unable_to_score_text"].eq(1), "patientUnitStayID"].nunique()
        ),
    }]).to_csv(out / "GCS_SPECIAL_TEXT_AUDIT.csv", index=False)

    # ------------------------------------------------------------------
    # 6) Laboratory mapping audit.
    # Draw time = labResultOffset; revision-entry guard = labResultRevisedOffset.
    # Candidate names are audited separately rather than silently merged.
    # ------------------------------------------------------------------
    L = tables["lab"]
    lab_candidates = {
        "hemoglobin": ["Hgb"],
        "wbc": ["WBC x 1000"],
        "platelets": ["platelets x 1000"],
        "bun": ["BUN"],
        "bicarbonate": ["bicarbonate", "HCO3", "Total CO2"],
        "sodium": ["sodium"],
        "potassium": ["potassium"],
        "glucose": ["glucose", "bedside glucose"],
        "creatinine": ["creatinine"],
    }
    flat_names = sorted({x for xs in lab_candidates.values() for x in xs})
    names_sql = ",".join("'" + x.replace("'", "''") + "'" for x in flat_names)

    lab_label = con.execute(f"""
        SELECT
            l.labName,
            l.labTypeID,
            l.labMeasureNameSystem,
            l.labMeasureNameInterface,
            COUNT(*) AS rows_drawn_0_24h,
            SUM(CASE WHEN l.labResultRevisedOffset <= 1440 THEN 1 ELSE 0 END)
                AS rows_revision_admissible,
            SUM(CASE WHEN l.labResultRevisedOffset > 1440 THEN 1 ELSE 0 END)
                AS rows_revised_after_landmark,
            COUNT(DISTINCT l.patientUnitStayID) AS stays_drawn_0_24h,
            COUNT(DISTINCT CASE WHEN l.labResultRevisedOffset <= 1440
                                THEN l.patientUnitStayID END)
                AS stays_revision_admissible,
            COUNT(DISTINCT b.hospitalID) AS hospitals
        FROM read_csv_auto('{qpath(L)}', header=true, sample_size=200000) l
        JOIN base b USING (patientUnitStayID)
        WHERE l.labResultOffset BETWEEN 0 AND 1440
          AND l.labName IN ({names_sql})
        GROUP BY 1,2,3,4
        ORDER BY labName, stays_revision_admissible DESC, rows_revision_admissible DESC
    """).df()
    lab_label = normalize_id_columns(lab_label)

    lab_name_to_construct = {}
    for construct, names in lab_candidates.items():
        for name in names:
            lab_name_to_construct[name] = construct
    lab_label.insert(
        0, "clinical_construct",
        lab_label["labName"].map(lab_name_to_construct)
    )
    lab_label.to_csv(out / "LAB_TARGET_LABEL_UNIT_AUDIT.csv", index=False)

    # Per-stay candidate-label extremes for distribution/coverage comparison.
    lab_stay = con.execute(f"""
        SELECT
            l.patientUnitStayID,
            b.hospitalID,
            l.labName,
            l.labTypeID,
            MIN(l.labResult) AS min_value,
            MAX(l.labResult) AS max_value,
            COUNT(*) AS admissible_rows
        FROM read_csv_auto('{qpath(L)}', header=true, sample_size=200000) l
        JOIN base b USING (patientUnitStayID)
        WHERE l.labResultOffset BETWEEN 0 AND 1440
          AND l.labResultRevisedOffset <= 1440
          AND l.labName IN ({names_sql})
          AND l.labResult IS NOT NULL
        GROUP BY 1,2,3,4
    """).df()
    lab_stay = normalize_id_columns(lab_stay)

    lab_dist_rows = []
    lab_hosp_rows = []
    for name in flat_names:
        sub = lab_stay.loc[lab_stay["labName"].eq(name)].copy()
        construct = lab_name_to_construct[name]
        if construct in {"hemoglobin", "platelets", "bicarbonate"}:
            selected = sub["min_value"]
            aggregation = "minimum"
        elif construct in {"wbc", "bun", "potassium", "glucose"}:
            selected = sub["max_value"]
            aggregation = "maximum"
        elif construct in {"sodium", "creatinine"}:
            # Both ends are relevant for sodium; creatinine is a renal dependency.
            selected = sub["max_value"]
            aggregation = "max_for_audit"
        else:
            selected = sub["max_value"]
            aggregation = "max_for_audit"

        q = quantiles(selected)
        lab_dist_rows.append({
            "clinical_construct": construct,
            "labName": name,
            "aggregation_for_summary": aggregation,
            "available_stays": int(sub["patientUnitStayID"].nunique()),
            "available_pct": float(sub["patientUnitStayID"].nunique() / n_base * 100.0),
            "hospitals": int(sub["hospitalID"].nunique()),
            **q,
        })

        # Candidate-specific hospital capture.
        base_h = base.groupby("hospitalID").size().rename("landmark_stays").reset_index()
        sh = sub.groupby("hospitalID")["patientUnitStayID"].nunique().rename("available_stays").reset_index()
        hh = base_h.merge(sh, on="hospitalID", how="left")
        hh["available_stays"] = hh["available_stays"].fillna(0).astype(int)
        hh["available_pct"] = 100.0 * hh["available_stays"] / hh["landmark_stays"]
        hh["clinical_construct"] = construct
        hh["labName"] = name
        lab_hosp_rows.append(hh)

    pd.DataFrame(lab_dist_rows).to_csv(out / "LAB_TARGET_DISTRIBUTION_SUMMARY.csv", index=False)
    pd.concat(lab_hosp_rows, ignore_index=True).to_csv(
        out / "LAB_TARGET_COVERAGE_BY_HOSPITAL.csv", index=False
    )

    # Same draw-time duplicate/conflict audit.
    lab_dups = con.execute(f"""
        WITH g AS (
            SELECT
                l.labName,
                l.patientUnitStayID,
                l.labResultOffset,
                COUNT(*) AS n_rows,
                MIN(l.labResult) AS min_v,
                MAX(l.labResult) AS max_v
            FROM read_csv_auto('{qpath(L)}', header=true, sample_size=200000) l
            JOIN base b USING (patientUnitStayID)
            WHERE l.labResultOffset BETWEEN 0 AND 1440
              AND l.labResultRevisedOffset <= 1440
              AND l.labName IN ({names_sql})
              AND l.labResult IS NOT NULL
            GROUP BY 1,2,3
        )
        SELECT
            labName,
            COUNT(*) AS same_time_groups,
            SUM(CASE WHEN n_rows>1 THEN 1 ELSE 0 END) AS duplicate_groups,
            SUM(CASE WHEN n_rows>1 AND min_v<>max_v THEN 1 ELSE 0 END)
                AS conflicting_duplicate_groups,
            MAX(n_rows) AS max_rows_same_time
        FROM g
        GROUP BY 1
        ORDER BY duplicate_groups DESC, same_time_groups DESC
    """).df()
    lab_dups = normalize_id_columns(lab_dups)
    lab_dups.insert(
        0, "clinical_construct",
        lab_dups["labName"].map(lab_name_to_construct)
    )
    lab_dups.to_csv(out / "LAB_SAME_TIME_DUPLICATE_AUDIT.csv", index=False)

    # ------------------------------------------------------------------
    # 7) Vasoactive mapping audit.
    # Positive-rate rows are distinguished from zero/unparseable rows.
    # This audit does NOT declare absence of a row to mean no treatment.
    # ------------------------------------------------------------------
    ID = tables["infusionDrug"]
    vaso = con.execute(f"""
        WITH x AS (
            SELECT
                i.patientUnitStayID,
                b.hospitalID,
                i.infusionOffset,
                i.drugName,
                TRY_CAST(i.drugRate AS DOUBLE) AS drug_rate_num,
                TRY_CAST(i.infusionRate AS DOUBLE) AS infusion_rate_num,
                CASE
                    WHEN regexp_matches(lower(i.drugName), 'norepinephrine|noradrenaline|levophed')
                        THEN 'norepinephrine'
                    WHEN regexp_matches(lower(i.drugName), 'epinephrine|adrenaline')
                        THEN 'epinephrine'
                    WHEN regexp_matches(lower(i.drugName), 'phenylephrine|neosynephrine|neo-synephrine')
                        THEN 'phenylephrine'
                    WHEN regexp_matches(lower(i.drugName), 'vasopressin')
                        THEN 'vasopressin'
                    WHEN regexp_matches(lower(i.drugName), 'dopamine')
                        THEN 'dopamine'
                    WHEN regexp_matches(lower(i.drugName), 'dobutamine')
                        THEN 'dobutamine'
                    WHEN regexp_matches(lower(i.drugName), 'milrinone')
                        THEN 'milrinone'
                    ELSE NULL
                END AS drug_class
            FROM read_csv_auto('{qpath(ID)}', header=true, sample_size=200000) i
            JOIN base b USING (patientUnitStayID)
            WHERE i.infusionOffset BETWEEN -1440 AND 1440
        )
        SELECT * FROM x WHERE drug_class IS NOT NULL
    """).df()
    vaso = normalize_id_columns(vaso)

    vaso["in_0_24h"] = vaso["infusionOffset"].between(0, 1440)
    vaso["positive_numeric_rate"] = (
        (pd.to_numeric(vaso["drug_rate_num"], errors="coerce") > 0)
        | (pd.to_numeric(vaso["infusion_rate_num"], errors="coerce") > 0)
    )
    both_missing = (
        pd.to_numeric(vaso["drug_rate_num"], errors="coerce").isna()
        & pd.to_numeric(vaso["infusion_rate_num"], errors="coerce").isna()
    )
    vaso["no_numeric_rate"] = both_missing
    vaso["zero_or_nonpositive_numeric_rate"] = (
        ~both_missing & ~vaso["positive_numeric_rate"]
    )

    vaso_name_rows = []
    for (drug_class, drug_name), g in vaso.groupby(["drug_class", "drugName"], dropna=False):
        w = g.loc[g["in_0_24h"]]
        vaso_name_rows.append({
            "drug_class": drug_class,
            "drugName": drug_name,
            "rows_minus24_to_plus24h": int(len(g)),
            "rows_0_24h": int(len(w)),
            "stays_0_24h": int(w["patientUnitStayID"].nunique()),
            "hospitals_0_24h": int(w["hospitalID"].nunique()),
            "positive_numeric_rate_rows_0_24h": int(w["positive_numeric_rate"].sum()),
            "zero_or_nonpositive_numeric_rate_rows_0_24h": int(
                w["zero_or_nonpositive_numeric_rate"].sum()
            ),
            "no_numeric_rate_rows_0_24h": int(w["no_numeric_rate"].sum()),
            "min_offset": int(g["infusionOffset"].min()) if len(g) else None,
            "max_offset": int(g["infusionOffset"].max()) if len(g) else None,
            "negative_offset_rows": int((g["infusionOffset"] < 0).sum()),
        })
    pd.DataFrame(vaso_name_rows).sort_values(
        ["drug_class", "stays_0_24h", "rows_0_24h"],
        ascending=[True, False, False],
    ).to_csv(out / "VASOACTIVE_LABEL_RATE_AUDIT.csv", index=False)

    # Hospital/site capture, with treatment table used only as additional
    # positive evidence of documentation. Treatment absence is never negative.
    T = tables["treatment"]
    any_inf = con.execute(f"""
        SELECT DISTINCT i.patientUnitStayID
        FROM read_csv_auto('{qpath(ID)}', header=true, sample_size=200000) i
        JOIN base b USING (patientUnitStayID)
        WHERE i.infusionOffset BETWEEN 0 AND 1440
    """).df()
    any_inf = normalize_id_columns(any_inf)

    target_rows = vaso.loc[vaso["in_0_24h"], ["patientUnitStayID", "hospitalID"]].drop_duplicates()
    target_positive = vaso.loc[
        vaso["in_0_24h"] & vaso["positive_numeric_rate"],
        ["patientUnitStayID", "hospitalID"]
    ].drop_duplicates()

    treatment_pos = con.execute(f"""
        SELECT DISTINCT
            t.patientUnitStayID,
            b.hospitalID
        FROM read_csv_auto('{qpath(T)}', header=true, sample_size=200000) t
        JOIN base b USING (patientUnitStayID)
        WHERE t.treatmentOffset BETWEEN 0 AND 1440
          AND regexp_matches(
                lower(t.treatmentString),
                'vasopressor|norepinephrine|noradrenaline|levophed|epinephrine|adrenaline|phenylephrine|neosynephrine|vasopressin|dopamine|dobutamine|milrinone'
              )
    """).df()
    treatment_pos = normalize_id_columns(treatment_pos)

    base_h = base.groupby("hospitalID").size().rename("landmark_stays").reset_index()

    def hosp_count(df, name):
        return (
            df.groupby("hospitalID")["patientUnitStayID"]
            .nunique()
            .rename(name)
            .reset_index()
        )

    any_inf_h = base.merge(
        any_inf.assign(any_inf=1), on="patientUnitStayID", how="left"
    )
    any_inf_h = (
        any_inf_h.groupby("hospitalID")["any_inf"]
        .apply(lambda s: int(s.notna().sum()))
        .rename("stays_with_any_infusion_row")
        .reset_index()
    )

    site = base_h.merge(any_inf_h, on="hospitalID", how="left")
    site = site.merge(
        hosp_count(target_rows, "stays_with_target_name_row"),
        on="hospitalID", how="left"
    )
    site = site.merge(
        hosp_count(target_positive, "stays_with_positive_numeric_target_rate"),
        on="hospitalID", how="left"
    )
    site = site.merge(
        hosp_count(treatment_pos, "stays_with_target_treatment_positive_evidence"),
        on="hospitalID", how="left"
    )
    for c in [
        "stays_with_any_infusion_row",
        "stays_with_target_name_row",
        "stays_with_positive_numeric_target_rate",
        "stays_with_target_treatment_positive_evidence",
    ]:
        site[c] = site[c].fillna(0).astype(int)
        site[c.replace("stays_with_", "") + "_pct"] = 100.0 * site[c] / site["landmark_stays"]

    site.to_csv(out / "VASOACTIVE_SITE_CAPTURE_AUDIT.csv", index=False)

    # ------------------------------------------------------------------
    # 8) Compact executive evidence table.
    # This is descriptive only; no GREEN/AMBER/RED status is assigned here.
    # ------------------------------------------------------------------
    summary = []
    for _, r in pd.read_csv(out / "VITAL_SOURCE_DISTRIBUTION_SUMMARY.csv").iterrows():
        summary.append({
            "domain": "vital",
            "construct": r["variable"],
            "candidate_source": r["source"],
            "available_stays": int(r["available_stays"]),
            "available_pct": float(r["available_pct"]),
            "note": "descriptive source-comparison evidence only",
        })

    gcs_read = pd.read_csv(out / "GCS_REPRESENTATION_AUDIT.csv")
    for _, r in gcs_read.loc[
        gcs_read["representation"].isin(["direct_total", "component_sum"])
    ].iterrows():
        summary.append({
            "domain": "gcs",
            "construct": "gcs_min_0_24h",
            "candidate_source": str(r["representation"]),
            "available_stays": int(r["available_stays"]),
            "available_pct": float(r["available_pct"]),
            "note": "representation audit; no mapping fixed",
        })

    lab_read = pd.read_csv(out / "LAB_TARGET_DISTRIBUTION_SUMMARY.csv")
    for _, r in lab_read.iterrows():
        summary.append({
            "domain": "lab",
            "construct": str(r["clinical_construct"]),
            "candidate_source": str(r["labName"]),
            "available_stays": int(r["available_stays"]),
            "available_pct": float(r["available_pct"]),
            "note": "candidate label kept separate; no silent merging",
        })

    pd.DataFrame(summary).to_csv(out / "PREDICTOR_MEASUREMENT_EVIDENCE_SUMMARY.csv", index=False)

    # ------------------------------------------------------------------
    # 9) Safety + manifest.
    # ------------------------------------------------------------------
    for path in out.glob("*.csv"):
        df0 = pd.read_csv(path, nrows=5)
        assert_safe_columns(df0.columns, path.name)

    output_hashes = {
        p.name: sha256_file(p)
        for p in sorted(out.glob("*.csv"), key=lambda x: x.name)
    }
    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "outcome-blind predictor measurement evidence for eICU harmonization",
        "raw_data_modified": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "hospitalDischargeStatus_selected": False,
        "unitDischargeStatus_selected": False,
        "adult_plus24h_stays": n_base,
        "hospitals": int(base["hospitalID"].nunique()),
        "required_tables": REQUIRED_TABLES,
        "source_files": {k: v.name for k, v in tables.items()},
        "generated_sha256": output_hashes,
        "interpretation": (
            "Descriptive outcome-blind measurement evidence only. "
            "No harmonization mapping, source precedence, site threshold, "
            "absence=zero rule, or GREEN/AMBER/RED status is fixed by this audit."
        ),
    }
    (out / "MEASUREMENT_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("=" * 78)
    print("eICU PREDICTOR MEASUREMENT AUDIT: PASS")
    print(f"Adult +24 h stays: {n_base:,}")
    print(f"Hospitals: {base['hospitalID'].nunique():,}")
    print(f"Output: {out}")
    print("Outcome/discharge-status values selected: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("Harmonization status fixed by this script: NO")
    print("=" * 78)


if __name__ == "__main__":
    main()
