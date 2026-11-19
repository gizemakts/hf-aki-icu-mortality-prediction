#!/usr/bin/env python3
"""
Outcome-blind eICU primary-predictor missingness audit.

Purpose
-------
Quantify patient-level missingness for the 17 NON-RENAL columns of the fixed
19-column primary model after applying ONLY already-fixed:
- adult +24 h landmark;
- structured HF proxy;
- amended hospital-wide source-complete + exact-routine-target-complete site rule.

The two renal dummy columns are intentionally not audited as generic missing
predictors here because renal evaluability/AKI positivity is governed by the
separate fixed Cr/RRT phenotype contract and is cohort-defining.

This script does NOT:
- attach outcomes;
- apply final chronic-KRT / Cr-RRT AKI eligibility;
- perform final episode selection;
- impute any value;
- select a missingness threshold;
- exclude patients for predictor missingness;
- fit or evaluate a model.

When a canonical MIMIC primary-model missingness summary is provided through
--mimic-missingness, the script also compares missingness across cohorts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
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

LAB_SPECS = {
    "hemoglobin_min_gdl": {
        "name": "Hgb", "type": 3, "lo": 0.0, "hi": 50.0, "agg": "min"
    },
    "wbc_max_log2": {
        "name": "WBC x 1000", "type": 3, "lo": 0.0, "hi": 1000.0, "agg": "max"
    },
    "platelet_min_log2": {
        "name": "platelets x 1000", "type": 3, "lo": 0.0, "hi": 10000.0, "agg": "min"
    },
    "bun_max_log2": {
        "name": "BUN", "type": 1, "lo": 0.0, "hi": 300.0, "agg": "max"
    },
    "bicarbonate_min_per5": {
        "name": "bicarbonate", "type": 1, "lo": 0.0, "hi": 10000.0, "agg": "min"
    },
    "sodium_max_abs_deviation_from140_per5": {
        "name": "sodium", "type": 1, "lo": 0.0, "hi": 200.0, "agg": "minmax"
    },
    "potassium_max_mmol_l": {
        "name": "potassium", "type": 1, "lo": 0.0, "hi": 30.0, "agg": "max"
    },
    "glucose_max_log2": {
        "name": "glucose", "type": 1, "lo": 0.0, "hi": 10000.0, "agg": "max"
    },
}

ROUTINE_SITE_TARGETS = [
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

NONRENAL_PREDICTORS = [
    "age_per10y",
    "sex_female",
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
    "vasoactive_any_by_landmark",
]

DRUG_PATTERNS = [
    ("norepinephrine", re.compile(
        r"\b(?:norepinephrine|noradrenaline|levophed)\b"
        r"|^nss with levo \(ml/hr\)$"
        r"|^nss w/ levo/vaso \(ml/hr\)$",
        re.I
    )),
    ("epinephrine", re.compile(
        r"\b(?:epinephrine|epinepherine|adrenalin|adrenaline)\b"
        r"|^epi \(mcg/min\)$",
        re.I
    )),
    ("phenylephrine", re.compile(
        r"\bphenylephrine\b"
        r"|\bneo[- ]?synephrine\b"
        r"|\bneosynephrine\b"
        r"|\bneosynsprine\b",
        re.I
    )),
    ("vasopressin", re.compile(r"\bvasopressin\b", re.I)),
    ("dopamine", re.compile(r"\bdopamine\b", re.I)),
    ("dobutamine", re.compile(r"\bdobu\w*\b", re.I)),
    ("milrinone", re.compile(
        r"\b(?:milrinone|milronone|primacor|primacore)\b",
        re.I
    )),
]

def discover(root: Path, stem: str) -> Path:
    wanted = {f"{stem}.csv".lower(), f"{stem}.csv.gz".lower()}
    matches = sorted(
        [
            p.resolve()
            for p in root.rglob("*")
            if p.is_file() and p.name.lower() in wanted
        ],
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

def assert_safe_columns(columns, label: str) -> None:
    bad = [
        str(c) for c in columns
        if any(tok in str(c).lower() for tok in PROHIBITED_TOKENS)
    ]
    if bad:
        raise AssertionError(f"{label}: prohibited columns detected: {bad}")

def norm_label(x) -> str:
    s = unicodedata.normalize("NFKC", str(x or ""))
    return re.sub(r"\s+", " ", s.strip().lower())

def classify_drug(x):
    s = norm_label(x)
    hits = [name for name, pat in DRUG_PATTERNS if pat.search(s)]
    if len(hits) > 1:
        raise AssertionError(f"Ambiguous vasoactive label: {x!r} -> {hits}")
    return hits[0] if hits else None

def finite_positive(x) -> bool:
    try:
        v = float(str(x).strip())
        return math.isfinite(v) and v > 0
    except Exception:
        return False

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eicu-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mimic-missingness", required=False)
    args = ap.parse_args()

    import duckdb
    import numpy as np
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

    # ---------------------------------------------------------------
    # 1) Outcome-safe patient skeleton and adult +24 h denominator.
    # ---------------------------------------------------------------
    patient = con.execute(f"""
        SELECT
            patientUnitStayID AS stay_id,
            hospitalID AS hospital_id,
            age AS age,
            gender AS gender,
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
        raise AssertionError("Duplicate patientUnitStayID in patient table")

    def age_for_eligibility(x):
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

    def age_for_model(x):
        if x is None:
            return np.nan
        s = str(x).strip()
        if not s:
            return np.nan
        if s.startswith(">"):
            y = 90.5
        else:
            try:
                y = float(s)
            except Exception:
                return np.nan
        return min(y, 91.0) / 10.0

    patient["age_num_eligibility"] = patient["age"].map(age_for_eligibility)
    unit_off = pd.to_numeric(patient["unit_discharge_offset"], errors="coerce")
    hosp_off = pd.to_numeric(
        patient["hospital_discharge_offset"], errors="coerce"
    )

    adult = patient.loc[
        patient["age_num_eligibility"].fillna(-1).ge(18)
        & unit_off.gt(1440)
        & hosp_off.gt(1440),
        ["stay_id", "hospital_id", "age", "gender"],
    ].copy()

    con.register("patient_ids_df", patient[["stay_id", "hospital_id"]])
    con.execute("CREATE TEMP TABLE patient_ids AS SELECT * FROM patient_ids_df")
    con.register("adult_df", adult)
    con.execute("CREATE TEMP TABLE adult AS SELECT * FROM adult_df")

    # ---------------------------------------------------------------
    # 2) Fixed hospital-wide site-eligibility rule.
    # ---------------------------------------------------------------
    interface_sql = {
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

    site_sets = {
        name: set(con.execute(sql).df()["hospital_id"].unique())
        for name, sql in interface_sql.items()
    }
    source_complete = set.intersection(*site_sets.values())

    # ---------------------------------------------------------------
    # 2A) Amended exact routine predictor-target completeness rule.
    #     Structural target presence is hospital-wide across all released
    #     stays and does not use the +24 h patient-level timing window.
    # ---------------------------------------------------------------
    nurse_all = con.execute(f"""
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
            nurse_all["vallabel"].eq("Heart Rate")
            & nurse_all["valname"].eq("Heart Rate")
            & nurse_all["v"].gt(0) & nurse_all["v"].lt(300)
        ),
        "map_min_per10": (
            (
                nurse_all["vallabel"].eq("Non-Invasive BP")
                & nurse_all["valname"].eq("Non-Invasive BP Mean")
            )
            | (
                nurse_all["vallabel"].eq("Invasive BP")
                & nurse_all["valname"].eq("Invasive BP Mean")
            )
            | (
                nurse_all["vallabel"].eq("MAP (mmHg)")
                & nurse_all["valname"].eq("Value")
            )
            | (
                nurse_all["vallabel"].eq("Arterial Line MAP (mmHg)")
                & nurse_all["valname"].eq("Value")
            )
        ) & nurse_all["v"].gt(0) & nurse_all["v"].lt(300),
        "resp_rate_max_per5": (
            nurse_all["vallabel"].eq("Respiratory Rate")
            & nurse_all["valname"].eq("Respiratory Rate")
            & nurse_all["v"].gt(0) & nurse_all["v"].lt(70)
        ),
        "temperature_max_abs_deviation_from_37c": (
            (
                nurse_all["vallabel"].eq("Temperature")
                & nurse_all["valname"].eq("Temperature (C)")
                & nurse_all["v"].gt(10) & nurse_all["v"].lt(50)
            )
            | (
                nurse_all["vallabel"].eq("Temperature")
                & nurse_all["valname"].eq("Temperature (F)")
                & nurse_all["v"].gt(70) & nurse_all["v"].lt(120)
            )
        ),
        "spo2_deficit_per5": (
            (
                nurse_all["vallabel"].eq("O2 Saturation")
                & nurse_all["valname"].eq("O2 Saturation")
            )
            | (
                nurse_all["vallabel"].eq("SpO2")
                & nurse_all["valname"].eq("Value")
            )
        ) & nurse_all["v"].gt(0) & nurse_all["v"].le(100),
        "gcs_deficit_from15": (
            (
                nurse_all["vallabel"].eq("Glasgow coma score")
                & nurse_all["valname"].eq("GCS Total")
            )
            | (
                nurse_all["vallabel"].eq("Score (Glasgow Coma Scale)")
                & nurse_all["valname"].eq("Value")
            )
        ) & nurse_all["v"].between(3, 15, inclusive="both"),
    }

    target_sets = {
        target: set(nurse_all.loc[mask, "hospital_id"].unique())
        for target, mask in nurse_masks.items()
    }

    site_lab_names = sorted({spec["name"] for spec in LAB_SPECS.values()})
    site_names_sql = ",".join(
        "'" + name.replace("'", "''") + "'" for name in site_lab_names
    )
    site_lab = con.execute(f"""
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
        WHERE l.labName IN ({site_names_sql})
    """).df()

    for target, spec in LAB_SPECS.items():
        mask = (
            site_lab["lab_name"].eq(spec["name"])
            & pd.to_numeric(
                site_lab["lab_type_id"], errors="coerce"
            ).eq(spec["type"])
            & site_lab["v"].gt(spec["lo"])
            & site_lab["v"].le(spec["hi"])
        )
        target_sets[target] = set(
            site_lab.loc[mask, "hospital_id"].unique()
        )

    target_complete = set.intersection(
        *[target_sets[target] for target in ROUTINE_SITE_TARGETS]
    )
    site_eligible = source_complete & target_complete

    if len(source_complete) != 152:
        raise AssertionError(
            f"Source-complete hospital count drift: expected 152, "
            f"observed {len(source_complete)}"
        )
    if len(site_eligible) != 145:
        raise AssertionError(
            f"Amended site-eligible hospital count drift: expected 145, "
            f"observed {len(site_eligible)}"
        )

    site_qc = pd.DataFrame([{
        "all_released_hospitals": int(patient["hospital_id"].nunique()),
        "source_complete_hospitals": int(len(source_complete)),
        "site_eligible_hospitals": int(len(site_eligible)),
        "site_rule": (
            "hospital-wide source completeness PLUS hospital-wide usable "
            "presence of all 14 exact routine nonrenal primary predictor targets"
        ),
        "required_exact_routine_targets": ";".join(ROUTINE_SITE_TARGETS),
        "percentage_capture_threshold_used": False,
    }])
    site_qc.to_csv(out / "SITE_RULE_RECHECK.csv", index=False)

    # ---------------------------------------------------------------
    # 3) Fixed HF proxy, then site restriction.
    # ---------------------------------------------------------------
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

    cohort = adult.merge(hf_ids.assign(hf_proxy=1), on="stay_id", how="inner")
    cohort = cohort.loc[cohort["hospital_id"].isin(site_eligible)].copy()

    if len(cohort) != 15034:
        raise AssertionError(
            f"Amended HF +24 h audit-stay count drift: expected 15,034, "
            f"observed {len(cohort):,}"
        )

    if cohort["stay_id"].duplicated().any():
        raise AssertionError("Duplicate stay in audit cohort")

    con.register("cohort_df", cohort)
    con.execute("CREATE TEMP TABLE cohort AS SELECT * FROM cohort_df")

    # ---------------------------------------------------------------
    # 4) Demographics.
    # ---------------------------------------------------------------
    pred = cohort[["stay_id", "hospital_id"]].copy()
    pred["age_per10y"] = cohort["age"].map(age_for_model)

    gender = cohort["gender"].fillna("").astype(str).str.strip().str.lower()
    pred["sex_female"] = np.where(
        gender.eq("female"), 1.0,
        np.where(gender.eq("male"), 0.0, np.nan)
    )

    # ---------------------------------------------------------------
    # 5) Fixed nurseCharting vitals.
    # ---------------------------------------------------------------
    vitals = con.execute(f"""
        WITH x AS (
            SELECT
                n.patientUnitStayID AS stay_id,
                n.nursingChartCellTypeValLabel AS vallabel,
                n.nursingChartCellTypeValName AS valname,
                TRY_CAST(n.nursingChartValue AS DOUBLE) AS v
            FROM read_csv_auto(
                '{qpath(NURSE)}',
                header=true,
                sample_size=200000
            ) n
            JOIN cohort c ON n.patientUnitStayID=c.stay_id
            WHERE n.nursingChartOffset BETWEEN 0 AND 1440
              AND n.nursingChartEntryOffset<=1440
              AND n.nursingChartCellTypeCat IN
                  ('Vital Signs','Scores','Other Vital Signs and Infusions')
        )
        SELECT
            stay_id,
            MAX(CASE
                WHEN vallabel='Heart Rate'
                 AND valname='Heart Rate'
                 AND v>0 AND v<300 THEN v END
            ) AS hr_max,
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
            END) AS map_min,
            MAX(CASE
                WHEN vallabel='Respiratory Rate'
                 AND valname='Respiratory Rate'
                 AND v>0 AND v<70 THEN v END
            ) AS rr_max,
            MIN(CASE
                WHEN vallabel='Temperature'
                 AND valname='Temperature (C)'
                 AND v>10 AND v<50 THEN v
                WHEN vallabel='Temperature'
                 AND valname='Temperature (F)'
                 AND v>70 AND v<120 THEN (v-32.0)/1.8
            END) AS temp_min_c,
            MAX(CASE
                WHEN vallabel='Temperature'
                 AND valname='Temperature (C)'
                 AND v>10 AND v<50 THEN v
                WHEN vallabel='Temperature'
                 AND valname='Temperature (F)'
                 AND v>70 AND v<120 THEN (v-32.0)/1.8
            END) AS temp_max_c,
            MIN(CASE
                WHEN vallabel='O2 Saturation'
                 AND valname='O2 Saturation'
                 AND v>0 AND v<=100 THEN v
                WHEN vallabel='SpO2'
                 AND valname='Value'
                 AND v>0 AND v<=100 THEN v
            END) AS spo2_min
        FROM x
        GROUP BY stay_id
    """).df()

    pred = pred.merge(vitals, on="stay_id", how="left", validate="1:1")
    pred["heart_rate_max_per10"] = pred["hr_max"] / 10.0
    pred["map_min_per10"] = pred["map_min"] / 10.0
    pred["resp_rate_max_per5"] = pred["rr_max"] / 5.0
    pred["temperature_max_abs_deviation_from_37c"] = np.maximum(
        (pred["temp_min_c"] - 37.0).abs(),
        (pred["temp_max_c"] - 37.0).abs(),
    )
    pred.loc[
        pred[["temp_min_c", "temp_max_c"]].isna().all(axis=1),
        "temperature_max_abs_deviation_from_37c",
    ] = np.nan
    pred["spo2_deficit_per5"] = (100.0 - pred["spo2_min"]) / 5.0

    pred = pred.drop(
        columns=[
            "hr_max", "map_min", "rr_max",
            "temp_min_c", "temp_max_c", "spo2_min",
        ]
    )

    # ---------------------------------------------------------------
    # 6) Fixed direct-total GCS.
    # ---------------------------------------------------------------
    gcs = con.execute(f"""
        WITH x AS (
            SELECT
                n.patientUnitStayID AS stay_id,
                n.nursingChartOffset AS chartoffset,
                n.nursingChartCellTypeValLabel AS vallabel,
                n.nursingChartCellTypeValName AS valname,
                TRY_CAST(n.nursingChartValue AS DOUBLE) AS v
            FROM read_csv_auto(
                '{qpath(NURSE)}',
                header=true,
                sample_size=200000
            ) n
            JOIN cohort c ON n.patientUnitStayID=c.stay_id
            WHERE n.nursingChartOffset BETWEEN 0 AND 1440
              AND n.nursingChartEntryOffset<=1440
              AND n.nursingChartCellTypeCat IN
                  ('Scores','Other Vital Signs and Infusions')
        ),
        same_offset AS (
            SELECT
                stay_id,
                chartoffset,
                MIN(CASE
                    WHEN vallabel='Glasgow coma score'
                     AND valname='GCS Total'
                     AND v BETWEEN 3 AND 15 THEN v
                    WHEN vallabel='Score (Glasgow Coma Scale)'
                     AND valname='Value'
                     AND v BETWEEN 3 AND 15 THEN v
                END) AS direct_total
            FROM x
            GROUP BY stay_id, chartoffset
        )
        SELECT
            stay_id,
            MIN(direct_total) AS gcs_min
        FROM same_offset
        GROUP BY stay_id
    """).df()

    pred = pred.merge(gcs, on="stay_id", how="left", validate="1:1")
    pred["gcs_deficit_from15"] = 15.0 - pred["gcs_min"]
    pred = pred.drop(columns=["gcs_min"])

    # ---------------------------------------------------------------
    # 7) Fixed laboratory mappings.
    # ---------------------------------------------------------------
    lab_names = sorted({x["name"] for x in LAB_SPECS.values()})
    names_sql = ",".join("'" + x.replace("'", "''") + "'" for x in lab_names)

    lab = con.execute(f"""
        SELECT
            l.patientUnitStayID AS stay_id,
            l.labName AS lab_name,
            l.labTypeID AS lab_type_id,
            TRY_CAST(l.labResult AS DOUBLE) AS value
        FROM read_csv_auto(
            '{qpath(LAB)}',
            header=true,
            sample_size=200000
        ) l
        JOIN cohort c ON l.patientUnitStayID=c.stay_id
        WHERE l.labResultOffset BETWEEN 0 AND 1440
          AND l.labResultRevisedOffset<=1440
          AND l.labName IN ({names_sql})
    """).df()

    for model_col, spec in LAB_SPECS.items():
        x = lab.loc[
            lab["lab_name"].eq(spec["name"])
            & pd.to_numeric(lab["lab_type_id"], errors="coerce").eq(spec["type"])
        ].copy()
        x = x.loc[
            x["value"].gt(spec["lo"]) & x["value"].le(spec["hi"])
        ]

        if spec["agg"] == "min":
            vals = x.groupby("stay_id")["value"].min()
        elif spec["agg"] == "max":
            vals = x.groupby("stay_id")["value"].max()
        elif spec["agg"] == "minmax":
            g = x.groupby("stay_id")["value"].agg(["min", "max"])
            vals = pd.concat(
                [(g["min"] - 140.0).abs(), (g["max"] - 140.0).abs()],
                axis=1,
            ).max(axis=1)
        else:
            raise RuntimeError(spec["agg"])

        if model_col in {"wbc_max_log2", "platelet_min_log2",
                         "bun_max_log2", "glucose_max_log2"}:
            vals = np.log2(vals)
        elif model_col == "bicarbonate_min_per5":
            vals = vals / 5.0
        elif model_col == "sodium_max_abs_deviation_from140_per5":
            vals = vals / 5.0

        pred = pred.merge(
            vals.rename(model_col).reset_index(),
            on="stay_id",
            how="left",
            validate="1:1",
        )

    # ---------------------------------------------------------------
    # 8) Fixed vasoactive positive-evidence rule.
    #    Within a fixed SITE-ELIGIBLE hospital, no qualifying captured
    #    target row is encoded as 0 under the fixed external site contract.
    # ---------------------------------------------------------------
    vaso = con.execute(f"""
        SELECT
            i.patientUnitStayID AS stay_id,
            i.drugName AS drug_name,
            i.drugRate AS drug_rate,
            i.infusionRate AS infusion_rate
        FROM read_csv_auto(
            '{qpath(INF)}',
            header=true,
            sample_size=200000
        ) i
        JOIN cohort c ON i.patientUnitStayID=c.stay_id
        WHERE i.infusionOffset BETWEEN 0 AND 1440
    """).df()

    if len(vaso):
        vaso["drug_class"] = vaso["drug_name"].map(classify_drug)
        vaso["positive_rate"] = [
            finite_positive(a) or finite_positive(b)
            for a, b in zip(vaso["drug_rate"], vaso["infusion_rate"])
        ]
        positive_stays = set(
            vaso.loc[
                vaso["drug_class"].notna() & vaso["positive_rate"],
                "stay_id",
            ].unique()
        )
    else:
        positive_stays = set()

    pred["vasoactive_any_by_landmark"] = (
        pred["stay_id"].isin(positive_stays).astype(float)
    )

    # ---------------------------------------------------------------
    # 9) Missingness outputs.
    # ---------------------------------------------------------------
    if list(pred["stay_id"]) != list(cohort["stay_id"]):
        # Order can change after merges; uniqueness is what matters.
        if set(pred["stay_id"]) != set(cohort["stay_id"]):
            raise AssertionError("Predictor matrix membership drift")
    if pred["stay_id"].duplicated().any():
        raise AssertionError("Predictor matrix stay duplication")

    n = len(pred)
    missing_rows = []
    for col in NONRENAL_PREDICTORS:
        s = pred[col]
        missing_n = int(s.isna().sum())
        missing_rows.append({
            "model_term": col,
            "n": n,
            "missing_n": missing_n,
            "missing_pct": 100.0 * missing_n / n if n else None,
            "nonmissing_n": int(s.notna().sum()),
            "n_unique_nonmissing": int(s.dropna().nunique()),
            "minimum_nonmissing": (
                float(pd.to_numeric(s, errors="coerce").min())
                if s.notna().any() else None
            ),
            "maximum_nonmissing": (
                float(pd.to_numeric(s, errors="coerce").max())
                if s.notna().any() else None
            ),
        })

    miss = pd.DataFrame(missing_rows)
    miss.to_csv(out / "EICU_NONRENAL_PREDICTOR_MISSINGNESS.csv", index=False)

    missing_mask = pred[NONRENAL_PREDICTORS].isna()
    pred["missing_predictor_count"] = missing_mask.sum(axis=1)

    dist = (
        pred.groupby("missing_predictor_count")
        .size()
        .rename("stays")
        .reset_index()
        .sort_values("missing_predictor_count")
    )
    dist["pct"] = 100.0 * dist["stays"] / n
    dist.to_csv(out / "PATIENT_MISSING_PREDICTOR_COUNT_DISTRIBUTION.csv", index=False)

    complete = pd.DataFrame([{
        "audit_population": "site_eligible_fixed_hf_plus24h",
        "stays": n,
        "hospitals_with_candidates": int(pred["hospital_id"].nunique()),
        "nonrenal_predictors_audited": len(NONRENAL_PREDICTORS),
        "complete_case_stays_17_nonrenal": int(
            pred["missing_predictor_count"].eq(0).sum()
        ),
        "complete_case_pct_17_nonrenal": float(
            100.0 * pred["missing_predictor_count"].eq(0).mean()
        ),
        "stays_missing_at_least_1_nonrenal": int(
            pred["missing_predictor_count"].gt(0).sum()
        ),
        "stays_missing_at_least_3_nonrenal": int(
            pred["missing_predictor_count"].ge(3).sum()
        ),
        "stays_missing_at_least_5_nonrenal": int(
            pred["missing_predictor_count"].ge(5).sum()
        ),
        "generic_missingness_exclusion_applied": False,
        "imputation_performed": False,
        "renal_terms_included_in_generic_missingness_audit": False,
    }])
    complete.to_csv(out / "MISSINGNESS_AUDIT_SUMMARY.csv", index=False)

    # Most common exact missingness patterns.
    patt = missing_mask.copy()
    patt.columns = [f"missing__{c}" for c in patt.columns]
    patt = pd.concat(
        [pred[["stay_id", "hospital_id"]].reset_index(drop=True),
         patt.reset_index(drop=True)],
        axis=1,
    )
    flag_cols = [c for c in patt.columns if c.startswith("missing__")]
    patt_summary = (
        patt.groupby(flag_cols, dropna=False)
        .size()
        .rename("stays")
        .reset_index()
        .sort_values("stays", ascending=False)
    )
    patt_summary["pct"] = 100.0 * patt_summary["stays"] / n
    patt_summary.to_csv(
        out / "NONRENAL_MISSINGNESS_PATTERN_SUMMARY.csv",
        index=False,
    )

    # Hospital-level missingness: descriptive QC, not a new site threshold.
    hosp_rows = []
    for hospital_id, g in pred.groupby("hospital_id"):
        for col in NONRENAL_PREDICTORS:
            hosp_rows.append({
                "hospital_id": hospital_id,
                "model_term": col,
                "stays": int(len(g)),
                "missing_n": int(g[col].isna().sum()),
                "missing_pct": float(100.0 * g[col].isna().mean()),
            })
    pd.DataFrame(hosp_rows).to_csv(
        out / "PREDICTOR_MISSINGNESS_BY_HOSPITAL.csv",
        index=False,
    )

    # Variable-level descriptive candidate cut-points only.
    cut_rows = []
    for _, r in miss.iterrows():
        for threshold in [5, 10, 25, 50]:
            cut_rows.append({
                "model_term": r["model_term"],
                "observed_missing_pct": r["missing_pct"],
                "candidate_reporting_threshold_pct": threshold,
                "exceeds_threshold": bool(r["missing_pct"] > threshold),
                "threshold_selected": False,
            })
    pd.DataFrame(cut_rows).to_csv(
        out / "DESCRIPTIVE_MISSINGNESS_THRESHOLD_GRID.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 10) Optional comparison with canonical MIMIC missingness report.
    # ---------------------------------------------------------------
    mimic_info = None
    if args.mimic_missingness:
        mp = Path(args.mimic_missingness).expanduser().resolve()
        if not mp.is_file():
            raise FileNotFoundError(mp)

        mm = pd.read_csv(mp)
        needed = {"model_term", "model_role", "n", "missing_n", "missing_pct"}
        if not needed.issubset(mm.columns):
            raise KeyError(
                f"MIMIC missingness report missing columns: "
                f"{sorted(needed - set(mm.columns))}"
            )
        mmp = mm.loc[
            mm["model_term"].isin(NONRENAL_PREDICTORS)
            & mm["model_role"].eq("PRIMARY_CORE"),
            ["model_term", "n", "missing_n", "missing_pct"],
        ].copy()
        mmp = mmp.rename(columns={
            "n": "mimic_n",
            "missing_n": "mimic_missing_n",
            "missing_pct": "mimic_missing_pct",
        })

        comp = miss[
            ["model_term", "n", "missing_n", "missing_pct"]
        ].rename(columns={
            "n": "eicu_n",
            "missing_n": "eicu_missing_n",
            "missing_pct": "eicu_missing_pct",
        }).merge(mmp, on="model_term", how="left", validate="1:1")

        comp["absolute_missing_pct_difference_eicu_minus_mimic"] = (
            comp["eicu_missing_pct"] - comp["mimic_missing_pct"]
        )
        comp["missingness_ratio_eicu_to_mimic"] = np.where(
            comp["mimic_missing_pct"].gt(0),
            comp["eicu_missing_pct"] / comp["mimic_missing_pct"],
            np.nan,
        )
        comp.to_csv(
            out / "EICU_VS_MIMIC_PREDICTOR_MISSINGNESS.csv",
            index=False,
        )

        mimic_info = {
            "path": str(mp),
            "sha256": sha256_file(mp),
        }

    # ---------------------------------------------------------------
    # 11) Safety + manifest.
    # ---------------------------------------------------------------
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
        "purpose": "outcome-blind external primary-predictor missingness audit",
        "population": "adult +24h, fixed HF proxy, amended 145-site source+target-complete hospitals",
        "audit_stays": int(n),
        "audit_hospitals_with_hf_candidates": int(pred["hospital_id"].nunique()),
        "source_complete_hospitals_rechecked": int(len(source_complete)),
        "site_eligible_hospitals_rechecked": int(len(site_eligible)),
        "amended_site_rule_expected_count_check": True,
        "amended_site_rule_hospital_id_set_sha256_sorted_newline": (
            "c34a413540c99d1bec9ca030e20606582e8e9084ccac802c912fb62bef656786"
        ),
        "nonrenal_predictors_audited": NONRENAL_PREDICTORS,
        "renal_dummy_columns_generic_missingness_audit": False,
        "renal_reason": (
            "renal evaluability/AKI positivity is governed by the fixed "
            "Cr/RRT phenotype contract and is cohort-defining"
        ),
        "imputation_performed": False,
        "missingness_threshold_selected": False,
        "patient_exclusion_for_missing_predictors": False,
        "final_chronic_krt_aki_eligibility_applied": False,
        "final_episode_selection_applied": False,
        "final_cohort_membership_fixed": False,
        "outcome_or_discharge_status_selected": False,
        "discharge_location_selected": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "canonical_mimic_science_changed": False,
        "mimic_missingness_reference": mimic_info,
        "generated_sha256": hashes,
        "interpretation": (
            "Descriptive pre-outcome missingness evidence only. This audit does "
            "not choose complete-case analysis, imputation strategy, failure "
            "threshold, or final cohort membership."
        ),
    }

    (out / "EXTERNAL_MISSINGNESS_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("EICU EXTERNAL PRIMARY-PREDICTOR MISSINGNESS AUDIT: PASS")
    print(f"Source-complete hospitals rechecked: {len(source_complete):,}")
    print(f"Amended source+target-complete hospitals rechecked: {len(site_eligible):,}")
    print(f"Fixed HF +24 h audit stays in eligible hospitals: {n:,}")
    print(f"Hospitals contributing those audit stays: {pred['hospital_id'].nunique():,}")
    print("Renal dummy columns treated as generic missing predictors: NO")
    print("Imputation performed: NO")
    print("Missingness threshold selected: NO")
    print("Patient exclusion for missing predictors: NO")
    print("Final chronic-KRT / AKI eligibility applied: NO")
    print("Final episode selection applied: NO")
    print("Outcome/discharge-status values selected: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("Final cohort membership fixed: NO")
    print("=" * 78)

if __name__ == "__main__":
    main()
