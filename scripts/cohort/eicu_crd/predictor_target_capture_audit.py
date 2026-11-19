#!/usr/bin/env python3
"""
Outcome-blind eICU predictor-target capture adjudication audit.

Purpose
-------
Distinguish generic source-interface presence from target-specific availability
for routine primary predictors after the site-availability gate.

This audit asks whether a hospital that has, for example, nurseCharting or lab
rows at all also has at least one usable released row for each exact fixed
primary predictor target.

It is intentionally limited to routine predictors for which absence is a
missing measurement rather than a clinical negative:
- HR, MAP, respiratory rate, temperature, SpO2, GCS;
- Hgb, WBC, platelets, BUN, bicarbonate, sodium, potassium, glucose.

It does NOT require a positive vasoactive or RRT event, because those are
event-driven constructs. Creatinine is already an explicit target-specific
requirement in the fixed site rule.

No threshold or site exclusion is selected by this script.
No outcome fields are read.
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

TARGETS = [
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

    # ------------------------------------------------------------------
    # 1) Outcome-safe patient skeleton and adult +24 h denominator.
    # ------------------------------------------------------------------
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
    hosp_off = pd.to_numeric(
        patient["hospital_discharge_offset"], errors="coerce"
    )

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

    # ------------------------------------------------------------------
    # 2) Rebuild fixed source-complete site rule.
    # ------------------------------------------------------------------
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
    site_eligible = set.intersection(*site_sets.values())

    # ------------------------------------------------------------------
    # 3) Fixed HF proxy; retain site-eligible audit denominator.
    # ------------------------------------------------------------------
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
    hf_site = hf.loc[hf["hospital_id"].isin(site_eligible)].copy()

    con.register("hf_site_df", hf_site)
    con.execute("CREATE TEMP TABLE hf_site AS SELECT * FROM hf_site_df")

    # ------------------------------------------------------------------
    # 4) Hospital-wide target-specific usable-row presence.
    # ------------------------------------------------------------------
    # nurseCharting targets
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

    hospital_target_sets = {}
    for target, mask in nurse_masks.items():
        hospital_target_sets[target] = set(
            nurse_all.loc[mask, "hospital_id"].unique()
        )

    # lab targets
    lab_names = sorted({v[0] for v in LAB_SPECS.values()})
    names_sql = ",".join("'" + x.replace("'", "''") + "'" for x in lab_names)

    lab_all = con.execute(f"""
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
            lab_all["lab_name"].eq(name)
            & pd.to_numeric(
                lab_all["lab_type_id"], errors="coerce"
            ).eq(lab_type)
            & lab_all["v"].gt(lo)
            & lab_all["v"].le(hi)
        )
        hospital_target_sets[target] = set(
            lab_all.loc[mask, "hospital_id"].unique()
        )

    # ------------------------------------------------------------------
    # 5) Timing-valid target capture within adult +24 h and HF +24 h.
    # ------------------------------------------------------------------
    def timing_valid_target_sets(base_table: str):
        if base_table not in {"adult", "hf_site"}:
            raise ValueError(base_table)

        nurse = con.execute(f"""
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
            JOIN {base_table} b ON n.patientUnitStayID=b.stay_id
            WHERE n.nursingChartOffset BETWEEN 0 AND 1440
              AND n.nursingChartEntryOffset<=1440
              AND n.nursingChartCellTypeCat IN
                  ('Vital Signs','Scores','Other Vital Signs and Infusions')
        """).df()

        nm = {
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

        result = {}
        for target, mask in nm.items():
            result[target] = set(nurse.loc[mask, "stay_id"].unique())

        lab = con.execute(f"""
            SELECT
                l.patientUnitStayID AS stay_id,
                l.labName AS lab_name,
                l.labTypeID AS lab_type_id,
                TRY_CAST(l.labResult AS DOUBLE) AS v
            FROM read_csv_auto(
                '{qpath(LAB)}',
                header=true,
                sample_size=200000
            ) l
            JOIN {base_table} b ON l.patientUnitStayID=b.stay_id
            WHERE l.labResultOffset BETWEEN 0 AND 1440
              AND l.labResultRevisedOffset<=1440
              AND l.labName IN ({names_sql})
        """).df()

        for target, (name, lab_type, lo, hi) in LAB_SPECS.items():
            mask = (
                lab["lab_name"].eq(name)
                & pd.to_numeric(
                    lab["lab_type_id"], errors="coerce"
                ).eq(lab_type)
                & lab["v"].gt(lo)
                & lab["v"].le(hi)
            )
            result[target] = set(lab.loc[mask, "stay_id"].unique())

        return result

    adult_target_sets = timing_valid_target_sets("adult")
    hf_target_sets = timing_valid_target_sets("hf_site")

    # ------------------------------------------------------------------
    # 6) Hospital-level matrix restricted to fixed site-eligible hospitals.
    # ------------------------------------------------------------------
    eligible_hospitals = sorted(site_eligible)
    matrix = pd.DataFrame({"hospital_id": eligible_hospitals})

    adult_counts = (
        adult.loc[adult["hospital_id"].isin(site_eligible)]
        .groupby("hospital_id")
        .size()
        .rename("adult_plus24h_stays")
        .reset_index()
    )
    hf_counts = (
        hf_site.groupby("hospital_id")
        .size()
        .rename("hf_plus24h_stays")
        .reset_index()
    )

    matrix = matrix.merge(adult_counts, on="hospital_id", how="left")
    matrix = matrix.merge(hf_counts, on="hospital_id", how="left")
    matrix["adult_plus24h_stays"] = (
        matrix["adult_plus24h_stays"].fillna(0).astype(int)
    )
    matrix["hf_plus24h_stays"] = (
        matrix["hf_plus24h_stays"].fillna(0).astype(int)
    )

    for target in TARGETS:
        matrix[f"{target}__hospital_wide_present"] = (
            matrix["hospital_id"].isin(hospital_target_sets[target]).astype(int)
        )

        # adult timing-valid capture count
        ad = adult.loc[adult["hospital_id"].isin(site_eligible), [
            "stay_id", "hospital_id"
        ]].copy()
        ad["present"] = ad["stay_id"].isin(adult_target_sets[target]).astype(int)
        adg = (
            ad.groupby("hospital_id")["present"]
            .sum()
            .rename(f"{target}__adult_plus24h_present_stays")
            .reset_index()
        )
        matrix = matrix.merge(adg, on="hospital_id", how="left")
        c = f"{target}__adult_plus24h_present_stays"
        matrix[c] = matrix[c].fillna(0).astype(int)
        matrix[f"{target}__adult_plus24h_pct"] = (
            100.0 * matrix[c] / matrix["adult_plus24h_stays"]
        )

        # HF timing-valid capture count
        htmp = hf_site[["stay_id", "hospital_id"]].copy()
        htmp["present"] = htmp["stay_id"].isin(hf_target_sets[target]).astype(int)
        hg = (
            htmp.groupby("hospital_id")["present"]
            .sum()
            .rename(f"{target}__hf_plus24h_present_stays")
            .reset_index()
        )
        matrix = matrix.merge(hg, on="hospital_id", how="left")
        hc = f"{target}__hf_plus24h_present_stays"
        matrix[hc] = matrix[hc].fillna(0).astype(int)
        matrix[f"{target}__hf_plus24h_pct"] = matrix.apply(
            lambda r: (
                100.0 * r[hc] / r["hf_plus24h_stays"]
                if r["hf_plus24h_stays"] > 0 else None
            ),
            axis=1,
        )

    matrix.to_csv(
        out / "TARGET_SPECIFIC_CAPTURE_BY_SITE_ELIGIBLE_HOSPITAL.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 7) Summary of structural target absence.
    # ------------------------------------------------------------------
    rows = []
    for target in TARGETS:
        absent = matrix[f"{target}__hospital_wide_present"].eq(0)
        adult_zero = matrix[f"{target}__adult_plus24h_present_stays"].eq(0)
        hf_contrib = matrix["hf_plus24h_stays"].gt(0)
        hf_zero = (
            matrix[f"{target}__hf_plus24h_present_stays"].eq(0)
            & hf_contrib
        )

        rows.append({
            "model_term": target,
            "site_eligible_hospitals": len(matrix),
            "hospital_wide_target_absent_hospitals": int(absent.sum()),
            "adult_plus24h_stays_in_hospital_wide_target_absent_hospitals": int(
                matrix.loc[absent, "adult_plus24h_stays"].sum()
            ),
            "hf_plus24h_stays_in_hospital_wide_target_absent_hospitals": int(
                matrix.loc[absent, "hf_plus24h_stays"].sum()
            ),
            "adult_plus24h_zero_target_capture_hospitals": int(adult_zero.sum()),
            "adult_zero_but_hospital_wide_target_present_hospitals": int(
                (adult_zero & ~absent).sum()
            ),
            "hf_contributing_hospitals_zero_target_capture": int(hf_zero.sum()),
            "hf_zero_but_hospital_wide_target_present_hospitals": int(
                (hf_zero & ~absent).sum()
            ),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(
        out / "TARGET_SPECIFIC_STRUCTURAL_ABSENCE_SUMMARY.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 8) Nonbinding all-routine-target-present simulation.
    # ------------------------------------------------------------------
    presence_cols = [
        f"{target}__hospital_wide_present" for target in TARGETS
    ]
    matrix["all_routine_primary_targets_hospital_wide_present"] = (
        matrix[presence_cols].eq(1).all(axis=1)
    )

    sim = pd.DataFrame([{
        "rule": (
            "fixed_source_complete_site_rule_PLUS_hospital_wide_presence_"
            "of_all_14_routine_nonrenal_predictor_targets"
        ),
        "rule_is_fixed": False,
        "current_site_eligible_hospitals": int(len(matrix)),
        "target_complete_hospitals": int(
            matrix["all_routine_primary_targets_hospital_wide_present"].sum()
        ),
        "hospitals_lost": int(
            (~matrix["all_routine_primary_targets_hospital_wide_present"]).sum()
        ),
        "adult_plus24h_stays_retained": int(
            matrix.loc[
                matrix["all_routine_primary_targets_hospital_wide_present"],
                "adult_plus24h_stays",
            ].sum()
        ),
        "adult_plus24h_stays_excluded": int(
            matrix.loc[
                ~matrix["all_routine_primary_targets_hospital_wide_present"],
                "adult_plus24h_stays",
            ].sum()
        ),
        "hf_plus24h_stays_retained": int(
            matrix.loc[
                matrix["all_routine_primary_targets_hospital_wide_present"],
                "hf_plus24h_stays",
            ].sum()
        ),
        "hf_plus24h_stays_excluded": int(
            matrix.loc[
                ~matrix["all_routine_primary_targets_hospital_wide_present"],
                "hf_plus24h_stays",
            ].sum()
        ),
    }])
    sim.to_csv(
        out / "NONBINDING_TARGET_COMPLETE_SITE_SIMULATION.csv",
        index=False,
    )

    # Which target combinations are absent?
    absent_flags = []
    for target in TARGETS:
        col = f"absent__{target}"
        matrix[col] = matrix[f"{target}__hospital_wide_present"].eq(0)
        absent_flags.append(col)

    pattern = (
        matrix.groupby(absent_flags, dropna=False)
        .agg(
            hospitals=("hospital_id", "size"),
            adult_plus24h_stays=("adult_plus24h_stays", "sum"),
            hf_plus24h_stays=("hf_plus24h_stays", "sum"),
        )
        .reset_index()
        .sort_values(
            ["hospitals", "hf_plus24h_stays"],
            ascending=False,
        )
    )
    pattern.to_csv(
        out / "TARGET_STRUCTURAL_ABSENCE_PATTERN_SUMMARY.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 9) Safety + manifest.
    # ------------------------------------------------------------------
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
        "purpose": "outcome-blind predictor-target structural capture adjudication",
        "fixed_source_complete_site_rule_rechecked": True,
        "site_eligible_hospitals_rechecked": int(len(site_eligible)),
        "fixed_hf_plus24h_site_eligible_stays": int(len(hf_site)),
        "routine_targets_audited": TARGETS,
        "event_driven_positive_target_required": False,
        "vasoactive_positive_event_required_for_site_availability": False,
        "rrt_positive_event_required_for_site_availability": False,
        "threshold_selected": False,
        "site_exclusion_applied": False,
        "site_rule_amended_by_script": False,
        "imputation_performed": False,
        "outcome_or_discharge_status_selected": False,
        "discharge_location_selected": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "final_episode_selection_applied": False,
        "final_cohort_membership_fixed": False,
        "generated_sha256": hashes,
        "interpretation": (
            "This audit tests whether generic source-complete hospitals also "
            "contain at least one usable released row for each exact routine "
            "primary predictor target. Any site-rule amendment remains a "
            "separate adjudication decision and is not made by this script."
        ),
    }

    (out / "PREDICTOR_TARGET_CAPTURE_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("EICU PREDICTOR-TARGET CAPTURE ADJUDICATION AUDIT: PASS")
    print(f"Fixed source-complete site-eligible hospitals: {len(site_eligible):,}")
    print(f"Site-eligible fixed HF +24 h audit stays: {len(hf_site):,}")
    print(f"Output: {out}")
    print("Target-specific hospital-wide presence audited: YES")
    print("Threshold selected: NO")
    print("Site exclusion applied: NO")
    print("Site rule amended by script: NO")
    print("Imputation performed: NO")
    print("Outcome/discharge-status values selected: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("Final cohort membership fixed: NO")
    print("=" * 78)

if __name__ == "__main__":
    main()
