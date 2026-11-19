#!/usr/bin/env python3
"""
Outcome-blind eICU renal adjudication audit.

Purpose
-------
Refine the source audit into prespecified candidate measurement contracts for:
1) rolling creatinine KDIGO transport,
2) definite chronic-KRT evidence,
3) delivered RRT evidence,
4) isolated-ultrafiltration identification.

This is still an EVIDENCE audit, not a phenotype lock.

It deliberately does NOT:
- use mortality/discharge status/discharge location,
- fit a model or calculate performance,
- attach outcomes,
- perform final episode selection,
- lock chronic-KRT exclusion,
- lock acute RRT,
- lock KDIGO,
- use the historical >7-365 d creatinine proxy as if it were equivalent.

The already-fixed HF proxy and +24 h landmark are reused only to define the
stay-level audit population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

# Exact / near-exact chronic-KRT candidate strings observed in the source audit.
PH_HD_VALUE = "renal failure - hemodialysis"
PH_PD_VALUE = "renal failure - peritoneal dialysis"
PH_TRANSPLANT_VALUE = "s/p renal transplant"

RENAL_FAILURE_PATH_PREFIX = (
    "notes/Progress Notes/Past History/Organ Systems/"
    "Renal  (R)/Renal Failure/"
)

CHRONIC_TREATMENT_ALLOWLIST = (
    "renal|dialysis|hemodialysis|for chronic renal failure",
    "renal|dialysis|peritoneal dialysis|for chronic renal failure",
    "renal|dialysis|ultrafiltration (fluid removal only)|for chronic renal failure",
)

ACUTE_RRT_TREATMENT_ALLOWLIST = (
    "renal|dialysis|hemodialysis",
    "renal|dialysis|hemodialysis|emergent",
    "renal|dialysis|C V V H D",
    "renal|dialysis|hemodialysis|for acute renal failure",
    "renal|dialysis|peritoneal dialysis",
    "renal|electrolyte correction|treatment of hyperkalemia|dialysis",
    "renal|electrolyte correction|treatment of hyperphosphatemia|dialysis",
    "renal|dialysis|C V V H",
    "renal|dialysis|ultrafiltration (fluid removal only)",
    "renal|dialysis|ultrafiltration (fluid removal only)|emergent",
    "endocrine|electrolyte correction|treatment of hyperkalemia|dialysis",
    "toxicology|drug overdose|drug removal measures|hemodialysis",
    "renal|dialysis|SLED",
    "renal|dialysis|ultrafiltration (fluid removal only)|for acute renal failure",
)

IO_RRT_PATH_ALLOWLIST = (
    "flowsheet|Flowsheet Cell Labels|I&O|Dialysis (ml)|Out",
    "flowsheet|Flowsheet Cell Labels|I&O|Dialysis (ml)|In",
    "flowsheet|Flowsheet Cell Labels|I&O|Intake (ml)|Generic Intake (ml)|Hemodialysis Intake (mL)",
    "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Hemofiltration",
    'flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|CRRT Actual Pt Fluid Removed ("C")',
    "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|Ultrafiltration",
    "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|HemodialysisOut",
    "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|CRRT Out",
    "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|CRRT- UF removed",
    "flowsheet|Flowsheet Cell Labels|I&O|Output (ml)|CRRT - UF removed",
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
        str(c)
        for c in columns
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
    LAB = discover(root, "lab")
    TREAT = discover(root, "treatment")
    IO = discover(root, "intakeOutput")
    APS = discover(root, "apacheApsVar")
    DX = discover(root, "diagnosis")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    # ------------------------------------------------------------------
    # 1) Fixed stay-level audit population: adult +24 h + structured HF.
    # ------------------------------------------------------------------
    patient = con.execute(f"""
        SELECT
            patientUnitStayID AS stay_id,
            patientHealthSystemStayID AS hstay_id,
            uniquePid AS unique_pid,
            hospitalID AS hospital_id,
            age AS age,
            hospitalAdmitOffset AS hospital_admit_offset,
            unitDischargeOffset AS unit_discharge_offset,
            hospitalDischargeOffset AS hospital_discharge_offset
        FROM read_csv_auto(
            '{qpath(PATIENT)}',
            header=true,
            sample_size=200000
        )
    """).df()
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

    patient["age_num"] = patient["age"].map(age_num)
    unit_off = pd.to_numeric(
        patient["unit_discharge_offset"], errors="coerce"
    )
    hosp_off = pd.to_numeric(
        patient["hospital_discharge_offset"], errors="coerce"
    )
    patient["landmark_plus24"] = (
        patient["age_num"].fillna(-1).ge(18)
        & unit_off.gt(1440)
        & hosp_off.gt(1440)
    )

    base = patient.loc[
        patient["landmark_plus24"],
        [
            "stay_id",
            "hstay_id",
            "unique_pid",
            "hospital_id",
            "hospital_admit_offset",
        ],
    ].copy()

    con.register(
        "base_ids",
        base[["stay_id"]].rename(columns={"stay_id": "patientUnitStayID"}),
    )
    values_sql = ",".join("'" + v.replace("'", "''") + "'" for v in HF_VALUES)

    hf = con.execute(f"""
        SELECT DISTINCT ph.patientUnitStayID AS stay_id
        FROM read_csv_auto(
            '{qpath(PH)}',
            header=true,
            sample_size=200000
        ) ph
        JOIN base_ids b
          ON ph.patientUnitStayID=b.patientUnitStayID
        WHERE ph.pastHistoryOffset <= 1440
          AND ph.pastHistoryEnteredOffset <= 1440
          AND ph.pastHistoryValue IN ({values_sql})
          AND ph.pastHistoryValueText=ph.pastHistoryValue
          AND ph.pastHistoryPath=
              '{HF_PATH_PREFIX.replace("'", "''")}' || ph.pastHistoryValue
    """).df()

    cand = base.merge(hf.assign(hf_proxy=1), on="stay_id", how="inner")
    if cand["stay_id"].duplicated().any():
        raise AssertionError("HF candidate duplicated")

    n = int(len(cand))
    nh = int(cand["hospital_id"].nunique())
    con.register("cand_df", cand)
    con.execute("CREATE TEMP TABLE cand AS SELECT * FROM cand_df")

    # ------------------------------------------------------------------
    # 2) Creatinine rolling transport candidate.
    #
    # Mirrors the FIXED MIMIC rolling component:
    # - creatinine mg/dL, 0 < value <= 150
    # - same-time AVG
    # - prior minimum within 48 h and 7 d
    # - current events 0..1440
    # - revised-entry offset <=1440
    #
    # The >7-365 d strict-outpatient MIMIC proxy is NOT reconstructed here.
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE TEMP TABLE cr_raw AS
        SELECT
            l.patientUnitStayID AS stay_id,
            c.hospital_id AS hospital_id,
            TRY_CAST(c.hospital_admit_offset AS DOUBLE)
                AS hospital_admit_offset,
            TRY_CAST(l.labResultOffset AS DOUBLE) AS result_offset,
            TRY_CAST(l.labResultRevisedOffset AS DOUBLE) AS revised_offset,
            TRY_CAST(l.labResult AS DOUBLE) AS creatinine
        FROM read_csv_auto(
            '{qpath(LAB)}',
            header=true,
            sample_size=200000
        ) l
        JOIN cand c
          ON l.patientUnitStayID=c.stay_id
        WHERE lower(l.labName)='creatinine'
          AND l.labTypeID=1
          AND TRY_CAST(l.labResult AS DOUBLE)>0
          AND TRY_CAST(l.labResult AS DOUBLE)<=150
          AND TRY_CAST(l.labResultRevisedOffset AS DOUBLE)<=1440
    """)

    # Same-minute aggregation exactly parallels same-charttime AVG in MIMIC.
    con.execute("""
        CREATE TEMP TABLE cr_same_offset AS
        SELECT
            stay_id,
            hospital_id,
            hospital_admit_offset,
            result_offset,
            AVG(creatinine) AS cr_avg,
            MIN(creatinine) AS cr_min,
            MAX(creatinine) AS cr_max,
            COUNT(*) AS raw_rows,
            COUNT(DISTINCT creatinine) AS distinct_values
        FROM cr_raw
        GROUP BY
            stay_id, hospital_id, hospital_admit_offset, result_offset
    """)

    con.execute("""
        CREATE TEMP TABLE cr_rolling AS
        SELECT
            *,
            MIN(cr_avg) OVER (
                PARTITION BY stay_id
                ORDER BY result_offset
                RANGE BETWEEN 2880 PRECEDING AND 1 PRECEDING
            ) AS low48_all,
            MIN(cr_avg) OVER (
                PARTITION BY stay_id
                ORDER BY result_offset
                RANGE BETWEEN 10080 PRECEDING AND 1 PRECEDING
            ) AS low7d_all,
            MIN(
                CASE
                    WHEN result_offset>=hospital_admit_offset
                    THEN cr_avg
                END
            ) OVER (
                PARTITION BY stay_id
                ORDER BY result_offset
                RANGE BETWEEN 2880 PRECEDING AND 1 PRECEDING
            ) AS low48_current_hospital,
            MIN(
                CASE
                    WHEN result_offset>=hospital_admit_offset
                    THEN cr_avg
                END
            ) OVER (
                PARTITION BY stay_id
                ORDER BY result_offset
                RANGE BETWEEN 10080 PRECEDING AND 1 PRECEDING
            ) AS low7d_current_hospital
        FROM cr_same_offset
        WHERE result_offset>=-525600
          AND result_offset<=1440
    """)

    con.execute("""
        CREATE TEMP TABLE cr_events AS
        SELECT
            *,
            CASE
                WHEN low48_all IS NULL AND low7d_all IS NULL THEN NULL
                WHEN low7d_all IS NOT NULL
                     AND cr_avg>=3.0*low7d_all THEN 3
                WHEN cr_avg>=4.0
                     AND (
                         (low48_all IS NOT NULL AND cr_avg>=low48_all+0.3)
                         OR
                         (low7d_all IS NOT NULL AND cr_avg>=1.5*low7d_all)
                     ) THEN 3
                WHEN low7d_all IS NOT NULL
                     AND cr_avg>=2.0*low7d_all THEN 2
                WHEN low48_all IS NOT NULL
                     AND cr_avg>=low48_all+0.3 THEN 1
                WHEN low7d_all IS NOT NULL
                     AND cr_avg>=1.5*low7d_all THEN 1
                ELSE 0
            END AS stage_rolling_all,
            CASE
                WHEN low48_current_hospital IS NULL
                     AND low7d_current_hospital IS NULL THEN NULL
                WHEN low7d_current_hospital IS NOT NULL
                     AND cr_avg>=3.0*low7d_current_hospital THEN 3
                WHEN cr_avg>=4.0
                     AND (
                         (
                             low48_current_hospital IS NOT NULL
                             AND cr_avg>=low48_current_hospital+0.3
                         )
                         OR
                         (
                             low7d_current_hospital IS NOT NULL
                             AND cr_avg>=1.5*low7d_current_hospital
                         )
                     ) THEN 3
                WHEN low7d_current_hospital IS NOT NULL
                     AND cr_avg>=2.0*low7d_current_hospital THEN 2
                WHEN low48_current_hospital IS NOT NULL
                     AND cr_avg>=low48_current_hospital+0.3 THEN 1
                WHEN low7d_current_hospital IS NOT NULL
                     AND cr_avg>=1.5*low7d_current_hospital THEN 1
                ELSE 0
            END AS stage_rolling_current_hospital
        FROM cr_rolling
        WHERE result_offset>=0
          AND result_offset<=1440
    """)

    cr_stay = con.execute("""
        SELECT
            c.stay_id,
            MAX(e.stage_rolling_all) AS stage_rolling_all,
            MAX(e.stage_rolling_current_hospital)
                AS stage_rolling_current_hospital,
            MAX(
                CASE WHEN e.stage_rolling_all IS NOT NULL THEN 1 ELSE 0 END
            ) AS evaluable_rolling_all,
            MAX(
                CASE
                    WHEN e.stage_rolling_current_hospital IS NOT NULL
                    THEN 1 ELSE 0
                END
            ) AS evaluable_rolling_current_hospital
        FROM cand c
        LEFT JOIN cr_events e USING(stay_id)
        GROUP BY c.stay_id
    """).df()

    cr_stage_summary = (
        cr_stay.groupby(
            ["stage_rolling_all", "stage_rolling_current_hospital"],
            dropna=False,
        )
        .size()
        .rename("stays")
        .reset_index()
        .sort_values("stays", ascending=False)
    )
    cr_stage_summary.to_csv(
        out / "CREATININE_ROLLING_STAGE_COMPARISON.csv",
        index=False,
    )

    cr_eval = pd.DataFrame([{
        "hf_plus24h_candidate_stays": n,
        "rolling_all_evaluable_stays": int(
            cr_stay["evaluable_rolling_all"].sum()
        ),
        "rolling_current_hospital_evaluable_stays": int(
            cr_stay["evaluable_rolling_current_hospital"].sum()
        ),
        "rolling_all_positive_stays": int(
            pd.to_numeric(
                cr_stay["stage_rolling_all"], errors="coerce"
            ).fillna(0).ge(1).sum()
        ),
        "rolling_current_hospital_positive_stays": int(
            pd.to_numeric(
                cr_stay["stage_rolling_current_hospital"], errors="coerce"
            ).fillna(0).ge(1).sum()
        ),
        "stage_difference_stays": int(
            (
                pd.to_numeric(
                    cr_stay["stage_rolling_all"], errors="coerce"
                ).fillna(-1)
                !=
                pd.to_numeric(
                    cr_stay["stage_rolling_current_hospital"],
                    errors="coerce",
                ).fillna(-1)
            ).sum()
        ),
    }])
    cr_eval.to_csv(
        out / "CREATININE_ROLLING_TRANSPORT_SUMMARY.csv",
        index=False,
    )

    cr_history = con.execute("""
        SELECT
            COUNT(*) AS attached_rows_gt7_to365d_preicu,
            COUNT(DISTINCT stay_id) AS stays_gt7_to365d_preicu,
            SUM(
                CASE
                    WHEN result_offset<hospital_admit_offset
                    THEN 1 ELSE 0
                END
            ) AS rows_gt7_to365d_predating_hospital_admit,
            COUNT(
                DISTINCT CASE
                    WHEN result_offset<hospital_admit_offset
                    THEN stay_id
                END
            ) AS stays_gt7_to365d_predating_hospital_admit,
            MIN(result_offset) AS minimum_result_offset_minutes,
            MAX(result_offset) AS maximum_result_offset_minutes
        FROM cr_same_offset
        WHERE result_offset>=-525600
          AND result_offset<-10080
    """).df()
    cr_history["interpretation"] = (
        "Availability only. eICU lab rows do not provide the fixed MIMIC "
        "strict-outpatient provenance contract; these rows are not accepted "
        "as an equivalent historical baseline by this audit."
    )
    cr_history.to_csv(
        out / "CREATININE_GT7_TO365D_ATTACHMENT_AUDIT.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3) Exact chronic-KRT candidate evidence.
    # ------------------------------------------------------------------
    ph_renal = con.execute(f"""
        SELECT
            p.patientUnitStayID AS stay_id,
            p.pastHistoryValue AS history_value,
            p.pastHistoryValueText AS history_value_text,
            p.pastHistoryPath AS history_path
        FROM read_csv_auto(
            '{qpath(PH)}',
            header=true,
            sample_size=200000
        ) p
        JOIN cand c
          ON p.patientUnitStayID=c.stay_id
        WHERE p.pastHistoryOffset<=1440
          AND p.pastHistoryEnteredOffset<=1440
          AND p.pastHistoryValue IN (
              '{PH_HD_VALUE}',
              '{PH_PD_VALUE}',
              '{PH_TRANSPLANT_VALUE}'
          )
    """).df()

    ph_path_norm = (
        ph_renal["history_path"].fillna("").astype(str).str.strip().str.lower()
    )
    ph_value_norm = (
        ph_renal["history_value"].fillna("").astype(str).str.strip().str.lower()
    )

    hd_path = (RENAL_FAILURE_PATH_PREFIX + PH_HD_VALUE).strip().lower()
    pd_path = (RENAL_FAILURE_PATH_PREFIX + PH_PD_VALUE).strip().lower()

    ph_hd_mask = (
        ph_value_norm.eq(PH_HD_VALUE.lower())
        & ph_path_norm.eq(hd_path)
    )
    ph_pd_mask = (
        ph_value_norm.eq(PH_PD_VALUE.lower())
        & ph_path_norm.eq(pd_path)
    )

    ph_hd = set(ph_renal.loc[ph_hd_mask, "stay_id"].unique())
    ph_pd = set(ph_renal.loc[ph_pd_mask, "stay_id"].unique())

    # Transplant history is descriptive/QC only and does not independently
    # define chronic KRT.
    transplant = set(
        ph_renal.loc[
            ph_value_norm.eq(PH_TRANSPLANT_VALUE.lower()),
            "stay_id",
        ].unique()
    )

    dx = con.execute(f"""
        SELECT
            d.patientUnitStayID AS stay_id,
            d.diagnosisOffset AS diagnosis_offset,
            d.diagnosisString AS diagnosis_string,
            d.ICD9Code AS icd_code
        FROM read_csv_auto(
            '{qpath(DX)}',
            header=true,
            sample_size=200000
        ) d
        JOIN cand c
          ON d.patientUnitStayID=c.stay_id
        WHERE d.diagnosisOffset<=1440
    """).df()

    diagnosis_norm = (
        dx["diagnosis_string"].fillna("").astype(str).str.strip().str.lower()
    )
    exact_esrd_path = "renal|disorder of kidney|esrd (end stage renal disease)"
    dx_esrd_path_mask = diagnosis_norm.eq(exact_esrd_path)

    def normalized_icd_tokens(value):
        if value is None:
            return set()
        s = str(value).strip()
        if not s or s.lower() == "nan":
            return set()
        out = set()
        for token in s.split(","):
            token = token.strip().upper()
            if not token:
                continue
            token = re.sub(r"[^A-Z0-9]", "", token)
            if token:
                out.add(token)
        return out

    dx["_icd_tokens"] = dx["icd_code"].map(normalized_icd_tokens)
    esrd_tokens = {"5856", "N186"}
    dependence_tokens = {"V4511", "Z992"}

    dx_esrd_code_mask = dx["_icd_tokens"].map(
        lambda s: bool(s & esrd_tokens)
    )
    dx_dependence_code_mask = dx["_icd_tokens"].map(
        lambda s: bool(s & dependence_tokens)
    )

    dx_chronic_state_mask = (
        dx_esrd_path_mask
        | dx_esrd_code_mask
        | dx_dependence_code_mask
    )

    dx_esrd_path = set(dx.loc[dx_esrd_path_mask, "stay_id"].unique())
    dx_esrd_code = set(dx.loc[dx_esrd_code_mask, "stay_id"].unique())
    dx_dependence_code = set(
        dx.loc[dx_dependence_code_mask, "stay_id"].unique()
    )
    dx_esrd = set(dx.loc[dx_chronic_state_mask, "stay_id"].unique())

    icd_qc = pd.DataFrame([{
        "exact_esrd_path_stays": len(dx_esrd_path),
        "exact_esrd_icd_5856_or_N186_stays": len(dx_esrd_code),
        "exact_dialysis_dependence_V4511_or_Z992_stays": len(dx_dependence_code),
        "chronic_state_diagnosis_union_stays": len(dx_esrd),
        "substring_matching_used": False,
    }])

    treatment = con.execute(f"""
        SELECT
            t.patientUnitStayID AS stay_id,
            t.treatmentOffset AS treatment_offset,
            t.treatmentString AS treatment_string
        FROM read_csv_auto(
            '{qpath(TREAT)}',
            header=true,
            sample_size=200000
        ) t
        JOIN cand c
          ON t.patientUnitStayID=c.stay_id
        WHERE t.treatmentOffset<=1440
    """).df()

    tstr = treatment["treatment_string"].fillna("").astype(str)
    treatment["_treatment_norm"] = tstr.str.strip().str.lower()

    chronic_treatment_norm = {
        x.strip().lower() for x in CHRONIC_TREATMENT_ALLOWLIST
    }
    acute_treatment_norm = {
        x.strip().lower() for x in ACUTE_RRT_TREATMENT_ALLOWLIST
    }

    treatment["explicit_chronic_rrt"] = (
        treatment["_treatment_norm"].isin(chronic_treatment_norm)
    )
    tr_chronic = set(
        treatment.loc[treatment["explicit_chronic_rrt"], "stay_id"].unique()
    )

    chronic = cand[["stay_id", "hospital_id"]].copy()
    chronic["ph_hemodialysis"] = chronic["stay_id"].isin(ph_hd).astype(int)
    chronic["ph_peritoneal_dialysis"] = chronic["stay_id"].isin(ph_pd).astype(int)
    chronic["current_esrd_diagnosis"] = chronic["stay_id"].isin(dx_esrd).astype(int)
    chronic["explicit_chronic_rrt_treatment"] = (
        chronic["stay_id"].isin(tr_chronic).astype(int)
    )
    chronic["renal_transplant_history"] = (
        chronic["stay_id"].isin(transplant).astype(int)
    )
    chronic["candidate_chronic_krt_union"] = (
        chronic[
            [
                "ph_hemodialysis",
                "ph_peritoneal_dialysis",
                "current_esrd_diagnosis",
                "explicit_chronic_rrt_treatment",
            ]
        ].max(axis=1)
    )

    chronic_overlap = (
        chronic.groupby(
            [
                "ph_hemodialysis",
                "ph_peritoneal_dialysis",
                "current_esrd_diagnosis",
                "explicit_chronic_rrt_treatment",
                "renal_transplant_history",
                "candidate_chronic_krt_union",
            ]
        )
        .size()
        .rename("stays")
        .reset_index()
        .sort_values("stays", ascending=False)
    )
    chronic_overlap.to_csv(
        out / "CHRONIC_KRT_CANDIDATE_OVERLAP.csv",
        index=False,
    )

    chronic_summary = pd.DataFrame([{
        "hf_plus24h_candidate_stays": n,
        "pastHistory_hemodialysis_stays": len(ph_hd),
        "pastHistory_peritoneal_dialysis_stays": len(ph_pd),
        "current_esrd_diagnosis_stays": len(dx_esrd),
        "exact_esrd_path_stays": len(dx_esrd_path),
        "exact_esrd_icd_5856_or_N186_stays": len(dx_esrd_code),
        "exact_dialysis_dependence_V4511_or_Z992_stays": len(dx_dependence_code),
        "explicit_chronic_rrt_treatment_stays": len(tr_chronic),
        "candidate_chronic_krt_union_stays": int(
            chronic["candidate_chronic_krt_union"].sum()
        ),
        "renal_transplant_history_stays": len(transplant),
        "transplant_and_candidate_chronic_krt_stays": int(
            (
                chronic["renal_transplant_history"].eq(1)
                & chronic["candidate_chronic_krt_union"].eq(1)
            ).sum()
        ),
    }])
    chronic_summary.to_csv(
        out / "CHRONIC_KRT_CANDIDATE_SUMMARY.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 4) Treatment delivered-RRT classification.
    #
    # This is an audit classifier. Catheter placement/access surgery are
    # intentionally excluded from "delivered treatment".
    # Explicit chronic-RRT treatment is separated above.
    # ------------------------------------------------------------------
    s = treatment["treatment_string"].fillna("").astype(str).str.lower()

    treatment["is_access_or_catheter"] = s.str.contains(
        r"insertion of .*catheter|dialysis access surgery|"
        r"arteriovenous shunt",
        regex=True,
        na=False,
    )

    # Primary delivered-RRT treatment evidence is exact-allowlist based.
    treatment["delivered_rrt_candidate"] = (
        treatment["_treatment_norm"].isin(acute_treatment_norm)
    )

    uf_labels = {
        "renal|dialysis|ultrafiltration (fluid removal only)",
        "renal|dialysis|ultrafiltration (fluid removal only)|emergent",
        "renal|dialysis|ultrafiltration (fluid removal only)|for acute renal failure",
    }
    crrt_labels = {
        "renal|dialysis|c v v h d",
        "renal|dialysis|c v v h",
    }
    sled_labels = {"renal|dialysis|sled"}
    pd_labels = {"renal|dialysis|peritoneal dialysis"}

    treatment["is_uf"] = treatment["_treatment_norm"].isin(uf_labels)
    treatment["is_crrt"] = treatment["_treatment_norm"].isin(crrt_labels)
    treatment["is_sled"] = treatment["_treatment_norm"].isin(sled_labels)
    treatment["is_pd"] = treatment["_treatment_norm"].isin(pd_labels)
    treatment["is_hd"] = (
        treatment["delivered_rrt_candidate"]
        & ~treatment["is_uf"]
        & ~treatment["is_crrt"]
        & ~treatment["is_sled"]
        & ~treatment["is_pd"]
    )

    treatment["modality"] = np.select(
        [
            treatment["is_crrt"],
            treatment["is_sled"],
            treatment["is_pd"],
            treatment["is_uf"],
            treatment["is_hd"],
        ],
        ["CRRT", "SLED", "PD", "UF", "HD/OTHER_DIALYSIS"],
        default="OTHER",
    )

    treatment["offset_num"] = pd.to_numeric(
        treatment["treatment_offset"], errors="coerce"
    )
    treatment["delivered_rrt_0_24"] = (
        treatment["delivered_rrt_candidate"]
        & treatment["offset_num"].between(0, 1440, inclusive="both")
    )
    treatment["delivered_rrt_negative_offset"] = (
        treatment["delivered_rrt_candidate"]
        & treatment["offset_num"].lt(0)
    )

    treatment_allowlist_qc = (
        treatment.loc[
            treatment["_treatment_norm"].isin(
                acute_treatment_norm | chronic_treatment_norm
            )
        ]
        .groupby("treatment_string", dropna=False)
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
        )
        .reset_index()
        .sort_values(["stays", "rows"], ascending=False)
    )
    treatment_allowlist_qc["is_hyperphosphatemia_dialysis"] = (
        treatment_allowlist_qc["treatment_string"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .eq(
            "renal|electrolyte correction|treatment of "
            "hyperphosphatemia|dialysis"
        )
    )

    treatment_audit = (
        treatment.loc[
            treatment[
                [
                    "delivered_rrt_candidate",
                    "explicit_chronic_rrt",
                    "is_access_or_catheter",
                ]
            ].any(axis=1)
        ]
        .groupby(
            [
                "treatment_string",
                "modality",
                "delivered_rrt_candidate",
                "explicit_chronic_rrt",
                "is_access_or_catheter",
            ],
            dropna=False,
        )
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            rows_0_24=("delivered_rrt_0_24", "sum"),
            rows_negative_offset=("delivered_rrt_negative_offset", "sum"),
        )
        .reset_index()
        .sort_values(["stays", "rows"], ascending=False)
    )
    treatment_audit.to_csv(
        out / "RRT_TREATMENT_ADJUDICATION.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 5) Intake/output delivered-RRT classification.
    #
    # Do NOT use dialysisTotal as independent RRT evidence because it is
    # cumulative and duplicated on unrelated rows by design.
    # ------------------------------------------------------------------
    io = con.execute(f"""
        SELECT
            i.patientUnitStayID AS stay_id,
            i.intakeOutputOffset AS io_offset,
            i.intakeOutputEntryOffset AS entry_offset,
            i.dialysisTotal AS dialysis_total,
            i.cellPath AS cell_path,
            i.cellLabel AS cell_label,
            i.cellValueNumeric AS cell_value_numeric
        FROM read_csv_auto(
            '{qpath(IO)}',
            header=true,
            sample_size=200000
        ) i
        JOIN cand c
          ON i.patientUnitStayID=c.stay_id
        WHERE i.intakeOutputOffset<=1440
          AND i.intakeOutputEntryOffset<=1440
    """).df()

    path_lower = io["cell_path"].fillna("").astype(str).str.strip().str.lower()
    label_lower = io["cell_label"].fillna("").astype(str).str.strip().str.lower()
    io["value_num"] = pd.to_numeric(
        io["cell_value_numeric"], errors="coerce"
    )
    io["offset_num"] = pd.to_numeric(io["io_offset"], errors="coerce")
    io["dialysis_total_num"] = pd.to_numeric(
        io["dialysis_total"], errors="coerce"
    )

    io_rrt_paths_norm = {
        x.strip().lower() for x in IO_RRT_PATH_ALLOWLIST
    }
    io["explicit_delivered_rrt_label"] = path_lower.isin(
        io_rrt_paths_norm
    )

    io["delivered_rrt_positive_value"] = (
        io["explicit_delivered_rrt_label"]
        & io["value_num"].gt(0)
    )
    io["delivered_rrt_0_24"] = (
        io["delivered_rrt_positive_value"]
        & io["offset_num"].between(0, 1440, inclusive="both")
    )
    io["delivered_rrt_negative_offset"] = (
        io["delivered_rrt_positive_value"]
        & io["offset_num"].lt(0)
    )
    io["is_uf"] = (
        path_lower.str.contains("ultrafiltration", regex=False)
        | path_lower.str.contains("uf removed", regex=False)
    )
    io["is_nonuf_rrt"] = (
        io["explicit_delivered_rrt_label"] & ~io["is_uf"]
    )

    io_audit = (
        io.loc[
            io["explicit_delivered_rrt_label"]
            | io["dialysis_total_num"].fillna(0).ne(0)
        ]
        .groupby(["cell_path", "cell_label"], dropna=False)
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            explicit_rrt_rows=("explicit_delivered_rrt_label", "sum"),
            positive_explicit_rrt_rows=("delivered_rrt_positive_value", "sum"),
            explicit_rrt_rows_0_24=("delivered_rrt_0_24", "sum"),
            explicit_rrt_rows_negative_offset=(
                "delivered_rrt_negative_offset", "sum"
            ),
            nonzero_dialysis_total_rows=(
                "dialysis_total_num",
                lambda x: int(pd.to_numeric(x, errors="coerce")
                              .fillna(0).ne(0).sum()),
            ),
        )
        .reset_index()
        .sort_values(["stays", "rows"], ascending=False)
    )
    io_audit.to_csv(
        out / "RRT_INTAKEOUTPUT_ADJUDICATION.csv",
        index=False,
    )

    false_total = pd.DataFrame([{
        "rows_with_nonzero_dialysisTotal": int(
            io["dialysis_total_num"].fillna(0).ne(0).sum()
        ),
        "stays_with_nonzero_dialysisTotal": int(
            io.loc[
                io["dialysis_total_num"].fillna(0).ne(0),
                "stay_id",
            ].nunique()
        ),
        "rows_nonzero_dialysisTotal_on_nonexplicit_rrt_label": int(
            (
                io["dialysis_total_num"].fillna(0).ne(0)
                & ~io["explicit_delivered_rrt_label"]
            ).sum()
        ),
        "stays_nonzero_dialysisTotal_on_nonexplicit_rrt_label": int(
            io.loc[
                io["dialysis_total_num"].fillna(0).ne(0)
                & ~io["explicit_delivered_rrt_label"],
                "stay_id",
            ].nunique()
        ),
        "interpretation": (
            "dialysisTotal is cumulative/duplicated and is not accepted "
            "as independent delivered-RRT evidence by this audit."
        ),
    }])

    # ------------------------------------------------------------------
    # 6) APACHE dialysis flag: supportive-only candidate evidence.
    # ------------------------------------------------------------------
    aps = con.execute(f"""
        SELECT
            a.patientUnitStayID AS stay_id,
            TRY_CAST(a.dialysis AS INTEGER) AS apache_dialysis
        FROM read_csv_auto(
            '{qpath(APS)}',
            header=true,
            sample_size=200000
        ) a
        JOIN cand c
          ON a.patientUnitStayID=c.stay_id
    """).df()
    apache_pos = set(
        aps.loc[aps["apache_dialysis"].eq(1), "stay_id"].unique()
    )

    # ------------------------------------------------------------------
    # 7) Stay-level RRT adjudication overlap after candidate chronic-KRT.
    # ------------------------------------------------------------------
    tr_primary = set(
        treatment.loc[treatment["delivered_rrt_0_24"], "stay_id"].unique()
    )
    tr_negative = set(
        treatment.loc[
            treatment["delivered_rrt_negative_offset"],
            "stay_id",
        ].unique()
    )
    io_primary = set(
        io.loc[io["delivered_rrt_0_24"], "stay_id"].unique()
    )
    io_negative = set(
        io.loc[io["delivered_rrt_negative_offset"], "stay_id"].unique()
    )

    tr_uf = set(
        treatment.loc[
            treatment["delivered_rrt_0_24"] & treatment["is_uf"],
            "stay_id",
        ].unique()
    )
    tr_nonuf = set(
        treatment.loc[
            treatment["delivered_rrt_0_24"] & ~treatment["is_uf"],
            "stay_id",
        ].unique()
    )
    io_uf = set(
        io.loc[
            io["delivered_rrt_0_24"] & io["is_uf"],
            "stay_id",
        ].unique()
    )
    io_nonuf = set(
        io.loc[
            io["delivered_rrt_0_24"] & io["is_nonuf_rrt"],
            "stay_id",
        ].unique()
    )

    rrt = chronic[
        [
            "stay_id",
            "hospital_id",
            "candidate_chronic_krt_union",
            "renal_transplant_history",
        ]
    ].copy()
    rrt["treatment_delivered_rrt_0_24"] = (
        rrt["stay_id"].isin(tr_primary).astype(int)
    )
    rrt["intakeoutput_delivered_rrt_0_24"] = (
        rrt["stay_id"].isin(io_primary).astype(int)
    )
    rrt["apache_dialysis_flag_1"] = (
        rrt["stay_id"].isin(apache_pos).astype(int)
    )
    rrt["treatment_negative_offset_rrt"] = (
        rrt["stay_id"].isin(tr_negative).astype(int)
    )
    rrt["intakeoutput_negative_offset_rrt"] = (
        rrt["stay_id"].isin(io_negative).astype(int)
    )
    rrt["primary_delivered_rrt_candidate"] = (
        rrt[
            [
                "treatment_delivered_rrt_0_24",
                "intakeoutput_delivered_rrt_0_24",
            ]
        ].max(axis=1)
    )
    rrt["primary_delivered_rrt_after_chronic_exclusion"] = (
        rrt["primary_delivered_rrt_candidate"].eq(1)
        & rrt["candidate_chronic_krt_union"].eq(0)
    ).astype(int)
    rrt["apache_incremental_after_chronic_and_primary"] = (
        rrt["apache_dialysis_flag_1"].eq(1)
        & rrt["candidate_chronic_krt_union"].eq(0)
        & rrt["primary_delivered_rrt_candidate"].eq(0)
    ).astype(int)

    rrt["uf_evidence_0_24"] = (
        rrt["stay_id"].isin(tr_uf | io_uf).astype(int)
    )
    rrt["nonuf_rrt_evidence_0_24"] = (
        rrt["stay_id"].isin(tr_nonuf | io_nonuf).astype(int)
    )
    rrt["isolated_uf_candidate"] = (
        rrt["uf_evidence_0_24"].eq(1)
        & rrt["nonuf_rrt_evidence_0_24"].eq(0)
    ).astype(int)
    rrt["isolated_uf_after_chronic_exclusion"] = (
        rrt["isolated_uf_candidate"].eq(1)
        & rrt["candidate_chronic_krt_union"].eq(0)
    ).astype(int)

    # Fixed-cohort UF-suppression audit:
    # remove only the UF-derived Stage-3 contribution and inspect the
    # remaining rolling-creatinine stage. This does NOT reselect patients,
    # episodes, or time zero.
    rrt = rrt.merge(
        cr_stay[["stay_id", "stage_rolling_all"]],
        on="stay_id",
        how="left",
        validate="1:1",
    )

    uf_fixed = rrt.loc[
        rrt["isolated_uf_after_chronic_exclusion"].eq(1),
        ["stay_id", "stage_rolling_all"],
    ].copy()

    def uf_stage_label(x):
        if pd.isna(x):
            return "RENAL_UNEVALUABLE"
        x = int(x)
        if x == 0:
            return "STAGE_0"
        return f"STAGE_{x}"

    uf_fixed["stage_after_uf_suppression"] = (
        uf_fixed["stage_rolling_all"].map(uf_stage_label)
    )

    uf_suppression = (
        uf_fixed.groupby("stage_after_uf_suppression", dropna=False)
        .size()
        .rename("stays")
        .reset_index()
        .sort_values("stage_after_uf_suppression")
    )
    uf_suppression.to_csv(
        out / "UF_SUPPRESSION_FIXED_COHORT_STAGE_AUDIT.csv",
        index=False,
    )

    uf_stage0 = int(
        (uf_fixed["stage_after_uf_suppression"] == "STAGE_0").sum()
    )
    uf_unevaluable = int(
        (
            uf_fixed["stage_after_uf_suppression"]
            == "RENAL_UNEVALUABLE"
        ).sum()
    )
    uf_loses_positive = uf_stage0 + uf_unevaluable

    uf_suppression_summary = pd.DataFrame([{
        "isolated_uf_after_chronic_exclusion_stays": int(len(uf_fixed)),
        "stage0_after_uf_suppression_stays": uf_stage0,
        "renal_unevaluable_after_uf_suppression_stays": uf_unevaluable,
        "would_lose_cr_rrt_aki_positivity_stays": uf_loses_positive,
        "cohort_membership_reselected": False,
        "episode_reselected": False,
        "time_zero_changed": False,
        "stage1_substitution_for_stage0_or_unevaluable": False,
    }])
    uf_suppression_summary.to_csv(
        out / "UF_SUPPRESSION_FIXED_COHORT_SUMMARY.csv",
        index=False,
    )

    rrt_overlap = (
        rrt.groupby(
            [
                "candidate_chronic_krt_union",
                "treatment_delivered_rrt_0_24",
                "intakeoutput_delivered_rrt_0_24",
                "apache_dialysis_flag_1",
                "primary_delivered_rrt_candidate",
                "primary_delivered_rrt_after_chronic_exclusion",
                "apache_incremental_after_chronic_and_primary",
                "isolated_uf_candidate",
                "isolated_uf_after_chronic_exclusion",
            ]
        )
        .size()
        .rename("stays")
        .reset_index()
        .sort_values("stays", ascending=False)
    )
    rrt_overlap.to_csv(
        out / "RRT_ADJUDICATION_OVERLAP.csv",
        index=False,
    )

    rrt_summary = pd.DataFrame([{
        "hf_plus24h_candidate_stays": n,
        "candidate_chronic_krt_union_stays": int(
            chronic["candidate_chronic_krt_union"].sum()
        ),
        "treatment_delivered_rrt_0_24_stays": len(tr_primary),
        "intakeoutput_delivered_rrt_0_24_stays": len(io_primary),
        "primary_delivered_rrt_union_stays": int(
            rrt["primary_delivered_rrt_candidate"].sum()
        ),
        "primary_delivered_rrt_after_candidate_chronic_exclusion_stays": int(
            rrt["primary_delivered_rrt_after_chronic_exclusion"].sum()
        ),
        "apache_dialysis_flag_1_stays": len(apache_pos),
        "apache_incremental_after_chronic_and_primary_stays": int(
            rrt["apache_incremental_after_chronic_and_primary"].sum()
        ),
        "treatment_negative_offset_rrt_stays": len(tr_negative),
        "intakeoutput_negative_offset_rrt_stays": len(io_negative),
        "isolated_uf_candidate_stays": int(
            rrt["isolated_uf_candidate"].sum()
        ),
        "isolated_uf_after_candidate_chronic_exclusion_stays": int(
            rrt["isolated_uf_after_chronic_exclusion"].sum()
        ),
        "uf_suppression_stage0_stays": uf_stage0,
        "uf_suppression_renal_unevaluable_stays": uf_unevaluable,
        "uf_suppression_would_lose_cr_rrt_aki_positivity_stays": uf_loses_positive,
    }])
    rrt_summary.to_csv(
        out / "RRT_ADJUDICATION_SUMMARY.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 8) Site-level source availability in the candidate population.
    # Absence is never automatically converted to clinical absence here.
    # ------------------------------------------------------------------
    site = (
        cand.groupby("hospital_id")
        .size()
        .rename("hf_plus24h_stays")
        .reset_index()
    )

    source_sets = {
        "any_creatinine": set(
            con.execute(
                "SELECT DISTINCT stay_id FROM cr_raw"
            ).df()["stay_id"].unique()
        ),
        "any_treatment": set(treatment["stay_id"].unique()),
        "any_intakeoutput": set(io["stay_id"].unique()),
        "any_apacheapsvar": set(aps["stay_id"].unique()),
    }

    for name, ids in source_sets.items():
        tmp = cand[["stay_id", "hospital_id"]].copy()
        tmp["present"] = tmp["stay_id"].isin(ids).astype(int)
        h = (
            tmp.groupby("hospital_id")["present"]
            .sum()
            .rename(f"{name}_stays")
            .reset_index()
        )
        site = site.merge(h, on="hospital_id", how="left")
        site[f"{name}_stays"] = site[f"{name}_stays"].fillna(0).astype(int)
        site[f"{name}_pct"] = (
            100.0 * site[f"{name}_stays"] / site["hf_plus24h_stays"]
        )

    site.to_csv(
        out / "RENAL_ADJUDICATION_SOURCE_CAPTURE_BY_HOSPITAL.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 9) Safety + manifest.
    # ------------------------------------------------------------------
    for path in out.glob("*.csv"):
        df0 = pd.read_csv(path, nrows=5)
        assert_safe_columns(df0.columns, path.name)

    hashes = {
        path.name: sha256_file(path)
        for path in sorted(out.glob("*.csv"), key=lambda p: p.name)
    }

    manifest = {
        "chronic_krt_code_summary": json.loads(icd_qc.to_json(orient="records", double_precision=15)),
        "dialysis_volume_comparison": json.loads(false_total.to_json(orient="records", double_precision=15)),
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "outcome-blind renal adjudication evidence after red-team corrections",
        "hf_plus24h_candidate_stays": n,
        "candidate_hospitals": nh,
        "outcome_or_discharge_status_selected": False,
        "discharge_location_selected": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "final_episode_selection_applied": False,
        "renal_contract_fixed_by_script": False,
        "chronic_krt_exclusion_fixed": False,
        "acute_rrt_definition_fixed": False,
        "kdigo_definition_fixed": False,
        "historical_outpatient_proxy_reconstructed": False,
        "exact_icd_token_matching_used": True,
        "hyperphosphatemia_dialysis_label_in_primary_allowlist": True,
        "exact_treatment_allowlist_used": True,
        "exact_intakeoutput_path_allowlist_used": True,
        "uf_suppression_fixed_cohort_stage_audited": True,
        "scientific_rule_tuned_from_outcome": False,
        "source_files": {
            "patient": PATIENT.name,
            "pastHistory": PH.name,
            "lab": LAB.name,
            "treatment": TREAT.name,
            "intakeOutput": IO.name,
            "apacheApsVar": APS.name,
            "diagnosis": DX.name,
        },
        "generated_sha256": hashes,
        "interpretation": (
            "Evidence-only adjudication audit. Candidate chronic-KRT union, "
            "delivered-RRT source union, isolated-UF flag, rolling creatinine "
            "transport, and APACHE incremental evidence are outside the scope of this "
            "script. The >7-365 d MIMIC strict-outpatient creatinine proxy is "
            "not claimed to be reconstructible from eICU."
        ),
    }

    (out / "RENAL_ADJUDICATION_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("EICU RENAL ADJUDICATION AUDIT: PASS")
    print(f"HF +24 h candidate stays: {n:,}")
    print(f"Hospitals: {nh:,}")
    print(f"Output: {out}")
    print("Outcome/discharge-status values selected: NO")
    print("Discharge-location values selected: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("Final episode selection applied: NO")
    print("Renal contract fixed by this script: NO")
    print("Historical outpatient proxy reconstructed: NO")
    print("=" * 78)

if __name__ == "__main__":
    main()
