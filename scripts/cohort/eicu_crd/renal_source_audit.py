#!/usr/bin/env python3
"""
Outcome-blind eICU renal source / creatinine / RRT feasibility audit.

Purpose
-------
Generate the evidence needed for the later creatinine / RRT / chronic-KRT /
KDIGO harmonization contract.

This script deliberately DOES NOT lock a renal phenotype and DOES NOT:
- attach mortality or discharge status/location,
- fit any model,
- compute performance,
- choose a KDIGO definition,
- label RRT as acute vs chronic,
- exclude chronic KRT patients,
- perform final episode selection.

Population
----------
Stay-level audit population only:
1) adult;
2) demonstrably present beyond +24 h;
3) fixed structured eICU HF proxy positive.

Episode selection is intentionally NOT applied because the fixed episode
contract requires complete stay-level eligibility first.

Primary audit domains
---------------------
A. Standardized creatinine availability, units, timing, plausibility.
B. Rolling-history availability for 48 h / 7 d creatinine comparisons.
C. Candidate RRT evidence in:
   - treatment
   - intakeOutput
   - apacheApsVar dialysis flag
D. Candidate chronic-KRT evidence in:
   - pastHistory
   - diagnosis
E. Hospital-level source capture and source overlap.

All candidate string searches are inventory screens only. They do not define
the final phenotype.
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

RRT_RE = re.compile(
    r"dialysis|hemodialysis|haemodialysis|hemofiltration|haemofiltration|"
    r"crrt|cvvh|cvvhd|cvvhdf|renal replacement|ultrafiltration",
    re.I,
)

RENAL_HISTORY_RE = re.compile(
    r"renal|kidney|dialysis|hemodialysis|haemodialysis|"
    r"end[- ]?stage|esrd|chronic kidney|chronic renal",
    re.I,
)

RENAL_DX_RE = re.compile(
    r"renal|kidney|dialysis|hemodialysis|haemodialysis|"
    r"end[- ]?stage|esrd|chronic kidney|chronic renal",
    re.I,
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

    # ---------------------------------------------------------------
    # 1) Outcome-safe adult +24 h stay population.
    # ---------------------------------------------------------------
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
    patient["adult"] = patient["age_num"].fillna(-1).ge(18)

    unit_off = pd.to_numeric(patient["unit_discharge_offset"], errors="coerce")
    hosp_off = pd.to_numeric(patient["hospital_discharge_offset"], errors="coerce")

    patient["landmark_plus24"] = (
        patient["adult"]
        & unit_off.gt(1440)
        & hosp_off.gt(1440)
    )

    base = patient.loc[
        patient["landmark_plus24"],
        [
            "stay_id", "hstay_id", "unique_pid", "hospital_id",
            "hospital_admit_offset"
        ]
    ].copy()

    if base["stay_id"].duplicated().any():
        raise AssertionError("patientUnitStayID duplicated in patient table")

    # ---------------------------------------------------------------
    # 2) Fixed HF-proxy exact tuple recheck.
    # ---------------------------------------------------------------
    con.register(
        "base_ids",
        base[["stay_id"]].rename(columns={"stay_id": "patientUnitStayID"})
    )
    values_sql = ",".join("'" + v.replace("'", "''") + "'" for v in HF_VALUES)

    hf = con.execute(f"""
        SELECT DISTINCT
            ph.patientUnitStayID AS stay_id
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
          AND ph.pastHistoryValueText = ph.pastHistoryValue
          AND ph.pastHistoryPath =
              '{HF_PATH_PREFIX.replace("'", "''")}' || ph.pastHistoryValue
    """).df()

    candidate = base.merge(hf.assign(hf_proxy=1), on="stay_id", how="inner")
    if candidate["stay_id"].duplicated().any():
        raise AssertionError("HF candidate duplicated")

    n = len(candidate)
    nh = candidate["hospital_id"].nunique()
    con.register("cand_df", candidate)
    con.execute("CREATE TEMP TABLE cand AS SELECT * FROM cand_df")

    # ---------------------------------------------------------------
    # 3) Creatinine audit.
    # ---------------------------------------------------------------
    cr = con.execute(f"""
        SELECT
            l.patientUnitStayID AS stay_id,
            c.hospital_id AS hospital_id,
            c.hospital_admit_offset AS hospital_admit_offset,
            l.labName AS lab_name,
            l.labTypeID AS lab_type_id,
            l.labResultOffset AS result_offset,
            l.labResultRevisedOffset AS revised_offset,
            l.labResult AS result_value,
            l.labMeasureNameSystem AS unit_system,
            l.labMeasureNameInterface AS unit_interface
        FROM read_csv_auto(
            '{qpath(LAB)}',
            header=true,
            sample_size=200000
        ) l
        JOIN cand c
          ON l.patientUnitStayID=c.stay_id
        WHERE lower(l.labName)='creatinine'
    """).df()
    assert_safe_columns(cr.columns, "creatinine_safe")

    cr["result_value_num"] = pd.to_numeric(
        cr["result_value"], errors="coerce"
    )
    cr["result_offset_num"] = pd.to_numeric(
        cr["result_offset"], errors="coerce"
    )
    cr["revised_offset_num"] = pd.to_numeric(
        cr["revised_offset"], errors="coerce"
    )
    cr["hospital_admit_offset_num"] = pd.to_numeric(
        cr["hospital_admit_offset"], errors="coerce"
    )

    cr["plausible"] = (
        cr["result_value_num"].gt(0)
        & cr["result_value_num"].le(150)
    )
    cr["entered_by_landmark"] = cr["revised_offset_num"].le(1440)

    cr["current_0_24h"] = (
        cr["result_offset_num"].ge(0)
        & cr["result_offset_num"].le(1440)
        & cr["entered_by_landmark"]
        & cr["plausible"]
    )

    cr["pre_icu_current_hospital"] = (
        cr["result_offset_num"].lt(0)
        & cr["result_offset_num"].ge(cr["hospital_admit_offset_num"])
        & cr["entered_by_landmark"]
        & cr["plausible"]
    )

    cr["predates_current_hospital_admit"] = (
        cr["result_offset_num"].lt(cr["hospital_admit_offset_num"])
        & cr["entered_by_landmark"]
        & cr["plausible"]
    )

    cr["any_pre_icu_48h"] = (
        cr["result_offset_num"].ge(-2880)
        & cr["result_offset_num"].lt(0)
        & cr["entered_by_landmark"]
        & cr["plausible"]
    )

    cr["any_pre_icu_7d"] = (
        cr["result_offset_num"].ge(-10080)
        & cr["result_offset_num"].lt(0)
        & cr["entered_by_landmark"]
        & cr["plausible"]
    )

    units = (
        cr.groupby(
            ["lab_name", "lab_type_id", "unit_system", "unit_interface"],
            dropna=False,
        )
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            hospitals=("hospital_id", "nunique"),
        )
        .reset_index()
        .sort_values(["stays", "rows"], ascending=False)
    )
    units.to_csv(out / "CREATININE_LABEL_UNIT_INVENTORY.csv", index=False)

    timing = pd.DataFrame([{
        "hf_plus24h_candidate_stays": n,
        "candidate_hospitals": nh,
        "creatinine_rows_total": int(len(cr)),
        "creatinine_stays_any": int(cr["stay_id"].nunique()),
        "rows_nonnumeric": int(cr["result_value_num"].isna().sum()),
        "rows_nonpositive": int(
            (
                cr["result_value_num"].notna()
                & cr["result_value_num"].le(0)
            ).sum()
        ),
        "rows_above_150": int(cr["result_value_num"].gt(150).sum()),
        "rows_revision_after_landmark": int(
            cr["revised_offset_num"].gt(1440).sum()
        ),
        "rows_negative_revision_offset": int(
            cr["revised_offset_num"].lt(0).sum()
        ),
        "rows_revision_before_draw": int(
            (
                cr["revised_offset_num"]
                < cr["result_offset_num"]
            ).sum()
        ),
        "stays_current_0_24h": int(
            cr.loc[cr["current_0_24h"], "stay_id"].nunique()
        ),
        "stays_pre_icu_current_hospital": int(
            cr.loc[cr["pre_icu_current_hospital"], "stay_id"].nunique()
        ),
        "stays_with_any_pre_icu_48h": int(
            cr.loc[cr["any_pre_icu_48h"], "stay_id"].nunique()
        ),
        "stays_with_any_pre_icu_7d": int(
            cr.loc[cr["any_pre_icu_7d"], "stay_id"].nunique()
        ),
        "rows_predating_current_hospital_admit": int(
            cr["predates_current_hospital_admit"].sum()
        ),
        "stays_predating_current_hospital_admit": int(
            cr.loc[
                cr["predates_current_hospital_admit"], "stay_id"
            ].nunique()
        ),
    }])

    # Same-offset duplicates for admissible current creatinine.
    cur = cr.loc[cr["current_0_24h"]].copy()
    dup = (
        cur.groupby(["stay_id", "result_offset_num"], as_index=False)
        .agg(
            rows=("result_value_num", "size"),
            distinct_values=("result_value_num", "nunique"),
            minimum=("result_value_num", "min"),
            maximum=("result_value_num", "max"),
            mean=("result_value_num", "mean"),
        )
    )
    dup_summary = pd.DataFrame([{
        "admissible_current_creatinine_rows": int(len(cur)),
        "stays_with_current_creatinine": int(cur["stay_id"].nunique()),
        "stay_offset_groups": int(len(dup)),
        "groups_with_multiple_rows": int((dup["rows"] > 1).sum()),
        "groups_with_multiple_distinct_values": int(
            (dup["distinct_values"] > 1).sum()
        ),
        "stays_with_any_same_offset_multiple_rows": int(
            dup.loc[dup["rows"] > 1, "stay_id"].nunique()
        ),
        "stays_with_any_same_offset_value_disagreement": int(
            dup.loc[dup["distinct_values"] > 1, "stay_id"].nunique()
        ),
    }])

    # Rolling-history availability for each current creatinine event.
    # This does NOT calculate stage; it audits whether 48h/7d comparators exist.
    current_events = cur[
        ["stay_id", "result_offset_num", "result_value_num"]
    ].copy()
    hist = cr.loc[
        cr["entered_by_landmark"] & cr["plausible"],
        ["stay_id", "result_offset_num", "result_value_num"],
    ].copy()

    con.register("cr_current", current_events)
    con.register("cr_hist", hist)

    roll = con.execute("""
        SELECT
            c.stay_id,
            c.result_offset_num AS current_offset,
            COUNT(h48.result_value_num) AS prior_48h_count,
            MIN(h48.result_value_num) AS prior_48h_min,
            COUNT(h7.result_value_num) AS prior_7d_count,
            MIN(h7.result_value_num) AS prior_7d_min
        FROM cr_current c
        LEFT JOIN cr_hist h48
          ON h48.stay_id=c.stay_id
         AND h48.result_offset_num < c.result_offset_num
         AND h48.result_offset_num >= c.result_offset_num - 2880
        LEFT JOIN cr_hist h7
          ON h7.stay_id=c.stay_id
         AND h7.result_offset_num < c.result_offset_num
         AND h7.result_offset_num >= c.result_offset_num - 10080
        GROUP BY c.stay_id, c.result_offset_num
    """).df()

    roll_summary = pd.DataFrame([{
        "current_creatinine_event_groups": int(len(roll)),
        "events_with_prior_48h_comparator": int(
            (roll["prior_48h_count"] > 0).sum()
        ),
        "events_with_prior_7d_comparator": int(
            (roll["prior_7d_count"] > 0).sum()
        ),
        "stays_with_any_current_event_having_prior_48h": int(
            roll.loc[roll["prior_48h_count"] > 0, "stay_id"].nunique()
        ),
        "stays_with_any_current_event_having_prior_7d": int(
            roll.loc[roll["prior_7d_count"] > 0, "stay_id"].nunique()
        ),
    }])
    roll_summary.to_csv(
        out / "CREATININE_ROLLING_HISTORY_AVAILABILITY.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 4) RRT candidate evidence — treatment.
    # ---------------------------------------------------------------
    treatment = con.execute(f"""
        SELECT
            t.patientUnitStayID AS stay_id,
            c.hospital_id AS hospital_id,
            t.treatmentOffset AS treatment_offset,
            t.treatmentString AS treatment_string
        FROM read_csv_auto(
            '{qpath(TREAT)}',
            header=true,
            sample_size=200000
        ) t
        JOIN cand c
          ON t.patientUnitStayID=c.stay_id
        WHERE t.treatmentOffset <= 1440
    """).df()
    assert_safe_columns(treatment.columns, "treatment_safe")

    treat_mask = treatment["treatment_string"].fillna("").str.contains(
        RRT_RE, na=False
    )
    trrt = treatment.loc[treat_mask].copy()

    trrt_inventory = (
        trrt.groupby("treatment_string", dropna=False)
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            hospitals=("hospital_id", "nunique"),
        )
        .reset_index()
        .sort_values(["stays", "rows"], ascending=False)
    )
    trrt_inventory.to_csv(
        out / "RRT_TREATMENT_LABEL_INVENTORY.csv", index=False
    )

    # ---------------------------------------------------------------
    # 5) RRT candidate evidence — intakeOutput.
    # ---------------------------------------------------------------
    io = con.execute(f"""
        SELECT
            i.patientUnitStayID AS stay_id,
            c.hospital_id AS hospital_id,
            i.intakeOutputOffset AS io_offset,
            i.intakeOutputEntryOffset AS io_entry_offset,
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
        WHERE i.intakeOutputOffset <= 1440
          AND i.intakeOutputEntryOffset <= 1440
    """).df()
    assert_safe_columns(io.columns, "intakeoutput_safe")

    io["dialysis_total_num"] = pd.to_numeric(
        io["dialysis_total"], errors="coerce"
    )
    io_text = (
        io["cell_path"].fillna("").astype(str) + " "
        + io["cell_label"].fillna("").astype(str)
    )
    io["dialysis_text_candidate"] = io_text.str.contains(
        RRT_RE, na=False
    )
    io["nonzero_dialysis_total"] = (
        io["dialysis_total_num"].notna()
        & io["dialysis_total_num"].ne(0)
    )

    io_cand = io.loc[
        io["dialysis_text_candidate"] | io["nonzero_dialysis_total"]
    ].copy()

    io_inventory = (
        io_cand.groupby(["cell_path", "cell_label"], dropna=False)
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            hospitals=("hospital_id", "nunique"),
            rows_text_candidate=("dialysis_text_candidate", "sum"),
            rows_nonzero_dialysis_total=("nonzero_dialysis_total", "sum"),
        )
        .reset_index()
        .sort_values(["stays", "rows"], ascending=False)
    )
    io_inventory.to_csv(
        out / "RRT_INTAKEOUTPUT_LABEL_INVENTORY.csv", index=False
    )

    # ---------------------------------------------------------------
    # 6) APACHE dialysis flag.
    # ---------------------------------------------------------------
    aps = con.execute(f"""
        SELECT
            a.patientUnitStayID AS stay_id,
            c.hospital_id AS hospital_id,
            a.dialysis AS apache_dialysis
        FROM read_csv_auto(
            '{qpath(APS)}',
            header=true,
            sample_size=200000
        ) a
        JOIN cand c
          ON a.patientUnitStayID=c.stay_id
    """).df()
    assert_safe_columns(aps.columns, "apache_safe")

    aps["apache_dialysis_num"] = pd.to_numeric(
        aps["apache_dialysis"], errors="coerce"
    )

    apache_summary = (
        aps.groupby("apache_dialysis_num", dropna=False)
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            hospitals=("hospital_id", "nunique"),
        )
        .reset_index()
    )

    # ---------------------------------------------------------------
    # 7) Candidate chronic renal / chronic KRT evidence — pastHistory.
    # ---------------------------------------------------------------
    ph_renal = con.execute(f"""
        SELECT
            p.patientUnitStayID AS stay_id,
            c.hospital_id AS hospital_id,
            p.pastHistoryOffset AS history_offset,
            p.pastHistoryEnteredOffset AS entered_offset,
            p.pastHistoryNoteType AS note_type,
            p.pastHistoryPath AS history_path,
            p.pastHistoryValue AS history_value,
            p.pastHistoryValueText AS history_value_text
        FROM read_csv_auto(
            '{qpath(PH)}',
            header=true,
            sample_size=200000
        ) p
        JOIN cand c
          ON p.patientUnitStayID=c.stay_id
        WHERE p.pastHistoryOffset <= 1440
          AND p.pastHistoryEnteredOffset <= 1440
    """).df()

    ph_text = (
        ph_renal["history_path"].fillna("").astype(str) + " "
        + ph_renal["history_value"].fillna("").astype(str) + " "
        + ph_renal["history_value_text"].fillna("").astype(str)
    )
    ph_renal = ph_renal.loc[
        ph_text.str.contains(RENAL_HISTORY_RE, na=False)
    ].copy()

    ph_inventory = (
        ph_renal.groupby(
            ["history_path", "history_value", "history_value_text"],
            dropna=False,
        )
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            hospitals=("hospital_id", "nunique"),
        )
        .reset_index()
        .sort_values(["stays", "rows"], ascending=False)
    )
    ph_inventory.to_csv(
        out / "CHRONIC_RENAL_PASTHISTORY_INVENTORY.csv", index=False
    )

    # ---------------------------------------------------------------
    # 8) Current-ICU renal diagnosis evidence — inventory only.
    # ---------------------------------------------------------------
    dx = con.execute(f"""
        SELECT
            d.patientUnitStayID AS stay_id,
            c.hospital_id AS hospital_id,
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
        WHERE d.diagnosisOffset <= 1440
    """).df()

    dx_text = (
        dx["diagnosis_string"].fillna("").astype(str) + " "
        + dx["icd_code"].fillna("").astype(str)
    )
    dx_renal = dx.loc[
        dx_text.str.contains(RENAL_DX_RE, na=False)
    ].copy()

    dx_inventory = (
        dx_renal.groupby(
            ["diagnosis_string", "icd_code"],
            dropna=False,
        )
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            hospitals=("hospital_id", "nunique"),
        )
        .reset_index()
        .sort_values(["stays", "rows"], ascending=False)
    )
    dx_inventory.to_csv(
        out / "RENAL_DIAGNOSIS_INVENTORY.csv", index=False
    )

    # ---------------------------------------------------------------
    # 9) Stay-level RRT-source overlap.
    # ---------------------------------------------------------------
    cand_stays = candidate[["stay_id", "hospital_id"]].copy()

    trrt_stays = set(trrt["stay_id"].unique())
    io_text_stays = set(
        io.loc[io["dialysis_text_candidate"], "stay_id"].unique()
    )
    io_total_stays = set(
        io.loc[io["nonzero_dialysis_total"], "stay_id"].unique()
    )
    aps_pos_stays = set(
        aps.loc[aps["apache_dialysis_num"].eq(1), "stay_id"].unique()
    )
    ph_renal_stays = set(ph_renal["stay_id"].unique())
    dx_renal_stays = set(dx_renal["stay_id"].unique())

    ov = cand_stays.copy()
    ov["treatment_rrt_candidate"] = ov["stay_id"].isin(trrt_stays).astype(int)
    ov["io_dialysis_text_candidate"] = ov["stay_id"].isin(io_text_stays).astype(int)
    ov["io_nonzero_dialysis_total"] = ov["stay_id"].isin(io_total_stays).astype(int)
    ov["apache_dialysis_flag_1"] = ov["stay_id"].isin(aps_pos_stays).astype(int)
    ov["renal_pasthistory_candidate"] = ov["stay_id"].isin(ph_renal_stays).astype(int)
    ov["renal_diagnosis_candidate"] = ov["stay_id"].isin(dx_renal_stays).astype(int)

    overlap = (
        ov.groupby(
            [
                "treatment_rrt_candidate",
                "io_dialysis_text_candidate",
                "io_nonzero_dialysis_total",
                "apache_dialysis_flag_1",
                "renal_pasthistory_candidate",
                "renal_diagnosis_candidate",
            ]
        )
        .size()
        .rename("stays")
        .reset_index()
        .sort_values("stays", ascending=False)
    )
    overlap.to_csv(out / "RENAL_SOURCE_OVERLAP.csv", index=False)

    summary = pd.DataFrame([{
        "hf_plus24h_candidate_stays": n,
        "candidate_hospitals": nh,
        "current_creatinine_0_24h_stays": int(
            cur["stay_id"].nunique()
        ),
        "pre_icu_current_hospital_creatinine_stays": int(
            cr.loc[cr["pre_icu_current_hospital"], "stay_id"].nunique()
        ),
        "treatment_rrt_candidate_stays": len(trrt_stays),
        "io_dialysis_text_candidate_stays": len(io_text_stays),
        "io_nonzero_dialysis_total_stays": len(io_total_stays),
        "apache_dialysis_flag_1_stays": len(aps_pos_stays),
        "renal_pasthistory_candidate_stays": len(ph_renal_stays),
        "renal_diagnosis_candidate_stays": len(dx_renal_stays),
        "any_rrt_candidate_stays": len(
            trrt_stays | io_text_stays | io_total_stays | aps_pos_stays
        ),
    }])
    summary.to_csv(out / "RENAL_SOURCE_SUMMARY.csv", index=False)

    # ---------------------------------------------------------------
    # 10) Hospital-level source capture.
    # Presence of source data != negative clinical state.
    # ---------------------------------------------------------------
    def hospital_capture(stay_ids, name):
        tmp = cand_stays.copy()
        tmp[name] = tmp["stay_id"].isin(set(stay_ids)).astype(int)
        return (
            tmp.groupby("hospital_id")[name]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": f"{name}_stays", "count": "landmark_hf_stays"})
        )

    hospitals = (
        candidate.groupby("hospital_id")
        .size()
        .rename("hf_plus24h_stays")
        .reset_index()
    )

    # Any source row capture, not clinical positives.
    source_sets = {}

    source_sets["any_creatinine_row"] = set(cr["stay_id"].unique())
    source_sets["any_treatment_row"] = set(treatment["stay_id"].unique())
    source_sets["any_intakeoutput_row"] = set(io["stay_id"].unique())
    source_sets["any_apacheapsvar_row"] = set(aps["stay_id"].unique())
    source_sets["any_pasthistory_row"] = set(
        con.execute(f"""
            SELECT DISTINCT p.patientUnitStayID AS stay_id
            FROM read_csv_auto(
                '{qpath(PH)}',
                header=true,
                sample_size=200000
            ) p
            JOIN cand c
              ON p.patientUnitStayID=c.stay_id
            WHERE p.pastHistoryOffset <= 1440
              AND p.pastHistoryEnteredOffset <= 1440
        """).df()["stay_id"].unique()
    )

    site = hospitals.copy()
    for name, ids in source_sets.items():
        h = hospital_capture(ids, name)
        site = site.merge(
            h[["hospital_id", f"{name}_stays"]],
            on="hospital_id",
            how="left",
        )
        site[f"{name}_stays"] = site[f"{name}_stays"].fillna(0).astype(int)
        site[f"{name}_pct"] = (
            100.0 * site[f"{name}_stays"] / site["hf_plus24h_stays"]
        )

    site.to_csv(out / "RENAL_SOURCE_CAPTURE_BY_HOSPITAL.csv", index=False)

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
        "creatinine_timing": json.loads(timing.to_json(orient="records", double_precision=15)),
        "creatinine_same_offset": json.loads(dup_summary.to_json(orient="records", double_precision=15)),
        "apache_dialysis_flags": json.loads(apache_summary.to_json(orient="records", double_precision=15)),
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "outcome-blind renal source, creatinine, RRT and chronic-KRT "
            "feasibility audit"
        ),
        "adult_plus24h_hf_proxy_stays": n,
        "candidate_hospitals": nh,
        "outcome_or_discharge_status_selected": False,
        "discharge_location_selected": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "renal_contract_fixed_by_script": False,
        "kdigo_stage_computed": False,
        "acute_vs_chronic_rrt_adjudicated": False,
        "chronic_krt_exclusion_applied": False,
        "final_episode_selection_applied": False,
        "canonical_mimic_science_changed": False,
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
            "Source audit of candidate renal evidence and dialysis flags. "
            "The outcome-blind adjudication stages define acute RRT, chronic KRT "
            "exclusion, creatinine baseline, KDIGO stage and episode eligibility."
        ),
    }

    (out / "RENAL_SOURCE_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("EICU RENAL SOURCE / CREATININE / RRT AUDIT: PASS")
    print(f"HF +24 h candidate stays: {n:,}")
    print(f"Hospitals: {nh:,}")
    print(f"Output: {out}")
    print("Outcome/discharge-status values selected: NO")
    print("Discharge-location values selected: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("KDIGO stage computed: NO")
    print("Chronic-KRT exclusion applied: NO")
    print("Final episode selection applied: NO")
    print("Renal contract fixed by this script: NO")
    print("=" * 78)

if __name__ == "__main__":
    main()
