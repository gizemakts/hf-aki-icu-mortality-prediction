#!/usr/bin/env python3
"""
Outcome-blind eICU HF-equivalence audit.

Purpose
-------
Provide the empirical evidence needed to lock the eICU HF-equivalence
contract relative to the fixed MIMIC primary HF definition.

This audit is READ-ONLY and PRE-OUTCOME. It does not select discharge status,
mortality, death, survival, model predictions, or performance metrics. It does
not fit a model.

Primary scientific question
---------------------------
Fixed MIMIC primary HF eligibility requires HF documented during a PRIOR
COMPLETED hospital admission. Core eICU cannot reproduce that longitudinal
instrument exactly. This audit evaluates `pastHistory` as the closest
pre-existing-HF proxy and keeps current-ICU diagnosis sources separate.

The script intentionally does NOT choose the final HF contract. It generates
outcome-blind evidence for that decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROHIBITED_TOKENS = (
    "hospitaldischargestatus",
    "unitdischargestatus",
    "mortality",
    "death",
    "survival",
    "auroc",
    "auc",
    "brier",
    "calibration",
)

HF_TEXT_RE = (
    r"(^|[^a-z])"
    r"(chf|congestive heart failure|heart failure)"
    r"([^a-z]|$)"
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

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def assert_safe_columns(columns, label: str) -> None:
    bad = [
        str(c) for c in columns
        if any(tok in str(c).lower() for tok in PROHIBITED_TOKENS)
    ]
    if bad:
        raise AssertionError(f"{label}: prohibited output columns detected: {bad}")

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

    P = discover(root, "patient")
    PH = discover(root, "pastHistory")
    D = discover(root, "diagnosis")
    A = discover(root, "admissionDx")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    # ---------------------------------------------------------------
    # 1) Outcome-blind adult +24 h denominator.
    # Only age and discharge OFFSETS are selected.
    # ---------------------------------------------------------------
    patient = con.execute(f"""
        SELECT
            patientUnitStayID AS stay_id,
            hospitalID AS hospital_id,
            age AS age,
            unitDischargeOffset AS unit_discharge_offset,
            hospitalDischargeOffset AS hospital_discharge_offset
        FROM read_csv_auto('{qpath(P)}', header=true, sample_size=200000)
    """).df()
    assert_safe_columns(patient.columns, "patient_safe")

    def age_num(x):
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        if s.startswith(">"):
            return 90.0   # eligibility only; not the model age transform
        try:
            return float(s)
        except Exception:
            return None

    patient["age_num"] = patient["age"].map(age_num)
    patient["adult_plus24h"] = (
        patient["age_num"].fillna(-1).ge(18)
        & pd.to_numeric(
            patient["unit_discharge_offset"], errors="coerce"
        ).gt(1440)
        & pd.to_numeric(
            patient["hospital_discharge_offset"], errors="coerce"
        ).gt(1440)
    )

    base = patient.loc[
        patient["adult_plus24h"], ["stay_id", "hospital_id"]
    ].copy()

    if base["stay_id"].duplicated().any():
        raise AssertionError("Duplicate patientUnitStayID in patient table")

    n_base = int(len(base))
    n_hosp = int(base["hospital_id"].nunique())

    con.register("base_df", base)
    con.execute("CREATE TEMP TABLE base AS SELECT * FROM base_df")

    # ---------------------------------------------------------------
    # 2) pastHistory — full timing/label evidence.
    # Do not pre-filter to +24 h, because timing discordance itself is QC.
    # ---------------------------------------------------------------
    ph = con.execute(f"""
        SELECT
            p.patientUnitStayID AS stay_id,
            b.hospital_id AS hospital_id,
            p.pastHistoryOffset AS history_offset,
            p.pastHistoryEnteredOffset AS entered_offset,
            p.pastHistoryNoteType AS note_type,
            p.pastHistoryPath AS history_path,
            p.pastHistoryValue AS history_value,
            p.pastHistoryValueText AS history_value_text
        FROM read_csv_auto('{qpath(PH)}', header=true, sample_size=200000) p
        JOIN base b ON p.patientUnitStayID=b.stay_id
    """).df()
    assert_safe_columns(ph.columns, "pastHistory_safe")

    concat_text = (
        ph["history_path"].fillna("").astype(str) + " "
        + ph["history_value"].fillna("").astype(str) + " "
        + ph["history_value_text"].fillna("").astype(str)
    ).str.lower()

    ph["text_hf_candidate"] = concat_text.str.contains(
        HF_TEXT_RE, regex=True, na=False
    )

    # Structural path flag is intentionally separate from the broad text
    # candidate so the final contract can choose the stricter source.
    ph["structured_chf_path"] = (
        ph["history_path"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains("congestive heart failure", regex=False)
    )

    hist_off = pd.to_numeric(ph["history_offset"], errors="coerce")
    ent_off = pd.to_numeric(ph["entered_offset"], errors="coerce")

    ph["history_by24"] = hist_off.le(1440)
    ph["entered_by24"] = ent_off.le(1440)
    ph["both_by24"] = ph["history_by24"] & ph["entered_by24"]

    # ---------------------------------------------------------------
    # 3) Exact HF label inventory under each timing interpretation.
    # ---------------------------------------------------------------
    hf_labels = ph.loc[ph["text_hf_candidate"]].copy()
    label_inventory = (
        hf_labels.groupby(
            ["history_path", "history_value", "history_value_text"],
            dropna=False,
        )
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            hospitals=("hospital_id", "nunique"),
            rows_entered_by24=("entered_by24", "sum"),
            rows_both_offsets_by24=("both_by24", "sum"),
        )
        .reset_index()
        .sort_values(["stays", "rows"], ascending=False)
    )
    label_inventory.to_csv(
        out / "HF_PASTHISTORY_LABEL_INVENTORY.csv", index=False
    )

    note_types = (
        hf_labels.groupby("note_type", dropna=False)
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            hospitals=("hospital_id", "nunique"),
            rows_entered_by24=("entered_by24", "sum"),
            rows_both_offsets_by24=("both_by24", "sum"),
        )
        .reset_index()
        .sort_values(["stays", "rows"], ascending=False)
    )
    note_types.to_csv(
        out / "HF_PASTHISTORY_NOTE_TYPE_AUDIT.csv", index=False
    )

    # ---------------------------------------------------------------
    # 4) Timing audit.
    # ---------------------------------------------------------------
    timing = pd.DataFrame([{
        "adult_plus24h_stays": n_base,
        "all_pasthistory_rows": int(len(ph)),
        "all_pasthistory_stays": int(ph["stay_id"].nunique()),
        "all_pasthistory_hospitals": int(ph["hospital_id"].nunique()),
        "hf_candidate_rows": int(ph["text_hf_candidate"].sum()),
        "hf_candidate_stays": int(
            ph.loc[ph["text_hf_candidate"], "stay_id"].nunique()
        ),
        "structured_chf_path_rows": int(ph["structured_chf_path"].sum()),
        "structured_chf_path_stays": int(
            ph.loc[ph["structured_chf_path"], "stay_id"].nunique()
        ),
        "hf_rows_entered_by24": int(
            (ph["text_hf_candidate"] & ph["entered_by24"]).sum()
        ),
        "hf_stays_entered_by24": int(
            ph.loc[
                ph["text_hf_candidate"] & ph["entered_by24"], "stay_id"
            ].nunique()
        ),
        "hf_rows_both_offsets_by24": int(
            (ph["text_hf_candidate"] & ph["both_by24"]).sum()
        ),
        "hf_stays_both_offsets_by24": int(
            ph.loc[
                ph["text_hf_candidate"] & ph["both_by24"], "stay_id"
            ].nunique()
        ),
        "hf_entered_by24_but_history_after24_rows": int(
            (
                ph["text_hf_candidate"]
                & ph["entered_by24"]
                & hist_off.gt(1440)
            ).sum()
        ),
        "hf_history_by24_but_entered_after24_rows": int(
            (
                ph["text_hf_candidate"]
                & ph["history_by24"]
                & ent_off.gt(1440)
            ).sum()
        ),
        "hf_negative_history_offset_rows": int(
            (ph["text_hf_candidate"] & hist_off.lt(0)).sum()
        ),
        "hf_negative_entered_offset_rows": int(
            (ph["text_hf_candidate"] & ent_off.lt(0)).sum()
        ),
        "hf_entered_before_history_offset_rows": int(
            (ph["text_hf_candidate"] & (ent_off < hist_off)).sum()
        ),
        "hf_missing_history_offset_rows": int(
            (ph["text_hf_candidate"] & hist_off.isna()).sum()
        ),
        "hf_missing_entered_offset_rows": int(
            (ph["text_hf_candidate"] & ent_off.isna()).sum()
        ),
    }])

    # ---------------------------------------------------------------
    # 5) Stay-level pastHistory evidence and hospital capture.
    # ---------------------------------------------------------------
    ph24 = ph.loc[ph["entered_by24"]].copy()
    ph_stay = (
        ph24.groupby(["stay_id", "hospital_id"], as_index=False)
        .agg(
            any_pasthistory_rows_by24=("stay_id", "size"),
            hf_text_evidence_by24=("text_hf_candidate", "max"),
            hf_structured_path_by24=("structured_chf_path", "max"),
        )
    )

    stay = base.merge(
        ph_stay, on=["stay_id", "hospital_id"], how="left"
    )
    for c in [
        "any_pasthistory_rows_by24",
        "hf_text_evidence_by24",
        "hf_structured_path_by24",
    ]:
        stay[c] = stay[c].fillna(0).astype(int)

    hospital = (
        stay.groupby("hospital_id")
        .agg(
            landmark_stays=("stay_id", "size"),
            stays_with_any_pasthistory_by24=(
                "any_pasthistory_rows_by24",
                lambda x: int((x > 0).sum()),
            ),
            hf_text_evidence_stays=(
                "hf_text_evidence_by24",
                "sum",
            ),
            hf_structured_path_stays=(
                "hf_structured_path_by24",
                "sum",
            ),
        )
        .reset_index()
    )
    hospital["any_pasthistory_pct"] = (
        100.0
        * hospital["stays_with_any_pasthistory_by24"]
        / hospital["landmark_stays"]
    )
    hospital["hf_text_evidence_pct"] = (
        100.0
        * hospital["hf_text_evidence_stays"]
        / hospital["landmark_stays"]
    )
    hospital["hf_structured_path_pct"] = (
        100.0
        * hospital["hf_structured_path_stays"]
        / hospital["landmark_stays"]
    )
    hospital.to_csv(
        out / "HF_PASTHISTORY_COVERAGE_BY_HOSPITAL.csv", index=False
    )

    hosp_summary = pd.DataFrame([{
        "adult_plus24h_stays": n_base,
        "hospitals": n_hosp,
        "stays_with_any_pasthistory_by24": int(
            (stay["any_pasthistory_rows_by24"] > 0).sum()
        ),
        "pct_stays_with_any_pasthistory_by24": float(
            100.0 * (stay["any_pasthistory_rows_by24"] > 0).mean()
        ),
        "hf_text_evidence_stays_by24": int(
            stay["hf_text_evidence_by24"].sum()
        ),
        "hf_structured_path_stays_by24": int(
            stay["hf_structured_path_by24"].sum()
        ),
        "hospitals_zero_pasthistory_capture_by24": int(
            (hospital["any_pasthistory_pct"] == 0).sum()
        ),
        "stays_in_zero_pasthistory_hospitals": int(
            hospital.loc[
                hospital["any_pasthistory_pct"] == 0, "landmark_stays"
            ].sum()
        ),
        "hospitals_pasthistory_capture_lt10pct": int(
            (hospital["any_pasthistory_pct"] < 10).sum()
        ),
        "hospitals_pasthistory_capture_ge90pct": int(
            (hospital["any_pasthistory_pct"] >= 90).sum()
        ),
    }])
    hosp_summary.to_csv(
        out / "HF_PASTHISTORY_CAPTURE_SUMMARY.csv", index=False
    )

    # ---------------------------------------------------------------
    # 6) Current-ICU HF evidence kept separate.
    # diagnosis = active diagnosis during ICU stay.
    # admissionDx = APACHE primary diagnosis for ICU admission.
    # These are discordance/sensitivity evidence only.
    # ---------------------------------------------------------------
    dx = con.execute(f"""
        SELECT
            d.patientUnitStayID AS stay_id,
            MAX(
                CASE WHEN d.diagnosisOffset<=1440 AND (
                    regexp_matches(
                        lower(coalesce(d.diagnosisString,'')),
                        '{HF_TEXT_RE}'
                    )
                    OR regexp_matches(
                        replace(coalesce(d.ICD9Code,''),'.',''),
                        '(^|,| )428[0-9]*($|,| )'
                    )
                )
                THEN 1 ELSE 0 END
            ) AS current_icu_hf_dx_by24
        FROM read_csv_auto(
            '{qpath(D)}', header=true, sample_size=200000
        ) d
        JOIN base b ON d.patientUnitStayID=b.stay_id
        GROUP BY 1
    """).df()

    adx = con.execute(f"""
        SELECT
            a.patientUnitStayID AS stay_id,
            MAX(
                CASE WHEN a.admitDxEnteredOffset<=1440
                  AND regexp_matches(
                    lower(
                        coalesce(a.admitDxPath,'') || ' ' ||
                        coalesce(a.admitDxName,'') || ' ' ||
                        coalesce(a.admitDxText,'')
                    ),
                    '{HF_TEXT_RE}'
                  )
                THEN 1 ELSE 0 END
            ) AS apache_admission_hf_by24
        FROM read_csv_auto(
            '{qpath(A)}', header=true, sample_size=200000
        ) a
        JOIN base b ON a.patientUnitStayID=b.stay_id
        GROUP BY 1
    """).df()

    overlap = stay[
        [
            "stay_id",
            "hospital_id",
            "hf_text_evidence_by24",
            "hf_structured_path_by24",
        ]
    ].merge(dx, on="stay_id", how="left").merge(
        adx, on="stay_id", how="left"
    )

    overlap["current_icu_hf_dx_by24"] = (
        overlap["current_icu_hf_dx_by24"].fillna(0).astype(int)
    )
    overlap["apache_admission_hf_by24"] = (
        overlap["apache_admission_hf_by24"].fillna(0).astype(int)
    )

    overlap_summary = (
        overlap.groupby(
            [
                "hf_structured_path_by24",
                "current_icu_hf_dx_by24",
                "apache_admission_hf_by24",
            ]
        )
        .size()
        .rename("stays")
        .reset_index()
        .sort_values("stays", ascending=False)
    )
    overlap_summary.to_csv(
        out / "HF_SOURCE_OVERLAP.csv", index=False
    )

    source_summary = pd.DataFrame([{
        "adult_plus24h_stays": n_base,
        "pastHistory_structured_hf_stays_by24": int(
            overlap["hf_structured_path_by24"].sum()
        ),
        "current_icu_diagnosis_hf_stays_by24": int(
            overlap["current_icu_hf_dx_by24"].sum()
        ),
        "apache_admission_hf_stays_by24": int(
            overlap["apache_admission_hf_by24"].sum()
        ),
        "pastHistory_only_vs_two_current_sources": int(
            (
                overlap["hf_structured_path_by24"].eq(1)
                & overlap["current_icu_hf_dx_by24"].eq(0)
                & overlap["apache_admission_hf_by24"].eq(0)
            ).sum()
        ),
        "current_source_positive_without_pastHistory_hf": int(
            (
                overlap["hf_structured_path_by24"].eq(0)
                & (
                    overlap["current_icu_hf_dx_by24"].eq(1)
                    | overlap["apache_admission_hf_by24"].eq(1)
                )
            ).sum()
        ),
    }])
    source_summary.to_csv(
        out / "HF_SOURCE_SUMMARY.csv", index=False
    )

    # ---------------------------------------------------------------
    # 7) Safety and manifest.
    # ---------------------------------------------------------------
    for path in out.glob("*.csv"):
        df0 = pd.read_csv(path, nrows=5)
        assert_safe_columns(df0.columns, path.name)

    hashes = {
        p.name: sha256_file(p)
        for p in sorted(out.glob("*.csv"), key=lambda x: x.name)
    }

    manifest = {
        "patient_history_timing": json.loads(timing.to_json(orient="records", double_precision=15)),
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "outcome-blind eICU HF-equivalence evidence",
        "adult_plus24h_stays": n_base,
        "hospitals": n_hosp,
        "outcome_or_discharge_status_selected": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "hf_contract_fixed_by_script": False,
        "primary_mimic_hf_changed": False,
        "source_files": {
            "patient": P.name,
            "pastHistory": PH.name,
            "diagnosis": D.name,
            "admissionDx": A.name,
        },
        "generated_sha256": hashes,
        "interpretation": (
            "Evidence-only audit. The final HF-equivalence rule is not chosen "
            "by this script. Current-ICU diagnosis sources are kept separate "
            "from prior-history evidence."
        ),
    }

    (out / "HF_EQUIVALENCE_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("=" * 78)
    print("EICU HF EQUIVALENCE AUDIT: PASS")
    print(f"Adult +24 h stays: {n_base:,}")
    print(f"Hospitals: {n_hosp:,}")
    print(f"Output: {out}")
    print("Outcome/discharge-status values selected: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("HF contract fixed by this script: NO")
    print("=" * 78)

if __name__ == "__main__":
    main()
