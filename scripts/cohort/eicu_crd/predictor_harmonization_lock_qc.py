#!/usr/bin/env python3
"""
Outcome-blind lock QC for the eICU predictor harmonization matrix.

This script does NOT:
- select hospitalDischargeStatus or unitDischargeStatus,
- derive mortality,
- fit any model,
- calculate performance metrics,
- choose or tune a harmonization rule from outcomes.

DuckDB query outputs use explicit snake_case aliases for consistent column
access in pandas.

It verifies only:
1) post-plausibility coverage for the eight fixed laboratory mappings,
2) laboratory revision-timing anomalies,
3) the explicit vasoactive drug-name allowlist and positive-rate semantics,
4) hospital-level infusion capture evidence for the later site-availability gate.
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

PROHIBITED = (
    "hospitaldischargestatus", "unitdischargestatus", "mortality",
    "death", "survival", "auroc", "auc", "brier", "calibration"
)

LAB_SPECS = {
    "hemoglobin_min_gdl": {
        "labName": "Hgb", "labTypeID": 3, "lower": 0.0, "upper": 50.0,
        "aggregation": "minimum"
    },
    "wbc_max_log2": {
        "labName": "WBC x 1000", "labTypeID": 3, "lower": 0.0, "upper": 1000.0,
        "aggregation": "maximum"
    },
    "platelet_min_log2": {
        "labName": "platelets x 1000", "labTypeID": 3, "lower": 0.0, "upper": 10000.0,
        "aggregation": "minimum"
    },
    "bun_max_log2": {
        "labName": "BUN", "labTypeID": 1, "lower": 0.0, "upper": 300.0,
        "aggregation": "maximum"
    },
    "bicarbonate_min_per5": {
        "labName": "bicarbonate", "labTypeID": 1, "lower": 0.0, "upper": 10000.0,
        "aggregation": "minimum"
    },
    "sodium_max_abs_deviation_from140_per5": {
        "labName": "sodium", "labTypeID": 1, "lower": 0.0, "upper": 200.0,
        "aggregation": "min_and_max"
    },
    "potassium_max_mmol_l": {
        "labName": "potassium", "labTypeID": 1, "lower": 0.0, "upper": 30.0,
        "aggregation": "maximum"
    },
    "glucose_max_log2": {
        "labName": "glucose", "labTypeID": 1, "lower": 0.0, "upper": 10000.0,
        "aggregation": "maximum"
    },
}

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

# Broad candidate screen used only to detect likely target labels that escaped the
# explicit classifier. Any such label is a hard QC failure.
# Standalone 'epi' is screened only at the START of a label. This deliberately
# avoids false-positive strings such as 'fentanyl epi (ml/hr)', where 'epi' is
# not sufficient evidence of epinephrine exposure.
TARGET_LIKE_RE = re.compile(
    r"norepinephrine|noradrenaline|levophed|(?:^|\W)levo(?:\W|$)|"
    r"epinephrine|epinepherine|adrenalin|adrenaline|^epi(?:\W|$)|"
    r"phenylephrine|neo[- ]?synephrine|neosynephrine|neosynsprine|"
    r"vasopressin|dopamine|dobu|milrinone|milronone|primacor|primacore",
    re.I
)

def discover(root: Path, stem: str) -> Path:
    wanted = {f"{stem}.csv".lower(), f"{stem}.csv.gz".lower()}
    matches = [p.resolve() for p in root.rglob("*") if p.is_file() and p.name.lower() in wanted]
    if len(matches) != 1:
        raise RuntimeError(f"{stem}: expected exactly one source file, found {len(matches)}")
    return matches[0]

def qpath(p: Path) -> str:
    return str(p).replace("'", "''")

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def norm_label(x: str) -> str:
    s = unicodedata.normalize("NFKC", str(x or ""))
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s

def classify_label(x: str):
    s = norm_label(x)
    hits = [name for name, pat in DRUG_PATTERNS if pat.search(s)]
    if len(hits) > 1:
        return s, None, hits
    return s, (hits[0] if hits else None), hits

def finite_positive(x):
    try:
        v = float(str(x).strip())
        return math.isfinite(v) and v > 0
    except Exception:
        return False

def main():
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

    patient_p = discover(root, "patient")
    lab_p = discover(root, "lab")
    infusion_p = discover(root, "infusionDrug")
    treatment_p = discover(root, "treatment")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    # Outcome-blind denominator: uses only age and discharge OFFSETS for +24h availability.
    base = con.execute(f"""
        SELECT
            patientUnitStayID AS stay_id,
            hospitalID AS hospital_id,
            age AS age,
            unitDischargeOffset AS unit_discharge_offset,
            hospitalDischargeOffset AS hospital_discharge_offset
        FROM read_csv_auto('{qpath(patient_p)}', header=true, sample_size=200000)
    """).df()

    def age_for_eligibility(x):
        s = str(x).strip()
        if s.startswith(">"):
            return 90.0
        try:
            return float(s)
        except Exception:
            return None

    base["age_num"] = base["age"].map(age_for_eligibility)
    base = base[
        base["age_num"].fillna(-1).ge(18)
        & pd.to_numeric(base["unit_discharge_offset"], errors="coerce").gt(1440)
        & pd.to_numeric(base["hospital_discharge_offset"], errors="coerce").gt(1440)
    ][["stay_id", "hospital_id"]].copy()

    if base["stay_id"].duplicated().any():
        raise AssertionError("Duplicate patientUnitStayID in patient table")

    n_base = int(len(base))
    n_hosp = int(base["hospital_id"].nunique())
    con.register("base_df", base)
    con.execute("CREATE TEMP TABLE base AS SELECT * FROM base_df")

    # ---------- LAB QC ----------
    lab_names = [spec["labName"] for spec in LAB_SPECS.values()]
    names_sql = ",".join("'" + x.replace("'", "''") + "'" for x in sorted(set(lab_names)))

    lab = con.execute(f"""
        SELECT
            l.patientUnitStayID AS stay_id,
            b.hospital_id AS hospital_id,
            l.labName AS lab_name,
            l.labTypeID AS lab_type_id,
            l.labResultOffset AS lab_result_offset,
            l.labResultRevisedOffset AS lab_result_revised_offset,
            l.labResult AS lab_result,
            l.labMeasureNameSystem AS lab_measure_name_system,
            l.labMeasureNameInterface AS lab_measure_name_interface
        FROM read_csv_auto('{qpath(lab_p)}', header=true, sample_size=200000) l
        JOIN base b ON l.patientUnitStayID=b.stay_id
        WHERE l.labResultOffset BETWEEN 0 AND 1440
          AND l.labName IN ({names_sql})
    """).df()

    coverage_rows = []
    timing_rows = []
    unit_rows = []

    for pred, spec in LAB_SPECS.items():
        x = lab[
            (lab["lab_name"] == spec["labName"])
            & (pd.to_numeric(lab["lab_type_id"], errors="coerce") == spec["labTypeID"])
        ].copy()
        result = pd.to_numeric(x["lab_result"], errors="coerce")
        revised = pd.to_numeric(x["lab_result_revised_offset"], errors="coerce")
        drawn = pd.to_numeric(x["lab_result_offset"], errors="coerce")

        timestamp_ok = revised.le(1440)
        numeric = result.notna()
        pre = x[timestamp_ok & numeric].copy()
        pre_val = pd.to_numeric(pre["lab_result"], errors="coerce")
        plaus = pre_val.gt(spec["lower"]) & pre_val.le(spec["upper"])
        post = pre[plaus].copy()

        coverage_rows.append({
            "model_column": pred,
            "labName": spec["labName"],
            "labTypeID": spec["labTypeID"],
            "aggregation": spec["aggregation"],
            "fixed_lower_exclusive": spec["lower"],
            "fixed_upper_inclusive": spec["upper"],
            "pre_plausibility_rows": int(len(pre)),
            "post_plausibility_rows": int(len(post)),
            "rows_excluded_by_plausibility": int(len(pre) - len(post)),
            "pre_plausibility_stays": int(pre["stay_id"].nunique()),
            "post_plausibility_stays": int(post["stay_id"].nunique()),
            "post_plausibility_pct": float(post["stay_id"].nunique() / n_base * 100.0),
            "pre_plausibility_hospitals": int(pre["hospital_id"].nunique()),
            "post_plausibility_hospitals": int(post["hospital_id"].nunique()),
        })

        timing_rows.append({
            "model_column": pred,
            "drawn_0_24h_rows": int(len(x)),
            "revision_after_landmark_rows": int(revised.gt(1440).sum()),
            "revision_at_or_before_landmark_rows": int(revised.le(1440).sum()),
            "negative_revision_offset_rows": int(revised.lt(0).sum()),
            "revision_before_draw_rows": int((revised < drawn).sum()),
            "missing_revision_offset_rows": int(revised.isna().sum()),
        })

        units = (
            x.groupby(["lab_measure_name_system", "lab_measure_name_interface"], dropna=False)
             .agg(rows=("stay_id", "size"), stays=("stay_id", "nunique"),
                  hospitals=("hospital_id", "nunique"))
             .reset_index()
        )
        units.insert(0, "model_column", pred)
        units.insert(1, "labName", spec["labName"])
        unit_rows.append(units)

    cov_df = pd.DataFrame(coverage_rows)
    time_df = pd.DataFrame(timing_rows)
    unit_df = pd.concat(unit_rows, ignore_index=True)

    cov_df.to_csv(out / "LAB_POST_PLAUSIBILITY_COVERAGE.csv", index=False)

    # ---------- VASOACTIVE QC ----------
    inf = con.execute(f"""
        SELECT
            i.patientUnitStayID AS stay_id,
            b.hospital_id AS hospital_id,
            i.infusionOffset AS infusion_offset,
            i.drugName AS drug_name,
            i.drugRate AS drug_rate,
            i.infusionRate AS infusion_rate
        FROM read_csv_auto('{qpath(infusion_p)}', header=true, sample_size=200000) i
        JOIN base b ON i.patientUnitStayID=b.stay_id
        WHERE i.infusionOffset BETWEEN 0 AND 1440
    """).df()

    # Classify all 0-24 h infusion rows under the fixed seven-agent allowlist.
    classified_all = inf.copy()
    normalized = []
    classes = []
    ambiguous = []
    for name in classified_all["drug_name"].tolist():
        n, cls, hits = classify_label(name)
        normalized.append(n)
        classes.append(cls)
        ambiguous.append("|".join(hits) if len(hits) > 1 else "")
    classified_all["normalized_drug_name"] = normalized
    classified_all["drug_class"] = classes
    classified_all["ambiguous_classes"] = ambiguous

    amb = classified_all[classified_all["ambiguous_classes"].ne("")]
    if len(amb):
        amb[["drug_name", "normalized_drug_name", "ambiguous_classes"]].drop_duplicates().to_csv(
            out / "VASOACTIVE_AMBIGUOUS_LABELS_FAIL.csv", index=False
        )
        raise AssertionError("Ambiguous vasoactive drug-name classification detected")

    candidate_like = classified_all[
        classified_all["drug_name"].astype("string").str.contains(TARGET_LIKE_RE, na=False)
    ].copy()
    unclassified_candidate = candidate_like[candidate_like["drug_class"].isna()].copy()
    if len(unclassified_candidate):
        (
            unclassified_candidate[["drug_name", "normalized_drug_name"]]
            .drop_duplicates()
            .sort_values("normalized_drug_name")
            .to_csv(out / "VASOACTIVE_UNCLASSIFIED_TARGETLIKE_LABELS_FAIL.csv", index=False)
        )
        raise AssertionError(
            "Target-like vasoactive labels escaped the fixed allowlist; review before lock"
        )

    target_like = classified_all[classified_all["drug_class"].notna()].copy()
    target_like["drug_rate_positive"] = target_like["drug_rate"].map(finite_positive)
    target_like["infusion_rate_positive"] = target_like["infusion_rate"].map(finite_positive)
    target_like["positive_evidence"] = (
        target_like["drug_class"].notna()
        & (target_like["drug_rate_positive"] | target_like["infusion_rate_positive"])
    )

    label_qc = (
        target_like.groupby(["drug_class", "drug_name", "normalized_drug_name"], dropna=False)
        .agg(
            rows=("stay_id", "size"),
            stays=("stay_id", "nunique"),
            hospitals=("hospital_id", "nunique"),
            positive_rows=("positive_evidence", "sum"),
            drug_rate_positive_rows=("drug_rate_positive", "sum"),
            infusion_rate_positive_rows=("infusion_rate_positive", "sum"),
        )
        .reset_index()
        .sort_values(["drug_class", "stays", "rows"], ascending=[True, False, False])
    )

    dpos = target_like["drug_rate_positive"]
    ipos = target_like["infusion_rate_positive"]
    rate_qc = pd.DataFrame([{
        "target_like_rows": int(len(candidate_like)),
        "classified_rows": int(candidate_like["drug_class"].notna().sum()),
        "unclassified_target_like_rows": int(candidate_like["drug_class"].isna().sum()),
        "positive_rows": int(target_like["positive_evidence"].sum()),
        "both_rate_fields_positive_rows": int((dpos & ipos).sum()),
        "drug_rate_only_positive_rows": int((dpos & ~ipos).sum()),
        "infusion_rate_only_positive_rows": int((~dpos & ipos).sum()),
        "neither_positive_rows": int((~dpos & ~ipos).sum()),
        "positive_stays": int(target_like.loc[target_like["positive_evidence"], "stay_id"].nunique()),
        "positive_hospitals": int(target_like.loc[target_like["positive_evidence"], "hospital_id"].nunique()),
    }])

    # Site capture: any infusion row, target-positive row, and treatment QC-only evidence.
    any_inf = inf[["stay_id", "hospital_id"]].drop_duplicates()
    pos_inf = target_like.loc[target_like["positive_evidence"], ["stay_id", "hospital_id"]].drop_duplicates()

    treatment = con.execute(f"""
        SELECT
            t.patientUnitStayID AS stay_id,
            b.hospital_id AS hospital_id,
            t.treatmentString AS treatment_string
        FROM read_csv_auto('{qpath(treatment_p)}', header=true, sample_size=200000) t
        JOIN base b ON t.patientUnitStayID=b.stay_id
        WHERE t.treatmentOffset BETWEEN 0 AND 1440
    """).df()
    treatment_pos = treatment[
        treatment["treatment_string"].astype("string").str.contains(TARGET_LIKE_RE, na=False)
    ][["stay_id", "hospital_id"]].drop_duplicates()

    base_h = base.groupby("hospital_id").size().rename("landmark_stays").reset_index()

    def hcount(df, name):
        return (
            df.groupby("hospital_id")["stay_id"].nunique()
              .rename(name).reset_index()
        )

    site = base_h.merge(hcount(any_inf, "stays_with_any_infusion_row"), on="hospital_id", how="left")
    site = site.merge(hcount(pos_inf, "stays_with_target_positive_evidence"), on="hospital_id", how="left")
    site = site.merge(hcount(treatment_pos, "stays_with_target_treatment_qc_evidence"), on="hospital_id", how="left")
    for c in [
        "stays_with_any_infusion_row",
        "stays_with_target_positive_evidence",
        "stays_with_target_treatment_qc_evidence",
    ]:
        site[c] = site[c].fillna(0).astype(int)
        site[c + "_pct"] = 100.0 * site[c] / site["landmark_stays"]


    zero_inf = site[site["stays_with_any_infusion_row"].eq(0)]
    site_summary = pd.DataFrame([{
        "adult_plus24h_stays": n_base,
        "hospitals": n_hosp,
        "hospitals_with_zero_infusion_rows": int(len(zero_inf)),
        "stays_in_zero_infusion_hospitals": int(zero_inf["landmark_stays"].sum()),
        "zero_infusion_hospitals_with_treatment_qc_evidence": int(
            (zero_inf["stays_with_target_treatment_qc_evidence"] > 0).sum()
        ),
        "target_positive_stays_under_fixed_rule": int(pos_inf["stay_id"].nunique()),
        "target_positive_hospitals_under_fixed_rule": int(pos_inf["hospital_id"].nunique()),
    }])
    site_summary.to_csv(out / "VASOACTIVE_SITE_CAPTURE_SUMMARY.csv", index=False)

    # ---------- Safety + manifest ----------
    for p in out.glob("*.csv"):
        cols = [str(c).lower() for c in pd.read_csv(p, nrows=0).columns]
        bad = [c for c in cols if any(tok in c for tok in PROHIBITED)]
        if bad:
            raise AssertionError(f"{p.name}: prohibited output columns: {bad}")

    rule_payload = {
        "labs": LAB_SPECS,
        "drug_patterns": [(name, pat.pattern) for name, pat in DRUG_PATTERNS],
        "target_like_screen": TARGET_LIKE_RE.pattern,
        "positive_rule": "finite(drugRate)>0 OR finite(infusionRate)>0",
        "infusion_offset_window": [0, 1440],
        "lab_draw_window": [0, 1440],
        "lab_revision_max": 1440,
    }
    rule_sha = hashlib.sha256(
        json.dumps(rule_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    hashes = {p.name: sha256_file(p) for p in sorted(out.glob("*.csv"))}
    manifest = {
        "lab_result_timing": json.loads(time_df.to_json(orient="records", double_precision=15)),
        "lab_units": json.loads(unit_df.to_json(orient="records", double_precision=15)),
        "vasoactive_label_classification": json.loads(label_qc.to_json(orient="records", double_precision=15)),
        "vasoactive_rate_evidence": json.loads(rate_qc.to_json(orient="records", double_precision=15)),
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "outcome-blind predictor harmonization lock QC",
        "adult_plus24h_stays": n_base,
        "hospitals": n_hosp,
        "outcome_or_discharge_status_selected": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "scientific_rule_tuned_from_outcome": False,
        "rule_sha256": rule_sha,
        "generated_sha256": hashes,
        "interpretation": (
            "QC only. Site capture eligibility itself is not chosen by this script. "
            "The script verifies the fixed predictor-mapping rules before the later site gate."
        ),
    }
    (out / "LOCK_QC_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("=" * 78)
    print("EICU PREDICTOR HARMONIZATION LOCK QC: PASS")
    print(f"Adult +24 h stays: {n_base:,}")
    print(f"Hospitals: {n_hosp:,}")
    print(f"Output: {out}")
    print("Outcome/discharge-status values selected: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("Site-capture threshold selected by this script: NO")
    print("=" * 78)

if __name__ == "__main__":
    main()
