#!/usr/bin/env python3
"""
Outcome-blind eICU episode-selection / +24 h landmark audit.

Purpose
-------
Generate the evidence needed to lock the eICU episode-selection and landmark
contract while preserving the fixed MIMIC principle: select the FIRST
QUALIFYING ICU episode per patient.

This script does NOT lock the episode rule. It audits:
1) +24 h landmark eligibility mechanics;
2) eICU ICU-unit ordering within one hospital admission;
3) patient-level ordering across hospital admissions;
4) repeated-stay ambiguity caused by eICU's de-identified chronology;
5) unitType / unitStayType composition;
6) multiplicity among the already fixed HF-proxy +24 h candidate stays.

Outcome safety
--------------
The script does NOT select:
- hospitalDischargeStatus
- unitDischargeStatus
- hospitalDischargeLocation
- unitDischargeLocation
- mortality/death/survival
and does not fit any model or compute performance.

The only discharge information used is:
- unitDischargeOffset
- hospitalDischargeOffset
- hospitalDischargeYear
for landmark duration and chronology QC, never outcome status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# Exact fixed HF proxy allowlist from the FIXED HF-equivalence contract.
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

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    # ------------------------------------------------------------------
    # 1) Patient table — explicitly select ONLY outcome-safe fields.
    # ------------------------------------------------------------------
    p = con.execute(f"""
        SELECT
            patientUnitStayID AS stay_id,
            patientHealthSystemStayID AS hstay_id,
            uniquePid AS unique_pid,
            hospitalID AS hospital_id,
            age AS age,
            hospitalAdmitOffset AS hospital_admit_offset,
            hospitalDischargeOffset AS hospital_discharge_offset,
            hospitalDischargeYear AS hospital_discharge_year,
            unitVisitNumber AS unit_visit_number,
            unitStayType AS unit_stay_type,
            unitType AS unit_type,
            unitDischargeOffset AS unit_discharge_offset
        FROM read_csv_auto(
            '{qpath(PATIENT)}',
            header=true,
            sample_size=200000
        )
    """).df()
    assert_safe_columns(p.columns, "patient_safe")

    if p["stay_id"].duplicated().any():
        raise AssertionError("patientUnitStayID is not unique")
    if p["unique_pid"].isna().any():
        raise AssertionError("uniquePid unexpectedly missing")
    if p["hstay_id"].isna().any():
        raise AssertionError("patientHealthSystemStayID unexpectedly missing")

    def age_num(x):
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        if s.startswith(">"):
            return 90.0  # adult eligibility only, not model-age transform
        try:
            return float(s)
        except Exception:
            return None

    p["age_num"] = p["age"].map(age_num)
    p["adult"] = p["age_num"].fillna(-1).ge(18)

    unit_off = pd.to_numeric(p["unit_discharge_offset"], errors="coerce")
    hosp_off = pd.to_numeric(p["hospital_discharge_offset"], errors="coerce")

    # Existing outcome-blind eICU feasibility denominator:
    # adult and demonstrably still in ICU + hospital after +24 h.
    p["landmark_plus24"] = (
        p["adult"]
        & unit_off.gt(1440)
        & hosp_off.gt(1440)
    )

    adult = p[p["adult"]].copy()
    base = p[p["landmark_plus24"]].copy()

    # ------------------------------------------------------------------
    # 2) Fixed HF-proxy recheck.
    # ------------------------------------------------------------------
    con.register(
        "landmark_ids",
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
        JOIN landmark_ids b
          ON ph.patientUnitStayID=b.patientUnitStayID
        WHERE ph.pastHistoryOffset <= 1440
          AND ph.pastHistoryEnteredOffset <= 1440
          AND ph.pastHistoryValue IN ({values_sql})
          AND ph.pastHistoryValueText = ph.pastHistoryValue
          AND ph.pastHistoryPath =
              '{HF_PATH_PREFIX.replace("'", "''")}' || ph.pastHistoryValue
    """).df()
    assert_safe_columns(hf.columns, "hf_safe")

    candidate = base.merge(hf.assign(hf_proxy=1), on="stay_id", how="inner")
    if candidate["stay_id"].duplicated().any():
        raise AssertionError("HF candidate stay duplicated after aggregation")

    # ------------------------------------------------------------------
    # 3) Landmark mechanics.
    # ------------------------------------------------------------------
    landmark_summary = pd.DataFrame([{
        "all_patient_unit_stays": int(len(p)),
        "adult_unit_stays": int(p["adult"].sum()),
        "adult_unit_discharge_exactly_1440": int(
            (p["adult"] & unit_off.eq(1440)).sum()
        ),
        "adult_hospital_discharge_exactly_1440": int(
            (p["adult"] & hosp_off.eq(1440)).sum()
        ),
        "adult_unit_gt1440": int(
            (p["adult"] & unit_off.gt(1440)).sum()
        ),
        "adult_hospital_gt1440": int(
            (p["adult"] & hosp_off.gt(1440)).sum()
        ),
        "adult_plus24h_stays": int(len(base)),
        "adult_plus24h_patients": int(base["unique_pid"].nunique()),
        "adult_plus24h_hospital_stays": int(base["hstay_id"].nunique()),
        "adult_plus24h_hospitals": int(base["hospital_id"].nunique()),
        "fixed_hf_proxy_plus24h_stays": int(len(candidate)),
        "fixed_hf_proxy_plus24h_patients": int(candidate["unique_pid"].nunique()),
        "fixed_hf_proxy_plus24h_hospital_stays": int(candidate["hstay_id"].nunique()),
    }])
    landmark_summary.to_csv(out / "EPISODE_LANDMARK_SUMMARY.csv", index=False)

    # ------------------------------------------------------------------
    # 4) Unit composition among base and fixed-HF candidate stays.
    # ------------------------------------------------------------------
    def dist(df, col, cohort_name):
        d = (
            df.groupby(col, dropna=False)
            .agg(
                stays=("stay_id", "size"),
                patients=("unique_pid", "nunique"),
                hospital_stays=("hstay_id", "nunique"),
                hospitals=("hospital_id", "nunique"),
            )
            .reset_index()
        )
        d["cohort"] = cohort_name
        d["pct_stays"] = 100.0 * d["stays"] / max(len(df), 1)
        return d

    unit_type = pd.concat([
        dist(base, "unit_type", "adult_plus24h"),
        dist(candidate, "unit_type", "fixed_hf_proxy_plus24h"),
    ], ignore_index=True)
    unit_type.to_csv(out / "UNIT_TYPE_DISTRIBUTION.csv", index=False)

    unit_stay_type = pd.concat([
        dist(base, "unit_stay_type", "adult_plus24h"),
        dist(candidate, "unit_stay_type", "fixed_hf_proxy_plus24h"),
    ], ignore_index=True)
    unit_stay_type.to_csv(
        out / "UNIT_STAY_TYPE_DISTRIBUTION.csv", index=False
    )

    # ------------------------------------------------------------------
    # 5) Within-hospital-admission ordering QC.
    #
    # Official eICU semantics:
    # - all offsets are centered on the unit admission;
    # - within one patientHealthSystemStayID, an EARLIER ICU stay has a
    #   LARGER hospitalAdmitOffset (closer to zero);
    # - unitVisitNumber identifies visit sequence.
    #
    # Audit both orderings; do not choose yet.
    # ------------------------------------------------------------------
    hs_rows = []
    for hstay_id, g in p.groupby("hstay_id", sort=False):
        g = g.copy()
        n = len(g)
        unique_pid_n = g["unique_pid"].nunique(dropna=False)
        hospital_n = g["hospital_id"].nunique(dropna=False)
        year_n = g["hospital_discharge_year"].nunique(dropna=False)

        visit = pd.to_numeric(g["unit_visit_number"], errors="coerce")
        hadm = pd.to_numeric(g["hospital_admit_offset"], errors="coerce")

        visit_unique = visit.notna().all() and not visit.duplicated().any()
        hadm_unique = hadm.notna().all() and not hadm.duplicated().any()

        # A visit-vs-offset comparison is meaningful only when BOTH
        # order variables are complete and unique within a multi-unit hospital
        # admission. Non-comparable admissions must not be counted as
        # empirical disagreements.
        comparison_possible = bool(
            n > 1 and visit_unique and hadm_unique
        )

        if comparison_possible:
            a = (
                g.assign(_visit=visit, _hadm=hadm)
                 .sort_values(["_visit", "stay_id"])["stay_id"].tolist()
            )
            b = (
                g.assign(_visit=visit, _hadm=hadm)
                 .sort_values(
                     ["_hadm", "stay_id"],
                     ascending=[False, True]
                 )["stay_id"].tolist()
            )
            rank_agreement = bool(a == b)
        else:
            rank_agreement = None

        hs_rows.append({
            "hstay_id": hstay_id,
            "unit_stays": n,
            "unique_pid_count": int(unique_pid_n),
            "hospital_count": int(hospital_n),
            "discharge_year_count": int(year_n),
            "unit_visit_number_complete_unique": bool(visit_unique),
            "hospital_admit_offset_complete_unique": bool(hadm_unique),
            "visit_vs_offset_order_comparable": comparison_possible,
            "visit_vs_offset_order_agreement": rank_agreement,
            "candidate_plus24h_stays": int(
                g["stay_id"].isin(candidate["stay_id"]).sum()
            ),
        })

    hs_qc = pd.DataFrame(hs_rows)

    hs_summary = pd.DataFrame([{
        "health_system_stays": int(len(hs_qc)),
        "health_system_stays_with_multiple_unit_stays": int(
            (hs_qc["unit_stays"] > 1).sum()
        ),
        "multiunit_hstays_visit_number_not_complete_unique": int(
            (
                (hs_qc["unit_stays"] > 1)
                & ~hs_qc["unit_visit_number_complete_unique"]
            ).sum()
        ),
        "multiunit_hstays_hospital_admit_offset_not_complete_unique": int(
            (
                (hs_qc["unit_stays"] > 1)
                & ~hs_qc["hospital_admit_offset_complete_unique"]
            ).sum()
        ),
        "multiunit_hstays_visit_vs_offset_not_comparable": int(
            (
                (hs_qc["unit_stays"] > 1)
                & ~hs_qc["visit_vs_offset_order_comparable"]
            ).sum()
        ),
        "multiunit_hstays_visit_vs_offset_order_disagree_among_comparable": int(
            (
                (hs_qc["unit_stays"] > 1)
                & hs_qc["visit_vs_offset_order_comparable"]
                & hs_qc["visit_vs_offset_order_agreement"].eq(False)
            ).sum()
        ),
        "hstays_with_multiple_fixed_hf_plus24h_candidates": int(
            (hs_qc["candidate_plus24h_stays"] > 1).sum()
        ),
        "invalid_hstay_unique_pid_multiplicity": int(
            (hs_qc["unique_pid_count"] != 1).sum()
        ),
        "invalid_hstay_hospital_multiplicity": int(
            (hs_qc["hospital_count"] != 1).sum()
        ),
        "invalid_hstay_discharge_year_multiplicity": int(
            (hs_qc["discharge_year_count"] != 1).sum()
        ),
    }])

    # Candidate-only exact offset ties / visit-number ties.
    cand_hs = candidate.copy()
    cand_hs["hospital_admit_offset_num"] = pd.to_numeric(
        cand_hs["hospital_admit_offset"], errors="coerce"
    )
    cand_hs["unit_visit_number_num"] = pd.to_numeric(
        cand_hs["unit_visit_number"], errors="coerce"
    )

    candidate_hstay_multiplicity = (
        cand_hs.groupby(["unique_pid", "hstay_id"], as_index=False)
        .agg(
            candidate_unit_stays=("stay_id", "size"),
            distinct_hospital_admit_offsets=(
                "hospital_admit_offset_num", "nunique"
            ),
            distinct_unit_visit_numbers=(
                "unit_visit_number_num", "nunique"
            ),
        )
    )
    candidate_hstay_multiplicity.to_csv(
        out / "HF_CANDIDATE_WITHIN_HOSPITAL_MULTIPLICITY.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 6) Across-hospital-admission ordering by uniquePid.
    #
    # eICU documentation explicitly states there is no systematic method
    # to order patientHealthSystemStayID values for the same patient
    # within the same year. We therefore quantify that ambiguity.
    # ------------------------------------------------------------------
    # One row per patient / represented hospital admission among ALL records.
    hadm = (
        p.groupby(["unique_pid", "hstay_id"], as_index=False)
        .agg(
            hospital_id=("hospital_id", "first"),
            hospital_discharge_year=("hospital_discharge_year", "first"),
            all_unit_stays=("stay_id", "size"),
        )
    )

    # Candidate admission set under the already-fixed HF +24h proxy only.
    cand_adm = (
        candidate.groupby(["unique_pid", "hstay_id"], as_index=False)
        .agg(
            hospital_id=("hospital_id", "first"),
            hospital_discharge_year=("hospital_discharge_year", "first"),
            candidate_unit_stays=("stay_id", "size"),
        )
    )

    patient_rows = []
    all_hadm_counts = hadm.groupby("unique_pid")["hstay_id"].nunique()
    all_hospital_counts = hadm.groupby("unique_pid")["hospital_id"].nunique()

    for unique_pid, g in cand_adm.groupby("unique_pid", sort=False):
        years = pd.to_numeric(
            g["hospital_discharge_year"], errors="coerce"
        )
        min_year = years.min()
        earliest = g.loc[years.eq(min_year)] if pd.notna(min_year) else g.iloc[0:0]

        # Same-year multiple qualifying hospital admissions cannot be
        # chronologically resolved from eICU release identifiers alone.
        same_earliest_year_ambiguous = len(earliest) > 1

        patient_rows.append({
            "unique_pid": unique_pid,
            "all_represented_hospital_stays": int(
                all_hadm_counts.get(unique_pid, 0)
            ),
            "all_represented_hospitals": int(
                all_hospital_counts.get(unique_pid, 0)
            ),
            "candidate_hospital_stays": int(g["hstay_id"].nunique()),
            "candidate_discharge_years": int(years.nunique(dropna=True)),
            "earliest_candidate_discharge_year": (
                int(min_year) if pd.notna(min_year) else None
            ),
            "candidate_hospital_stays_in_earliest_year": int(len(earliest)),
            "same_earliest_year_candidate_ambiguity": bool(
                same_earliest_year_ambiguous
            ),
            "candidate_unit_stays_total": int(
                g["candidate_unit_stays"].sum()
            ),
        })

    patient_qc = pd.DataFrame(patient_rows)

    # Do not expose pseudonymous uniquePid in the canonical QC output.
    # Aggregate the patient-level results.
    ambiguity_summary = pd.DataFrame([{
        "hf_plus24h_candidate_stays": int(len(candidate)),
        "hf_plus24h_candidate_patients": int(candidate["unique_pid"].nunique()),
        "candidate_patients_with_multiple_unit_stays": int(
            (
                candidate.groupby("unique_pid")["stay_id"].nunique() > 1
            ).sum()
        ),
        "candidate_patients_with_multiple_hospital_stays": int(
            (patient_qc["candidate_hospital_stays"] > 1).sum()
        ),
        "candidate_patients_with_multiple_candidate_discharge_years": int(
            (patient_qc["candidate_discharge_years"] > 1).sum()
        ),
        "candidate_patients_with_same_earliest_year_hstay_ambiguity": int(
            patient_qc["same_earliest_year_candidate_ambiguity"].sum()
        ),
        "candidate_patients_with_prior_represented_noncandidate_hstay": int(
            (
                patient_qc["all_represented_hospital_stays"]
                > patient_qc["candidate_hospital_stays"]
            ).sum()
        ),
        "candidate_patients_seen_at_multiple_hospitals": int(
            (patient_qc["all_represented_hospitals"] > 1).sum()
        ),
    }])
    ambiguity_summary.to_csv(
        out / "PATIENT_EPISODE_AMBIGUITY_SUMMARY.csv", index=False
    )

    # Distribution, no patient identifiers.
    ambiguity_dist = (
        patient_qc.groupby(
            [
                "candidate_hospital_stays",
                "candidate_discharge_years",
                "candidate_hospital_stays_in_earliest_year",
                "same_earliest_year_candidate_ambiguity",
            ],
            dropna=False,
        )
        .size()
        .rename("patients")
        .reset_index()
        .sort_values("patients", ascending=False)
    )
    ambiguity_dist.to_csv(
        out / "PATIENT_EPISODE_MULTIPLICITY_DISTRIBUTION.csv", index=False
    )

    # ------------------------------------------------------------------
    # 7) Nonbinding prospective ordering simulation for audit only.
    #
    # This is NOT a fixed rule. It asks: among HF+24h candidates, what
    # would happen if:
    # - earliest candidate hospital-discharge year is used across years;
    # - patients with >1 candidate hospital stay in that earliest year are
    #   declared chronology-ambiguous;
    # - within a uniquely selected hospital stay, earliest qualifying ICU
    #   unit stay is the largest hospitalAdmitOffset.
    # ------------------------------------------------------------------
    selected = []
    ambiguous_patients = set(
        patient_qc.loc[
            patient_qc["same_earliest_year_candidate_ambiguity"],
            "unique_pid",
        ].astype(str)
    )

    for unique_pid, g in candidate.groupby("unique_pid", sort=False):
        if str(unique_pid) in ambiguous_patients:
            continue

        years = pd.to_numeric(
            g["hospital_discharge_year"], errors="coerce"
        )
        if years.isna().all():
            continue
        min_year = years.min()
        gy = g.loc[years.eq(min_year)].copy()

        # There should be one candidate hstay in the earliest year if
        # this patient is not ambiguity-flagged.
        hsids = gy["hstay_id"].dropna().unique()
        if len(hsids) != 1:
            continue

        gh = gy[gy["hstay_id"].eq(hsids[0])].copy()
        gh["_visit"] = pd.to_numeric(
            gh["unit_visit_number"], errors="coerce"
        )
        if gh["_visit"].isna().any():
            continue

        min_visit = gh["_visit"].min()
        firsts = gh[gh["_visit"].eq(min_visit)]
        if len(firsts) != 1:
            # Missing/tied primary within-admission order is unresolved.
            continue
        selected.append(int(firsts.iloc[0]["stay_id"]))

    simulation = pd.DataFrame([{
        "analysis_scope": "exploratory_simulation",
        "within_admission_selector": "smallest_unitVisitNumber",
        "candidate_patients": int(candidate["unique_pid"].nunique()),
        "patients_excluded_same_year_candidate_hstay_ambiguity": int(
            patient_qc["same_earliest_year_candidate_ambiguity"].sum()
        ),
        "patients_with_unique_simulated_first_candidate_stay": int(
            len(selected)
        ),
        "patients_not_resolved_by_simulation": int(
            candidate["unique_pid"].nunique() - len(selected)
        ),
    }])
    simulation.to_csv(
        out / "NONBINDING_FIRST_EPISODE_SIMULATION.csv", index=False
    )

    # ------------------------------------------------------------------
    # 8) Safety + manifest.
    # ------------------------------------------------------------------
    for path in out.glob("*.csv"):
        df0 = pd.read_csv(path, nrows=5)
        assert_safe_columns(df0.columns, path.name)

    hashes = {
        path.name: sha256_file(path)
        for path in sorted(out.glob("*.csv"), key=lambda x: x.name)
    }

    manifest = {
        "within_hospital_stay_ordering": json.loads(hs_summary.to_json(orient="records", double_precision=15)),
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "outcome-blind eICU episode-selection and +24 h landmark audit",
        "adult_plus24h_stays": int(len(base)),
        "adult_plus24h_hospitals": int(base["hospital_id"].nunique()),
        "fixed_hf_proxy_plus24h_stays": int(len(candidate)),
        "outcome_or_discharge_status_selected": False,
        "discharge_location_selected": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "episode_contract_fixed_by_script": False,
        "canonical_mimic_science_changed": False,
        "hospital_discharge_year_used_only_for_chronology_qc": True,
        "source_files": {
            "patient": PATIENT.name,
            "pastHistory": PH.name,
        },
        "generated_sha256": hashes,
        "interpretation": (
            "Evidence-only audit. The final episode/index-stay rule is not "
            "chosen by this script. The nonbinding simulation is explicitly "
            "not a cohort lock."
        ),
    }

    (out / "EPISODE_SELECTION_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("EICU EPISODE / LANDMARK AUDIT: PASS")
    print(f"Adult +24 h stays: {len(base):,}")
    print(f"Hospitals: {base['hospital_id'].nunique():,}")
    print(f"Fixed HF-proxy +24 h stays: {len(candidate):,}")
    print(f"Output: {out}")
    print("Outcome/discharge-status values selected: NO")
    print("Discharge-location values selected: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("Episode contract fixed by this script: NO")
    print("=" * 78)

if __name__ == "__main__":
    main()
