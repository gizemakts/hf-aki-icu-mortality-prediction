#!/usr/bin/env python3
"""
Outcome-blind eICU site-availability / structural-missingness audit.

Purpose
-------
Quantify hospital-level capture of the source systems required by the already
fixed eICU harmonization contracts. This script DOES NOT select an exclusion
threshold and DOES NOT attach outcomes.

Denominators
------------
1) adult +24 h stays (site-interface denominator);
2) fixed HF-proxy +24 h stays (candidate-analysis denominator).

Audited source domains
----------------------
- pastHistory
- nurseCharting
- lab
- creatinine
- infusionDrug
- treatment
- intakeOutput
- apacheApsVar
- diagnosis

The output provides:
- hospital-level capture percentages;
- zero-capture / low-capture threshold grids;
- HF-candidate source-missingness patterns;
- counts of hospitals present in the adult denominator but absent from the
  fixed-HF candidate denominator.

No threshold is chosen here. No source absence is converted into a clinical
negative state by this script.
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
    APS = discover(root, "apacheApsVar")
    DX = discover(root, "diagnosis")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    # ---------------------------------------------------------------
    # 1) Adult +24 h denominator.
    # ---------------------------------------------------------------
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

    if adult["stay_id"].duplicated().any():
        raise AssertionError("Duplicate patientUnitStayID in adult denominator")

    con.register("adult_df", adult)
    con.execute("CREATE TEMP TABLE adult AS SELECT * FROM adult_df")

    # ---------------------------------------------------------------
    # 2) Fixed structured HF proxy.
    # ---------------------------------------------------------------
    values_sql = ",".join("'" + v.replace("'", "''") + "'" for v in HF_VALUES)

    hf_ids = con.execute(f"""
        SELECT DISTINCT
            ph.patientUnitStayID AS stay_id
        FROM read_csv_auto(
            '{qpath(PH)}',
            header=true,
            sample_size=200000
        ) ph
        JOIN adult a
          ON ph.patientUnitStayID=a.stay_id
        WHERE ph.pastHistoryOffset<=1440
          AND ph.pastHistoryEnteredOffset<=1440
          AND ph.pastHistoryValue IN ({values_sql})
          AND ph.pastHistoryValueText=ph.pastHistoryValue
          AND ph.pastHistoryPath=
              '{HF_PATH_PREFIX.replace("'", "''")}' || ph.pastHistoryValue
    """).df()

    hf = adult.merge(hf_ids.assign(hf_proxy=1), on="stay_id", how="inner")
    con.register("hf_df", hf)
    con.execute("CREATE TEMP TABLE hf AS SELECT * FROM hf_df")

    # ---------------------------------------------------------------
    # 3) Source-level stay capture sets by +24 h.
    #
    # These indicate whether the source interface contributes ANY admissible
    # row for a stay. They are NOT clinical-positive flags.
    # ---------------------------------------------------------------
    source_queries = {
        "pastHistory": f"""
            SELECT DISTINCT p.patientUnitStayID AS stay_id
            FROM read_csv_auto('{qpath(PH)}', header=true, sample_size=200000) p
            JOIN adult a ON p.patientUnitStayID=a.stay_id
            WHERE p.pastHistoryOffset<=1440
              AND p.pastHistoryEnteredOffset<=1440
        """,
        "nurseCharting": f"""
            SELECT DISTINCT n.patientUnitStayID AS stay_id
            FROM read_csv_auto('{qpath(NURSE)}', header=true, sample_size=200000) n
            JOIN adult a ON n.patientUnitStayID=a.stay_id
            WHERE n.nursingChartOffset>=0
              AND n.nursingChartOffset<=1440
              AND n.nursingChartEntryOffset<=1440
        """,
        "lab": f"""
            SELECT DISTINCT l.patientUnitStayID AS stay_id
            FROM read_csv_auto('{qpath(LAB)}', header=true, sample_size=200000) l
            JOIN adult a ON l.patientUnitStayID=a.stay_id
            WHERE l.labResultOffset>=0
              AND l.labResultOffset<=1440
              AND l.labResultRevisedOffset<=1440
        """,
        "creatinine": f"""
            SELECT DISTINCT l.patientUnitStayID AS stay_id
            FROM read_csv_auto('{qpath(LAB)}', header=true, sample_size=200000) l
            JOIN adult a ON l.patientUnitStayID=a.stay_id
            WHERE lower(l.labName)='creatinine'
              AND l.labTypeID=1
              AND TRY_CAST(l.labResult AS DOUBLE)>0
              AND TRY_CAST(l.labResult AS DOUBLE)<=150
              AND l.labResultOffset>=0
              AND l.labResultOffset<=1440
              AND l.labResultRevisedOffset<=1440
        """,
        "infusionDrug": f"""
            SELECT DISTINCT i.patientUnitStayID AS stay_id
            FROM read_csv_auto('{qpath(INF)}', header=true, sample_size=200000) i
            JOIN adult a ON i.patientUnitStayID=a.stay_id
            WHERE i.infusionOffset>=0
              AND i.infusionOffset<=1440
        """,
        "treatment": f"""
            SELECT DISTINCT t.patientUnitStayID AS stay_id
            FROM read_csv_auto('{qpath(TREAT)}', header=true, sample_size=200000) t
            JOIN adult a ON t.patientUnitStayID=a.stay_id
            WHERE t.treatmentOffset>=0
              AND t.treatmentOffset<=1440
        """,
        "intakeOutput": f"""
            SELECT DISTINCT i.patientUnitStayID AS stay_id
            FROM read_csv_auto('{qpath(IO)}', header=true, sample_size=200000) i
            JOIN adult a ON i.patientUnitStayID=a.stay_id
            WHERE i.intakeOutputOffset>=0
              AND i.intakeOutputOffset<=1440
              AND i.intakeOutputEntryOffset<=1440
        """,
        "apacheApsVar": f"""
            SELECT DISTINCT a2.patientUnitStayID AS stay_id
            FROM read_csv_auto('{qpath(APS)}', header=true, sample_size=200000) a2
            JOIN adult a ON a2.patientUnitStayID=a.stay_id
        """,
        "diagnosis": f"""
            SELECT DISTINCT d.patientUnitStayID AS stay_id
            FROM read_csv_auto('{qpath(DX)}', header=true, sample_size=200000) d
            JOIN adult a ON d.patientUnitStayID=a.stay_id
            WHERE d.diagnosisOffset<=1440
        """,
    }

    source_sets = {}
    for name, sql in source_queries.items():
        ids = con.execute(sql).df()
        assert_safe_columns(ids.columns, name)
        source_sets[name] = set(ids["stay_id"].unique())

    # ---------------------------------------------------------------
    # 4) Hospital-level capture matrix for adult and HF denominators.
    # ---------------------------------------------------------------
    def hospital_matrix(df, denominator_name):
        outdf = (
            df.groupby("hospital_id")
            .size()
            .rename("denominator_stays")
            .reset_index()
        )
        for source, ids in source_sets.items():
            tmp = df[["stay_id", "hospital_id"]].copy()
            tmp["present"] = tmp["stay_id"].isin(ids).astype(int)
            h = (
                tmp.groupby("hospital_id")["present"]
                .sum()
                .rename(f"{source}_stays")
                .reset_index()
            )
            outdf = outdf.merge(h, on="hospital_id", how="left")
            outdf[f"{source}_stays"] = (
                outdf[f"{source}_stays"].fillna(0).astype(int)
            )
            outdf[f"{source}_pct"] = (
                100.0
                * outdf[f"{source}_stays"]
                / outdf["denominator_stays"]
            )
        outdf.insert(1, "denominator", denominator_name)
        return outdf

    adult_site = hospital_matrix(adult, "adult_plus24h")
    hf_site = hospital_matrix(hf, "fixed_hf_plus24h")

    site_matrix = pd.concat([adult_site, hf_site], ignore_index=True)
    site_matrix.to_csv(out / "SITE_SOURCE_CAPTURE_MATRIX.csv", index=False)

    # ---------------------------------------------------------------
    # 5) Threshold grid — descriptive only, no threshold selection.
    # ---------------------------------------------------------------
    thresholds = [0, 1, 5, 10, 25, 50, 75, 90]
    rows = []
    for denom_name, sdf in (
        ("adult_plus24h", adult_site),
        ("fixed_hf_plus24h", hf_site),
    ):
        for source in source_sets:
            pct = sdf[f"{source}_pct"]
            for threshold in thresholds:
                if threshold == 0:
                    n_hosp = int(pct.eq(0).sum())
                    n_stays = int(
                        sdf.loc[pct.eq(0), "denominator_stays"].sum()
                    )
                    relation = "==0"
                else:
                    n_hosp = int(pct.lt(threshold).sum())
                    n_stays = int(
                        sdf.loc[pct.lt(threshold), "denominator_stays"].sum()
                    )
                    relation = f"<{threshold}"
                rows.append({
                    "denominator": denom_name,
                    "source": source,
                    "threshold_pct": threshold,
                    "relation": relation,
                    "hospitals_meeting_low_capture_rule": n_hosp,
                    "denominator_stays_in_those_hospitals": n_stays,
                })

    threshold_grid = pd.DataFrame(rows)
    threshold_grid.to_csv(
        out / "SITE_LOW_CAPTURE_THRESHOLD_GRID.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 6) HF-candidate stay-level structural source pattern.
    # ---------------------------------------------------------------
    patterns = hf[["stay_id", "hospital_id"]].copy()
    for source, ids in source_sets.items():
        patterns[source] = patterns["stay_id"].isin(ids).astype(int)

    domain_cols = list(source_sets.keys())
    patterns["sources_present_count"] = patterns[domain_cols].sum(axis=1)
    patterns["sources_missing_count"] = len(domain_cols) - patterns[
        "sources_present_count"
    ]

    pattern_summary = (
        patterns.groupby(
            domain_cols + ["sources_present_count", "sources_missing_count"],
            dropna=False,
        )
        .size()
        .rename("stays")
        .reset_index()
        .sort_values("stays", ascending=False)
    )
    pattern_summary.to_csv(
        out / "HF_CANDIDATE_SOURCE_PATTERN_SUMMARY.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 7) Zero-capture site summary + denominator reconciliation.
    # ---------------------------------------------------------------
    zero_rows = []
    for denom_name, sdf in (
        ("adult_plus24h", adult_site),
        ("fixed_hf_plus24h", hf_site),
    ):
        for source in source_sets:
            mask = sdf[f"{source}_pct"].eq(0)
            zero_rows.append({
                "denominator": denom_name,
                "source": source,
                "zero_capture_hospitals": int(mask.sum()),
                "stays_in_zero_capture_hospitals": int(
                    sdf.loc[mask, "denominator_stays"].sum()
                ),
            })
    pd.DataFrame(zero_rows).to_csv(
        out / "ZERO_CAPTURE_SITE_SUMMARY.csv",
        index=False,
    )

    adult_hospitals = set(adult["hospital_id"].unique())
    hf_hospitals = set(hf["hospital_id"].unique())

    reconciliation = pd.DataFrame([{
        "adult_plus24h_stays": int(len(adult)),
        "adult_plus24h_hospitals": int(len(adult_hospitals)),
        "fixed_hf_plus24h_stays": int(len(hf)),
        "fixed_hf_plus24h_hospitals": int(len(hf_hospitals)),
        "adult_hospitals_with_no_fixed_hf_candidate": int(
            len(adult_hospitals - hf_hospitals)
        ),
        "hf_hospitals_not_in_adult_denominator": int(
            len(hf_hospitals - adult_hospitals)
        ),
    }])
    reconciliation.to_csv(
        out / "SITE_DENOMINATOR_RECONCILIATION.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 8) Source capture distribution quantiles.
    # ---------------------------------------------------------------
    quantile_rows = []
    for denom_name, sdf in (
        ("adult_plus24h", adult_site),
        ("fixed_hf_plus24h", hf_site),
    ):
        for source in source_sets:
            pct = sdf[f"{source}_pct"]
            quantile_rows.append({
                "denominator": denom_name,
                "source": source,
                "min_pct": float(pct.min()),
                "p10_pct": float(pct.quantile(0.10)),
                "p25_pct": float(pct.quantile(0.25)),
                "median_pct": float(pct.median()),
                "p75_pct": float(pct.quantile(0.75)),
                "p90_pct": float(pct.quantile(0.90)),
                "max_pct": float(pct.max()),
            })

    pd.DataFrame(quantile_rows).to_csv(
        out / "SITE_SOURCE_CAPTURE_QUANTILES.csv",
        index=False,
    )

    # ---------------------------------------------------------------
    # 9) Safety + manifest.
    # ---------------------------------------------------------------
    for path in out.glob("*.csv"):
        df0 = pd.read_csv(path, nrows=5)
        assert_safe_columns(df0.columns, path.name)

    hashes = {
        path.name: sha256_file(path)
        for path in sorted(out.glob("*.csv"), key=lambda x: x.name)
    }

    manifest = {
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "outcome-blind site availability / structural missingness audit",
        "adult_plus24h_stays": int(len(adult)),
        "adult_plus24h_hospitals": int(adult["hospital_id"].nunique()),
        "fixed_hf_plus24h_stays": int(len(hf)),
        "fixed_hf_plus24h_hospitals": int(hf["hospital_id"].nunique()),
        "sources_audited": list(source_sets.keys()),
        "threshold_selected_by_script": False,
        "site_exclusion_applied": False,
        "source_absence_interpreted_as_clinical_negative": False,
        "outcome_or_discharge_status_selected": False,
        "discharge_location_selected": False,
        "model_fit_performed": False,
        "performance_metrics_computed": False,
        "final_episode_selection_applied": False,
        "final_cohort_membership_fixed": False,
        "canonical_mimic_science_changed": False,
        "generated_sha256": hashes,
        "interpretation": (
            "Descriptive structural-capture audit only. Thresholds in the grid "
            "are candidate reporting cut-points, not selected eligibility rules. "
            "Site eligibility remains to be prospectively adjudicated before "
            "outcome access."
        ),
    }

    (out / "SITE_AVAILABILITY_AUDIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("EICU SITE AVAILABILITY / STRUCTURAL MISSINGNESS AUDIT: PASS")
    print(f"Adult +24 h stays: {len(adult):,}")
    print(f"Adult +24 h hospitals: {adult['hospital_id'].nunique():,}")
    print(f"Fixed HF +24 h stays: {len(hf):,}")
    print(f"Fixed HF +24 h hospitals: {hf['hospital_id'].nunique():,}")
    print(f"Output: {out}")
    print("Threshold selected by script: NO")
    print("Site exclusion applied: NO")
    print("Source absence interpreted as clinical negative: NO")
    print("Outcome/discharge-status values selected: NO")
    print("Discharge-location values selected: NO")
    print("Model fitted: NO")
    print("Performance computed: NO")
    print("Final cohort membership fixed: NO")
    print("=" * 78)

if __name__ == "__main__":
    main()
