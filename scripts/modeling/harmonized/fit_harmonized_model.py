#!/usr/bin/env python3
"""
Harmonized model — Harmonized MIMIC Cr/RRT-only transportability/refit + fixed-FCE artifacts.

SCIENTIFIC PURPOSE
------------------
Build the common-denominator MIMIC development model required for formal
MIMIC-IV -> eICU transport:

    rolling creatinine KDIGO + active acute RRT
    NO urine-output KDIGO
    NO strict-outpatient historical creatinine augmentation

The canonical MIMIC analysis is NOT modified. This script creates a separate
harmonized transport/refit artifact set.

Implementation note: the current canonical `aki_target_preoutcome.parquet`
already contains `creatinine_evaluable`, but that column belongs to the
ADJUDICATED PRIMARY creatinine phenotype (rolling creatinine augmented by the
fixed strict-outpatient historical component).  The standalone creatinine
artifact contains the ROLLING-ONLY evaluability required for the Cr/RRT common
denominator.  The two columns are therefore intentionally kept distinct.  The
rolling-evaluable set is required to be a subset of the primary-evaluable set.

The script:
1. locking the Cr/RRT-only outcome-blind MIMIC cohort before MIMIC outcome use;
2. compares it with the canonical MIMIC cohort;
3. reconstructs all 17 nonrenal predictors from the fixed raw MIMIC mappings;
4. proves exact predictor agreement on the canonical 3,457-stay reference;
5. refits the same 19-parameter logistic model using canonical MI30/MICE10/PMM5;
6. reproduces the canonical primary fit as an implementation guard;
7. builds the fixed fixed-chained-equations (FCE) development artifacts;
8. hashes all Harmonized model outputs.

IMPORTANT
---------
- This script NEVER reads eICU data or eICU outcome.
- MIMIC hospital_expire_flag is used only AFTER harmonized cohort membership and
  the pre-outcome predictor matrix are written and hashed.
- Harmonized model is not considered fixed merely because this script returns PASS.
  Outputs must be independently reviewed before the Harmonized model lock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow
import scipy
import statsmodels.api as sm
from scipy.stats import t as student_t


EXPECTED_PRIMARY_N = 3457
EXPECTED_PRIMARY_EVENTS = 707
EXPECTED_TARGET_N = 10083

OUTCOME = "hospital_expire_flag"

PRIMARY_PREDICTORS = [
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
    "kdigo_stage_2_indicator",
    "kdigo_stage_3_indicator",
]

NON_STAGE_PREDICTORS = [
    c
    for c in PRIMARY_PREDICTORS
    if c not in ("kdigo_stage_2_indicator", "kdigo_stage_3_indicator")
]

FCE_TARGETS = [
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

N_IMPUTATIONS = 30
MICE_CYCLES = 10
PMM_DONORS = 5
MODEL_SEED = 2026091701

FCE_CONVERGENCE_TOL = 1e-6
FCE_MAX_CYCLES = 50


PROHIBITED_PREOUTCOME_TOKENS = (
    "death",
    "mortality",
    "expire",
    "outcome",
    "deathtime",
    "survival",
    "brier",
    "auc",
    "auroc",
    "calibration",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_sorted_ids(values) -> str:
    payload = "\n".join(str(int(x)) for x in sorted(set(values))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def assert_unique(df: pd.DataFrame, key: str, label: str) -> None:
    if df[key].isna().any():
        raise AssertionError(f"{label}: missing {key}")
    if df[key].duplicated().any():
        raise AssertionError(f"{label}: duplicate {key}")


def assert_preoutcome(df: pd.DataFrame, label: str) -> None:
    bad = [
        c
        for c in df.columns
        if any(tok in str(c).lower() for tok in PROHIBITED_PREOUTCOME_TOKENS)
    ]
    if bad:
        raise AssertionError(
            f"{label}: prohibited outcome/model-performance columns present: {bad}"
        )


def safe_stage(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def first_positive(df: pd.DataFrame, positive_col: str) -> pd.DataFrame:
    q = df.loc[df[positive_col].fillna(False).astype(bool)].copy()
    q = q.sort_values(["subject_id", "intime", "stay_id"], kind="mergesort")
    return q.drop_duplicates("subject_id", keep="first").copy()


def _draw_linear_beta(X, y, rng):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    inv = np.linalg.pinv(X.T @ X)
    bhat = inv @ X.T @ y
    resid = y - X @ bhat
    rank = np.linalg.matrix_rank(X)
    dfres = max(int(len(y) - rank), 1)
    rss = float(resid @ resid)
    s2hat = max(rss / dfres, 1e-12)
    chi = max(float(rng.chisquare(dfres)), 1e-12)
    s2draw = dfres * s2hat / chi
    cov = s2draw * inv
    cov = (cov + cov.T) / 2
    ev, evec = np.linalg.eigh(cov)
    ev = np.clip(ev, 0, None)
    covpsd = (evec * ev) @ evec.T
    return rng.multivariate_normal(
        bhat, covpsd, check_valid="ignore"
    )


def _pmm_update(cur, original, target, model_cols, missing_mask, rng, donors=5):
    obs = ~missing_mask
    other = [c for c in model_cols if c != target]
    X = cur[other].to_numpy(float)
    X = np.column_stack([np.ones(len(X)), X])
    Xo = X[obs.to_numpy()]
    Xm = X[missing_mask.to_numpy()]
    yo = original.loc[obs, target].to_numpy(float)

    if len(Xm) == 0:
        return cur

    beta = _draw_linear_beta(Xo, yo, rng)
    po = Xo @ beta
    pm = Xm @ beta
    k = min(int(donors), len(yo))
    vals = np.empty(len(pm), float)

    for j, v in enumerate(pm):
        d = np.abs(po - v)
        idx = np.argpartition(d, k - 1)[:k]
        vals[j] = yo[int(rng.choice(idx))]

    cur.loc[missing_mask, target] = vals
    return cur


def mice_pmm_imputations(
    df,
    predictors,
    outcome,
    m,
    cycles,
    donors,
    seed,
    verbose=False,
):
    cols = predictors + [outcome]
    original = df[cols].copy()
    masks = {c: original[c].isna() for c in predictors}
    incomplete = [c for c in predictors if masks[c].any()]

    for c in incomplete:
        vals = set(original[c].dropna().unique().tolist())
        if vals.issubset({0, 1, 0.0, 1.0}) and len(vals) <= 2:
            raise RuntimeError(
                f"Binary missing predictor requires logistic imputation: {c}"
            )

    out = []
    master = np.random.SeedSequence(seed)

    for ii, child in enumerate(master.spawn(m), start=1):
        rng = np.random.default_rng(child)
        cur = original.copy()

        for c in incomplete:
            mask = masks[c]
            donor = original.loc[~mask, c].to_numpy(float)
            if len(donor) == 0:
                raise RuntimeError(f"No observed donors for {c}")
            cur.loc[mask, c] = rng.choice(
                donor, size=int(mask.sum()), replace=True
            )

        for _ in range(cycles):
            for c in list(rng.permutation(incomplete)):
                cur = _pmm_update(
                    cur,
                    original,
                    c,
                    cols,
                    masks[c],
                    rng,
                    donors,
                )

        if cur[predictors].isna().any().any():
            raise AssertionError("Missing values remain after MICE.")

        out.append(cur)

        if verbose and (ii == 1 or ii % 5 == 0):
            print(f"  imputation {ii}/{m}")

    return out


def fit_logistic_mi(imputations, predictors, outcome):
    fits = []
    pred = np.zeros((len(imputations[0]), len(imputations)), float)
    coef_rows = []

    for i, dat in enumerate(imputations, start=1):
        X = sm.add_constant(
            dat[predictors].astype(float),
            has_constant="add",
        )
        y = dat[outcome].astype(float)
        fit = sm.GLM(
            y,
            X,
            family=sm.families.Binomial(),
        ).fit(maxiter=200, tol=1e-10)

        if not getattr(fit, "converged", True):
            raise RuntimeError(f"MI fit {i} did not converge.")

        fits.append(fit)
        pred[:, i - 1] = fit.predict(X)

        for term, value in fit.params.items():
            coef_rows.append(
                {
                    "imputation": i,
                    "term": term,
                    "coefficient": float(value),
                }
            )

    return fits, pred, pd.DataFrame(coef_rows)


def rubin_pool(fits, model_name):
    terms = list(fits[0].params.index)
    Q = np.vstack(
        [f.params.reindex(terms).to_numpy(float) for f in fits]
    )
    U = np.stack(
        [f.cov_params().loc[terms, terms].to_numpy(float) for f in fits]
    )

    m = len(fits)
    qbar = Q.mean(0)
    ubar = U.mean(0)
    B = np.cov(Q, rowvar=False, ddof=1)
    T = ubar + (1 + 1 / m) * B
    se = np.sqrt(np.diag(T))

    udiag = np.diag(ubar)
    bdiag = np.diag(B)
    extra = (1 + 1 / m) * bdiag

    df = np.full(len(terms), np.inf)
    nz = extra > 1e-15
    df[nz] = (m - 1) * (1 + udiag[nz] / extra[nz]) ** 2

    crit = np.where(
        np.isfinite(df),
        student_t.ppf(0.975, df),
        1.959963984540054,
    )
    lo = qbar - crit * se
    hi = qbar + crit * se

    fmi = np.zeros(len(terms), float)
    nz_t = np.diag(T) > 0
    fmi[nz_t] = extra[nz_t] / np.diag(T)[nz_t]

    return pd.DataFrame(
        {
            "model": model_name,
            "term": terms,
            "pooled_coef": qbar,
            "pooled_se": se,
            "df_rubin": df,
            "ci95_low_coef": lo,
            "ci95_high_coef": hi,
            "odds_ratio": np.exp(qbar),
            "ci95_low_or": np.exp(lo),
            "ci95_high_or": np.exp(hi),
            "fraction_missing_information_approx": fmi,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        default=".",
        help="Canonical repository root.",
    )
    ap.add_argument(
        "--mimic-root",
        default=None,
        help="Optional raw MIMIC root override.",
    )
    args = ap.parse_args()

    project = Path(args.project_root).resolve()

    if args.mimic_root:
        mimic_root = Path(args.mimic_root).resolve()
    else:
        if str(project) not in sys.path:
            sys.path.insert(0, str(project))
        from helpers._runtime import RAW_DATA_ROOT
        mimic_root = Path(RAW_DATA_ROOT).resolve()

    out_dir = (
        project
        / "outputs"
        / "08_external_transport"
        / "mimic_cr_rrt_harmonized_refit"
    )
    qc_dir = (
        project
        / "quality_control"
        / "eicu_external_validation"
        / "mimic_cr_rrt_harmonized_refit"
    )

    for d in (out_dir, qc_dir):
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(
                f"{d} already exists and is non-empty. "
                "Harmonized model outputs are immutable after a successful run. "
                "Delete only an incomplete failed-run directory after review."
            )
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("HARMONIZED_MODEL — HARMONIZED MIMIC Cr/RRT-ONLY REFIT + FCE ARTIFACTS")
    print("=" * 84)
    print("Project root :", project)
    print("MIMIC root   :", mimic_root)
    print("eICU outcome :", "NOT ACCESSED")
    print()

    # ------------------------------------------------------------------
    # 1) Canonical source artifacts
    # ------------------------------------------------------------------
    cohort_dir = project / "outputs" / "01_cohort"
    phenotype_dir = project / "outputs" / "02_phenotype"
    predictor_dir = project / "outputs" / "03_predictors"
    model_dir = project / "outputs" / "04_modeling"

    p_target = require_file(
        phenotype_dir / "aki_target_preoutcome.parquet"
    )
    p_cr = require_file(
        phenotype_dir / "creatinine" / "kdigo_creatinine.parquet"
    )
    p_rrt = require_file(
        phenotype_dir / "rrt" / "active_rrt.parquet"
    )
    p_primary_cohort = require_file(
        cohort_dir / "primary_cohort_preoutcome.parquet"
    )
    p_primary_manifest = require_file(
        cohort_dir / "primary_cohort_manifest.json"
    )
    p_primary_matrix = require_file(
        predictor_dir / "primary_predictor_matrix.parquet"
    )
    p_model_manifest = require_file(
        model_dir / "model_fit_manifest.json"
    )
    p_primary_coef = require_file(
        model_dir / "primary_pooled_coefficients.csv"
    )

    primary_manifest = json.loads(
        p_primary_manifest.read_text(encoding="utf-8")
    )
    model_manifest = json.loads(
        p_model_manifest.read_text(encoding="utf-8")
    )

    if (
        primary_manifest.get("status") != "PASS"
        or int(primary_manifest.get("primary_n", -1)) != EXPECTED_PRIMARY_N
    ):
        raise AssertionError(
            "Canonical primary cohort manifest is not PASS at N=3457."
        )

    if (
        model_manifest.get("status") != "PASS"
        or int(model_manifest.get("n", -1)) != EXPECTED_PRIMARY_N
        or int(model_manifest.get("events", -1)) != EXPECTED_PRIMARY_EVENTS
    ):
        raise AssertionError(
            "Canonical model-fit manifest does not match N=3457/events=707."
        )

    if (
        int(model_manifest.get("number_of_imputations", -1))
        != N_IMPUTATIONS
        or int(model_manifest.get("mice_cycles", -1)) != MICE_CYCLES
        or int(model_manifest.get("pmm_donors", -1)) != PMM_DONORS
    ):
        raise AssertionError(
            "Canonical MI settings differ from MI30/MICE10/PMM5."
        )

    input_artifact_hashes = {
        str(p.relative_to(project)).replace("\\", "/"): sha256_file(p)
        for p in (
            p_target,
            p_cr,
            p_rrt,
            p_primary_cohort,
            p_primary_manifest,
            p_primary_matrix,
            p_model_manifest,
            p_primary_coef,
        )
    }

    # ------------------------------------------------------------------
    # 2) Outcome-blind harmonized Cr/RRT-only phenotype and cohort
    # ------------------------------------------------------------------
    target = pd.read_parquet(p_target)
    assert_preoutcome(target, "canonical AKI target")
    assert_unique(target, "stay_id", "canonical AKI target")

    required_target_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "admittime",
        "intime",
        "landmark_time",
        "creatinine_stage",
        "urine_output_stage",
        "active_rrt_any_by_landmark",
        "aki_positive",
        "kdigo_stage",
    ]
    missing = [c for c in required_target_cols if c not in target.columns]
    if missing:
        raise KeyError(
            f"aki_target_preoutcome missing required columns: {missing}"
        )

    if len(target) != EXPECTED_TARGET_N:
        raise AssertionError(
            f"Target N={len(target):,}; expected {EXPECTED_TARGET_N:,}."
        )

    for c in ("admittime", "intime", "landmark_time"):
        target[c] = pd.to_datetime(target[c], errors="raise")

    cr = pd.read_parquet(
        p_cr,
        columns=[
            "stay_id",
            "kdigo_creatinine_stage",
            "creatinine_evaluable",
        ],
    ).rename(
        columns={
            "creatinine_evaluable":
                "_creatinine_evaluable_from_cr_artifact"
        }
    )
    rrt = pd.read_parquet(
        p_rrt,
        columns=[
            "stay_id",
            "active_rrt_any_by_landmark",
        ],
    )

    assert_unique(cr, "stay_id", "rolling creatinine")
    assert_unique(rrt, "stay_id", "active RRT")
    assert_preoutcome(cr, "rolling creatinine")
    assert_preoutcome(rrt, "active RRT")

    # The target already carries RRT; merge a second copy and prove identity.
    x = target.merge(
        cr,
        on="stay_id",
        how="left",
        validate="1:1",
    ).merge(
        rrt.rename(
            columns={
                "active_rrt_any_by_landmark":
                    "_rrt_from_canonical_rrt_artifact"
            }
        ),
        on="stay_id",
        how="left",
        validate="1:1",
    )

    a = pd.to_numeric(
        x["active_rrt_any_by_landmark"],
        errors="coerce",
    ).fillna(0).astype(int)
    b = pd.to_numeric(
        x["_rrt_from_canonical_rrt_artifact"],
        errors="coerce",
    ).fillna(0).astype(int)

    if not a.eq(b).all():
        raise AssertionError(
            "RRT evidence differs between target and canonical RRT artifact."
        )

    x["rolling_creatinine_stage"] = pd.to_numeric(
        x["kdigo_creatinine_stage"],
        errors="coerce",
    )
    # IMPORTANT SCIENTIFIC DISTINCTION
    # ---------------------------------
    # `aki_target_preoutcome.creatinine_evaluable` is the adjudicated PRIMARY
    # creatinine evaluability after strict-outpatient historical augmentation.
    # The standalone creatinine artifact is the ROLLING-ONLY 48h/7d component
    # required for the eICU-common-denominator Cr/RRT phenotype.
    #
    # They are NOT expected to be equal.  By construction, however, rolling
    # evaluability must be a subset of primary evaluability because the primary
    # construct is:
    #     rolling_evaluable OR historical_strict_evaluable.
    artifact_eval = pd.to_numeric(
        x["_creatinine_evaluable_from_cr_artifact"],
        errors="coerce",
    ).fillna(0).astype(int)

    x["rolling_creatinine_evaluable"] = artifact_eval

    if "creatinine_evaluable" in x.columns:
        primary_eval = pd.to_numeric(
            x["creatinine_evaluable"],
            errors="coerce",
        ).fillna(0).astype(int)

        rolling_only_eval = artifact_eval.eq(1) & primary_eval.ne(1)
        if rolling_only_eval.any():
            raise AssertionError(
                "Rolling-only creatinine evaluability is not a subset of "
                "adjudicated primary creatinine evaluability; "
                f"n={int(rolling_only_eval.sum())}"
            )

    x["rrt_positive"] = b.eq(1)

    x["cr_rrt_positive"] = (
        x["rolling_creatinine_stage"].fillna(0).ge(1)
        | x["rrt_positive"]
    )

    cr_stage_for_combine = (
        x["rolling_creatinine_stage"]
        .where(x["rolling_creatinine_stage"].ge(1))
    )
    x["cr_rrt_stage"] = cr_stage_for_combine
    x.loc[x["rrt_positive"], "cr_rrt_stage"] = 3.0

    if x.loc[x["cr_rrt_positive"], "cr_rrt_stage"].isna().any():
        raise AssertionError(
            "Positive Cr/RRT stay without Stage 1–3."
        )

    vals = set(
        x.loc[x["cr_rrt_positive"], "cr_rrt_stage"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    if not vals.issubset({1, 2, 3}):
        raise AssertionError(f"Invalid Cr/RRT stages: {sorted(vals)}")

    # Primary component flags, used only for transportability explanation.
    x["primary_creatinine_stage"] = pd.to_numeric(
        x["creatinine_stage"],
        errors="coerce",
    )

    # The adjudicated primary creatinine stage is max(rolling, historical);
    # therefore a finite rolling stage may never exceed the saved primary
    # creatinine stage.
    both_cr_stage = (
        x["rolling_creatinine_stage"].notna()
        & x["primary_creatinine_stage"].notna()
    )
    bad_stage_nesting = (
        both_cr_stage
        & x["rolling_creatinine_stage"].gt(
            x["primary_creatinine_stage"]
        )
    )
    if bad_stage_nesting.any():
        raise AssertionError(
            "Rolling creatinine stage exceeds adjudicated primary "
            f"creatinine stage; n={int(bad_stage_nesting.sum())}"
        )

    x["primary_uo_stage"] = pd.to_numeric(
        x["urine_output_stage"],
        errors="coerce",
    )
    x["primary_positive_saved"] = (
        x["aki_positive"].fillna(False).astype(bool)
    )

    # The harmonized phenotype must be a strict component-subset of primary.
    impossible = (
        x["cr_rrt_positive"]
        & ~x["primary_positive_saved"]
    )
    if impossible.any():
        raise AssertionError(
            "Cr/RRT-positive stay is not canonical-primary positive; "
            f"n={int(impossible.sum())}"
        )

    harmonized = first_positive(x, "cr_rrt_positive")
    assert_unique(harmonized, "subject_id", "harmonized cohort subject")
    assert_unique(harmonized, "stay_id", "harmonized cohort stay")
    assert_preoutcome(harmonized, "harmonized Cr/RRT cohort")

    canonical = pd.read_parquet(p_primary_cohort)
    assert_preoutcome(canonical, "canonical primary cohort")
    assert_unique(canonical, "subject_id", "canonical primary subject")
    assert_unique(canonical, "stay_id", "canonical primary stay")

    if len(canonical) != EXPECTED_PRIMARY_N:
        raise AssertionError(
            f"Canonical cohort N={len(canonical)} != {EXPECTED_PRIMARY_N}"
        )

    hsub = set(harmonized["subject_id"].astype("int64"))
    csub = set(canonical["subject_id"].astype("int64"))

    if not hsub.issubset(csub):
        raise AssertionError(
            "Harmonized Cr/RRT subject set is not a subset of the canonical "
            "primary subject set."
        )

    # Prepare compact pre-outcome cohort.
    harm_keep = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "admittime",
        "intime",
        "landmark_time",
        "rolling_creatinine_stage",
        "rolling_creatinine_evaluable",
        "rrt_positive",
        "cr_rrt_stage",
    ]
    harm_pre = harmonized[harm_keep].copy()
    harm_pre["kdigo_stage_2_indicator"] = (
        harm_pre["cr_rrt_stage"].eq(2).astype("int8")
    )
    harm_pre["kdigo_stage_3_indicator"] = (
        harm_pre["cr_rrt_stage"].eq(3).astype("int8")
    )
    assert_preoutcome(harm_pre, "harmonized cohort preoutcome")

    p_harm_cohort = (
        out_dir / "harmonized_cr_rrt_cohort_preoutcome.parquet"
    )
    harm_pre.to_parquet(p_harm_cohort, index=False)
    preoutcome_membership_sha = sha256_file(p_harm_cohort)

    # Target-level stage transition.
    primary_stage_all = pd.to_numeric(
        x["kdigo_stage"],
        errors="coerce",
    ).where(x["primary_positive_saved"], 0).fillna(0).astype(int)

    harm_stage_all = (
        pd.to_numeric(x["cr_rrt_stage"], errors="coerce")
        .where(x["cr_rrt_positive"], 0)
        .fillna(0)
        .astype(int)
    )

    target_transition = pd.crosstab(
        primary_stage_all,
        harm_stage_all,
        rownames=["canonical_stage"],
        colnames=["cr_rrt_stage"],
        dropna=False,
    ).reset_index()
    target_transition.to_csv(
        qc_dir / "CR_RRT_TARGET_STAGE_TRANSITION.csv",
        index=False,
    )

    # Subject-level membership comparison.
    c_idx = canonical[
        ["subject_id", "stay_id"]
    ].rename(columns={"stay_id": "canonical_stay_id"})

    h_idx = harmonized[
        ["subject_id", "stay_id"]
    ].rename(columns={"stay_id": "cr_rrt_stay_id"})

    membership = c_idx.merge(
        h_idx,
        on="subject_id",
        how="outer",
        indicator=True,
        validate="1:1",
    )

    if membership["_merge"].eq("right_only").any():
        raise AssertionError(
            "Unexpected harmonized-only subject after subset proof."
        )

    membership["membership_category"] = np.where(
        membership["_merge"].eq("left_only"),
        "canonical_only_dropped",
        np.where(
            membership["canonical_stay_id"].eq(
                membership["cr_rrt_stay_id"]
            ),
            "both_same_stay",
            "both_different_stay",
        ),
    )
    membership.drop(columns=["_merge"], inplace=True)

    # Time-shift QC for subjects selected at a later stay.
    time_map = target.set_index("stay_id")["intime"]
    membership["canonical_intime"] = (
        membership["canonical_stay_id"].map(time_map)
    )
    membership["cr_rrt_intime"] = (
        membership["cr_rrt_stay_id"].map(time_map)
    )
    membership["selection_shift_hours"] = (
        pd.to_datetime(membership["cr_rrt_intime"])
        - pd.to_datetime(membership["canonical_intime"])
    ).dt.total_seconds() / 3600.0

    shifted = membership["membership_category"].eq(
        "both_different_stay"
    )
    if (
        membership.loc[shifted, "selection_shift_hours"]
        .dropna()
        .lt(0)
        .any()
    ):
        raise AssertionError(
            "Cr/RRT first-positive episode occurs before canonical first-positive "
            "episode, which is impossible under a component-subset phenotype."
        )

    membership.to_csv(
        qc_dir / "PRIMARY_VS_CR_RRT_MEMBERSHIP_BY_SUBJECT.csv",
        index=False,
    )

    # Explain canonical-only subjects using their canonical selected stay.
    canonical_selected = (
        canonical[["subject_id", "stay_id"]]
        .merge(
            x[
                [
                    "stay_id",
                    "rolling_creatinine_stage",
                    "primary_creatinine_stage",
                    "primary_uo_stage",
                    "rrt_positive",
                ]
            ],
            on="stay_id",
            how="left",
            validate="1:1",
        )
    )

    canonical_selected["rolling_cr_positive"] = (
        pd.to_numeric(
            canonical_selected["rolling_creatinine_stage"],
            errors="coerce",
        ).fillna(0).ge(1)
    )
    canonical_selected["historical_only_cr_positive"] = (
        pd.to_numeric(
            canonical_selected["primary_creatinine_stage"],
            errors="coerce",
        ).fillna(0).ge(1)
        & ~canonical_selected["rolling_cr_positive"]
    )
    canonical_selected["uo_positive"] = (
        pd.to_numeric(
            canonical_selected["primary_uo_stage"],
            errors="coerce",
        ).fillna(0).ge(1)
    )
    canonical_selected["rrt_positive"] = (
        canonical_selected["rrt_positive"]
        .fillna(False)
        .astype(bool)
    )

    retained_subjects = hsub
    dropped = canonical_selected.loc[
        ~canonical_selected["subject_id"].astype("int64").isin(
            retained_subjects
        )
    ].copy()

    if (
        dropped["rolling_cr_positive"]
        | dropped["rrt_positive"]
    ).any():
        raise AssertionError(
            "Canonical-only dropped subject has Cr/RRT-positive canonical "
            "selected stay."
        )

    def component_pattern(row):
        parts = []
        if bool(row["historical_only_cr_positive"]):
            parts.append("HISTORICAL_CR_ONLY")
        if bool(row["uo_positive"]):
            parts.append("UO")
        if bool(row["rolling_cr_positive"]):
            parts.append("ROLLING_CR")
        if bool(row["rrt_positive"]):
            parts.append("RRT")
        return "+".join(parts) if parts else "UNEXPLAINED"

    dropped["canonical_selected_entry_pattern"] = dropped.apply(
        component_pattern,
        axis=1,
    )
    if dropped["canonical_selected_entry_pattern"].eq(
        "UNEXPLAINED"
    ).any():
        raise AssertionError(
            "Dropped canonical subject has no historical-Cr/UO explanation."
        )

    dropped_summary = (
        dropped["canonical_selected_entry_pattern"]
        .value_counts(dropna=False)
        .rename_axis("entry_pattern")
        .reset_index(name="subjects")
    )
    dropped_summary["pct_of_dropped"] = (
        100.0 * dropped_summary["subjects"] / max(len(dropped), 1)
    )
    dropped_summary.to_csv(
        qc_dir / "CANONICAL_DROPPED_SUBJECT_REASON_SUMMARY.csv",
        index=False,
    )

    stage_counts = (
        harm_pre["cr_rrt_stage"]
        .astype(int)
        .value_counts()
        .sort_index()
    )

    flow = pd.DataFrame(
        [
            {
                "metric": "canonical_target_stays",
                "value": len(target),
            },
            {
                "metric": "canonical_primary_subjects",
                "value": len(canonical),
            },
            {
                "metric": "harmonized_cr_rrt_subjects",
                "value": len(harm_pre),
            },
            {
                "metric": "canonical_only_dropped_subjects",
                "value": int(
                    membership["membership_category"]
                    .eq("canonical_only_dropped")
                    .sum()
                ),
            },
            {
                "metric": "both_same_stay_subjects",
                "value": int(
                    membership["membership_category"]
                    .eq("both_same_stay")
                    .sum()
                ),
            },
            {
                "metric": "both_different_stay_subjects",
                "value": int(
                    membership["membership_category"]
                    .eq("both_different_stay")
                    .sum()
                ),
            },
            {
                "metric": "harmonized_stage1",
                "value": int(stage_counts.get(1, 0)),
            },
            {
                "metric": "harmonized_stage2",
                "value": int(stage_counts.get(2, 0)),
            },
            {
                "metric": "harmonized_stage3",
                "value": int(stage_counts.get(3, 0)),
            },
        ]
    )
    flow.to_csv(
        qc_dir / "CR_RRT_COHORT_FLOW.csv",
        index=False,
    )

    shift_summary = pd.DataFrame(
        [
            {
                "different_stay_subjects": int(shifted.sum()),
                "median_shift_hours": (
                    float(
                        membership.loc[
                            shifted,
                            "selection_shift_hours",
                        ].median()
                    )
                    if shifted.any()
                    else 0.0
                ),
                "min_shift_hours": (
                    float(
                        membership.loc[
                            shifted,
                            "selection_shift_hours",
                        ].min()
                    )
                    if shifted.any()
                    else 0.0
                ),
                "max_shift_hours": (
                    float(
                        membership.loc[
                            shifted,
                            "selection_shift_hours",
                        ].max()
                    )
                    if shifted.any()
                    else 0.0
                ),
            }
        ]
    )
    shift_summary.to_csv(
        qc_dir / "CR_RRT_EPISODE_SHIFT_SUMMARY.csv",
        index=False,
    )

    print("OUTCOME-BLIND Cr/RRT COHORT: PASS")
    print(f"  canonical primary N : {len(canonical):,}")
    print(f"  harmonized Cr/RRT N : {len(harm_pre):,}")
    print(f"  dropped subjects    : {len(dropped):,}")
    print(
        "  different stay     : "
        f"{int(shifted.sum()):,}"
    )

    # ------------------------------------------------------------------
    # 3) Reconstruct exact 17 nonrenal predictors on
    #    canonical-primary + harmonized-selected union.
    # ------------------------------------------------------------------
    primary_ref = pd.read_parquet(p_primary_matrix).reset_index(drop=True)

    expected_cols = [
        "subject_id",
        "stay_id",
    ] + PRIMARY_PREDICTORS

    if primary_ref.columns.tolist() != expected_cols:
        raise AssertionError(
            "Canonical primary predictor matrix schema changed."
        )

    if (
        len(primary_ref) != EXPECTED_PRIMARY_N
        or primary_ref["stay_id"].duplicated().any()
    ):
        raise AssertionError(
            "Canonical primary predictor matrix identity check failed."
        )

    identity_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "admittime",
        "intime",
        "landmark_time",
    ]

    primary_identity = (
        target[identity_cols]
        .loc[
            target["stay_id"].isin(
                primary_ref["stay_id"].astype("int64")
            )
        ]
        .copy()
    )

    if len(primary_identity) != EXPECTED_PRIMARY_N:
        raise AssertionError(
            "Not all canonical primary stays found in target."
        )

    harm_identity = harm_pre[identity_cols].copy()

    union_index = pd.concat(
        [primary_identity, harm_identity],
        ignore_index=True,
    )
    union_index = (
        union_index.sort_values("stay_id")
        .drop_duplicates("stay_id", keep="first")
        .reset_index(drop=True)
    )

    assert_unique(union_index, "stay_id", "predictor reconstruction union")
    assert_preoutcome(union_index, "predictor reconstruction union")

    patients_path = require_file(
        mimic_root / "hosp" / "patients.csv.gz"
    )
    chartevents_path = require_file(
        mimic_root / "icu" / "chartevents.csv.gz"
    )
    labevents_path = require_file(
        mimic_root / "hosp" / "labevents.csv.gz"
    )
    inputevents_path = require_file(
        mimic_root / "icu" / "inputevents.csv.gz"
    )
    admissions_path = require_file(
        mimic_root / "hosp" / "admissions.csv.gz"
    )

    raw_cache = qc_dir / "predictor_reconstruction_cache"
    raw_cache.mkdir(exist_ok=True)

    def register_union(con):
        con.register("target_union_df", union_index)
        con.execute(
            """
            CREATE OR REPLACE TEMP TABLE target_union AS
            SELECT
                CAST(subject_id AS BIGINT) subject_id,
                CAST(hadm_id AS BIGINT) hadm_id,
                CAST(stay_id AS BIGINT) stay_id,
                CAST(admittime AS TIMESTAMP) admittime,
                CAST(intime AS TIMESTAMP) intime,
                CAST(landmark_time AS TIMESTAMP) landmark_time
            FROM target_union_df
            """
        )

    def get_raw_ce_aggregates():
        cache = raw_cache / "chartevents_aggregates.parquet"
        if cache.exists():
            return pd.read_parquet(cache)

        ids = [
            220045,
            220052,
            220181,
            225312,
            220210,
            224690,
            220277,
            223761,
            223762,
        ]

        con = duckdb.connect(database=":memory:")
        con.execute("PRAGMA threads=4")
        register_union(con)

        out = con.execute(
            f"""
            SELECT
                t.stay_id,
                CAST(c.itemid AS BIGINT) itemid,
                MIN(TRY_CAST(c.valuenum AS DOUBLE)) AS vmin_raw,
                MAX(TRY_CAST(c.valuenum AS DOUBLE)) AS vmax_raw
            FROM target_union t
            JOIN read_csv_auto(
                '{qpath(chartevents_path)}',
                header=true,
                sample_size=200000,
                ignore_errors=true
            ) c
              ON CAST(c.stay_id AS BIGINT)=t.stay_id
            WHERE CAST(c.itemid AS BIGINT) IN
                ({','.join(map(str, ids))})
              AND TRY_CAST(c.charttime AS TIMESTAMP) >= t.intime
              AND TRY_CAST(c.charttime AS TIMESTAMP) <= t.landmark_time
              AND TRY_CAST(c.storetime AS TIMESTAMP) <= t.landmark_time
              AND TRY_CAST(c.valuenum AS DOUBLE) IS NOT NULL
              AND (
                    (
                        CAST(c.itemid AS BIGINT)=220045
                        AND TRY_CAST(c.valuenum AS DOUBLE)>0
                        AND TRY_CAST(c.valuenum AS DOUBLE)<300
                    )
                 OR (
                        CAST(c.itemid AS BIGINT) IN
                            (220052,220181,225312)
                        AND TRY_CAST(c.valuenum AS DOUBLE)>0
                        AND TRY_CAST(c.valuenum AS DOUBLE)<300
                    )
                 OR (
                        CAST(c.itemid AS BIGINT) IN (220210,224690)
                        AND TRY_CAST(c.valuenum AS DOUBLE)>0
                        AND TRY_CAST(c.valuenum AS DOUBLE)<70
                    )
                 OR (
                        CAST(c.itemid AS BIGINT)=220277
                        AND TRY_CAST(c.valuenum AS DOUBLE)>0
                        AND TRY_CAST(c.valuenum AS DOUBLE)<=100
                    )
                 OR (
                        CAST(c.itemid AS BIGINT)=223761
                        AND TRY_CAST(c.valuenum AS DOUBLE)>70
                        AND TRY_CAST(c.valuenum AS DOUBLE)<120
                    )
                 OR (
                        CAST(c.itemid AS BIGINT)=223762
                        AND TRY_CAST(c.valuenum AS DOUBLE)>10
                        AND TRY_CAST(c.valuenum AS DOUBLE)<50
                    )
              )
            GROUP BY t.stay_id, CAST(c.itemid AS BIGINT)
            """
        ).df()
        con.close()
        out.to_parquet(cache, index=False)
        return out

    def get_raw_lab_aggregates():
        cache = raw_cache / "labevents_aggregates.parquet"
        if cache.exists():
            return pd.read_parquet(cache)

        ids = [
            51222,
            51301,
            51265,
            51006,
            50882,
            50983,
            50971,
            50931,
        ]

        con = duckdb.connect(database=":memory:")
        con.execute("PRAGMA threads=4")
        register_union(con)

        out = con.execute(
            f"""
            SELECT
                t.stay_id,
                CAST(l.itemid AS BIGINT) itemid,
                MIN(TRY_CAST(l.valuenum AS DOUBLE)) AS vmin_raw,
                MAX(TRY_CAST(l.valuenum AS DOUBLE)) AS vmax_raw
            FROM target_union t
            JOIN read_csv_auto(
                '{qpath(labevents_path)}',
                header=true,
                sample_size=200000,
                ignore_errors=true
            ) l
              ON CAST(l.hadm_id AS BIGINT)=t.hadm_id
            WHERE CAST(l.itemid AS BIGINT) IN
                ({','.join(map(str, ids))})
              AND TRY_CAST(l.charttime AS TIMESTAMP) >= t.intime
              AND TRY_CAST(l.charttime AS TIMESTAMP) <= t.landmark_time
              AND TRY_CAST(l.storetime AS TIMESTAMP) <= t.landmark_time
              AND TRY_CAST(l.valuenum AS DOUBLE) IS NOT NULL
              AND (
                    (
                        CAST(l.itemid AS BIGINT)=51222
                        AND TRY_CAST(l.valuenum AS DOUBLE)>0
                        AND TRY_CAST(l.valuenum AS DOUBLE)<=50
                    )
                 OR (
                        CAST(l.itemid AS BIGINT)=51301
                        AND TRY_CAST(l.valuenum AS DOUBLE)>0
                        AND TRY_CAST(l.valuenum AS DOUBLE)<=1000
                    )
                 OR (
                        CAST(l.itemid AS BIGINT)=51265
                        AND TRY_CAST(l.valuenum AS DOUBLE)>0
                        AND TRY_CAST(l.valuenum AS DOUBLE)<=10000
                    )
                 OR (
                        CAST(l.itemid AS BIGINT)=51006
                        AND TRY_CAST(l.valuenum AS DOUBLE)>0
                        AND TRY_CAST(l.valuenum AS DOUBLE)<=300
                    )
                 OR (
                        CAST(l.itemid AS BIGINT)=50882
                        AND TRY_CAST(l.valuenum AS DOUBLE)>0
                        AND TRY_CAST(l.valuenum AS DOUBLE)<=10000
                    )
                 OR (
                        CAST(l.itemid AS BIGINT)=50983
                        AND TRY_CAST(l.valuenum AS DOUBLE)>0
                        AND TRY_CAST(l.valuenum AS DOUBLE)<=200
                    )
                 OR (
                        CAST(l.itemid AS BIGINT)=50971
                        AND TRY_CAST(l.valuenum AS DOUBLE)>0
                        AND TRY_CAST(l.valuenum AS DOUBLE)<=30
                    )
                 OR (
                        CAST(l.itemid AS BIGINT)=50931
                        AND TRY_CAST(l.valuenum AS DOUBLE)>0
                        AND TRY_CAST(l.valuenum AS DOUBLE)<=10000
                    )
              )
            GROUP BY t.stay_id, CAST(l.itemid AS BIGINT)
            """
        ).df()
        con.close()
        out.to_parquet(cache, index=False)
        return out

    def agg_itemset(agg, ids, which):
        sub = agg[agg["itemid"].isin(ids)]
        if which == "max":
            v = sub.groupby("stay_id")["vmax_raw"].max()
        elif which == "min":
            v = sub.groupby("stay_id")["vmin_raw"].min()
        else:
            raise ValueError(which)
        return union_index["stay_id"].map(v).astype(float)

    # Demographics
    patients = pd.read_csv(
        patients_path,
        usecols=[
            "subject_id",
            "gender",
            "anchor_age",
            "anchor_year",
        ],
    )
    demo = union_index[
        ["stay_id", "subject_id", "admittime"]
    ].merge(
        patients,
        on="subject_id",
        how="left",
        validate="m:1",
    )

    admission_year = pd.to_datetime(
        demo["admittime"]
    ).dt.year
    age = (
        pd.to_numeric(demo["anchor_age"], errors="coerce")
        + admission_year
        - pd.to_numeric(demo["anchor_year"], errors="coerce")
    )
    age_per10 = age.clip(upper=91) / 10.0

    g = demo["gender"].astype("string").str.strip().str.upper()
    sex_female = pd.Series(
        np.where(
            g.eq("F"),
            1.0,
            np.where(g.notna(), 0.0, np.nan),
        ),
        index=demo.index,
        dtype=float,
    )

    backbone = union_index[["stay_id"]].copy()
    backbone["age_per10y"] = age_per10.to_numpy()
    backbone["sex_female"] = sex_female.to_numpy()

    # Vitals
    ce = get_raw_ce_aggregates()
    sid = union_index["stay_id"]

    backbone["heart_rate_max_per10"] = (
        agg_itemset(ce, [220045], "max") / 10.0
    )
    backbone["map_min_per10"] = (
        agg_itemset(
            ce,
            [220052, 220181, 225312],
            "min",
        )
        / 10.0
    )
    backbone["resp_rate_max_per5"] = (
        agg_itemset(
            ce,
            [220210, 224690],
            "max",
        )
        / 5.0
    )

    sp = agg_itemset(ce, [220277], "min")
    backbone["spo2_deficit_per5"] = (100.0 - sp) / 5.0

    fmin = (
        agg_itemset(ce, [223761], "min") - 32.0
    ) / 1.8
    fmax = (
        agg_itemset(ce, [223761], "max") - 32.0
    ) / 1.8
    cmin = agg_itemset(ce, [223762], "min")
    cmax = agg_itemset(ce, [223762], "max")

    tmin = pd.concat(
        [fmin, cmin],
        axis=1,
    ).min(axis=1, skipna=True)
    tmax = pd.concat(
        [fmax, cmax],
        axis=1,
    ).max(axis=1, skipna=True)

    no_temp = (
        fmin.isna()
        & fmax.isna()
        & cmin.isna()
        & cmax.isna()
    )
    tmin.loc[no_temp] = np.nan
    tmax.loc[no_temp] = np.nan

    backbone[
        "temperature_max_abs_deviation_from_37c"
    ] = pd.concat(
        [
            (tmin - 37.0).abs(),
            (tmax - 37.0).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)

    # Labs
    labs = get_raw_lab_aggregates()

    raw_hgb = agg_itemset(labs, [51222], "min")
    raw_wbc = agg_itemset(labs, [51301], "max")
    raw_plt = agg_itemset(labs, [51265], "min")
    raw_bun = agg_itemset(labs, [51006], "max")
    raw_bic = agg_itemset(labs, [50882], "min")
    raw_na_min = agg_itemset(labs, [50983], "min")
    raw_na_max = agg_itemset(labs, [50983], "max")
    raw_k = agg_itemset(labs, [50971], "max")
    raw_glu = agg_itemset(labs, [50931], "max")

    backbone["hemoglobin_min_gdl"] = raw_hgb
    with np.errstate(divide="ignore", invalid="ignore"):
        backbone["wbc_max_log2"] = np.log2(raw_wbc)
        backbone["platelet_min_log2"] = np.log2(raw_plt)
        backbone["bun_max_log2"] = np.log2(raw_bun)
        backbone["glucose_max_log2"] = np.log2(raw_glu)
    backbone["bicarbonate_min_per5"] = raw_bic / 5.0
    backbone[
        "sodium_max_abs_deviation_from140_per5"
    ] = pd.concat(
        [
            (raw_na_min - 140.0).abs(),
            (raw_na_max - 140.0).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False) / 5.0
    backbone["potassium_max_mmol_l"] = raw_k

    # GCS — exact source that independently reproduced the canonical primary
    # matrix in the fixed phenotype-refit machinery:
    # GCS logic adapted from MIT-LCP mimic-code, MIT License.
    # Copyright (c) 2019 MIT Laboratory for Computational Physiology.
    # https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/measurement/gcs.sql
    # See THIRD_PARTY_NOTICES.md for the original permission notice.
    # MIT-LCP prior-row-within-6h algorithm, but chart rows themselves are
    # restricted to [intime, landmark].
    gcs_cache = raw_cache / "gcs_exact_window_only.parquet"
    if gcs_cache.exists():
        gcs_raw = pd.read_parquet(gcs_cache)
    else:
        con = duckdb.connect(database=":memory:")
        con.execute("PRAGMA threads=4")
        register_union(con)
        gcs_raw = con.execute(
            f"""
            WITH base0 AS (
                SELECT
                    t.subject_id,
                    t.stay_id,
                    t.intime,
                    t.landmark_time,
                    TRY_CAST(c.charttime AS TIMESTAMP) AS charttime,

                    MAX(
                        CASE
                            WHEN CAST(c.itemid AS BIGINT)=223901
                            THEN TRY_CAST(c.valuenum AS DOUBLE)
                        END
                    ) AS gcsmotor,

                    MAX(
                        CASE
                            WHEN CAST(c.itemid AS BIGINT)=223900
                                 AND CAST(c.value AS VARCHAR)='No Response-ETT'
                            THEN 0
                            WHEN CAST(c.itemid AS BIGINT)=223900
                            THEN TRY_CAST(c.valuenum AS DOUBLE)
                        END
                    ) AS gcsverbal,

                    MAX(
                        CASE
                            WHEN CAST(c.itemid AS BIGINT)=220739
                            THEN TRY_CAST(c.valuenum AS DOUBLE)
                        END
                    ) AS gcseyes

                FROM target_union t
                JOIN read_csv_auto(
                    '{qpath(chartevents_path)}',
                    header=true,
                    sample_size=200000,
                    ignore_errors=true
                ) c
                  ON CAST(c.stay_id AS BIGINT)=t.stay_id

                WHERE CAST(c.itemid AS BIGINT) IN
                    (223900,223901,220739)
                  AND TRY_CAST(c.charttime AS TIMESTAMP) >= t.intime
                  AND TRY_CAST(c.charttime AS TIMESTAMP) <= t.landmark_time
                  AND TRY_CAST(c.storetime AS TIMESTAMP) <= t.landmark_time

                GROUP BY
                    t.subject_id,
                    t.stay_id,
                    t.intime,
                    t.landmark_time,
                    TRY_CAST(c.charttime AS TIMESTAMP)
            ),

            base AS (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY stay_id
                        ORDER BY charttime ASC
                    ) AS rn
                FROM base0
            ),

            gcs_rows AS (
                SELECT
                    b.subject_id,
                    b.stay_id,
                    b.intime,
                    b.landmark_time,
                    b.charttime,

                    CASE
                        WHEN b.gcsverbal=0 THEN 15
                        WHEN b.gcsverbal IS NULL
                             AND b2.gcsverbal=0 THEN 15
                        WHEN b2.gcsverbal=0 THEN
                            COALESCE(b.gcsmotor,6)
                            + COALESCE(b.gcsverbal,5)
                            + COALESCE(b.gcseyes,4)
                        ELSE
                            COALESCE(
                                b.gcsmotor,
                                COALESCE(b2.gcsmotor,6)
                            )
                            + COALESCE(
                                b.gcsverbal,
                                COALESCE(b2.gcsverbal,5)
                            )
                            + COALESCE(
                                b.gcseyes,
                                COALESCE(b2.gcseyes,4)
                            )
                    END AS gcs

                FROM base b
                LEFT JOIN base b2
                  ON b.stay_id=b2.stay_id
                 AND b.rn=b2.rn+1
                 AND b2.charttime >
                     b.charttime - INTERVAL '6' HOUR
            )

            SELECT
                t.stay_id,
                MIN(g.gcs) AS gcs_min_0_24h
            FROM target_union t
            LEFT JOIN gcs_rows g
              ON t.stay_id=g.stay_id
             AND g.charttime >= t.intime
             AND g.charttime <= t.landmark_time
            GROUP BY t.stay_id
            """
        ).df()
        con.close()
        gcs_raw.to_parquet(gcs_cache, index=False)

    gcs_map = gcs_raw.set_index("stay_id")["gcs_min_0_24h"]
    backbone["gcs_deficit_from15"] = (
        15.0
        - backbone["stay_id"].map(
            pd.to_numeric(gcs_map, errors="coerce")
        )
    )

    # Vasoactive support
    vaso_cache = raw_cache / "vasoactive_exact.parquet"
    if vaso_cache.exists():
        vaso = pd.read_parquet(vaso_cache)
    else:
        allids = [
            221906,
            221289,
            221749,
            222315,
            221662,
            221653,
            221986,
        ]
        con = duckdb.connect(database=":memory:")
        con.execute("PRAGMA threads=4")
        register_union(con)
        vaso = con.execute(
            f"""
            SELECT
                t.stay_id,
                MAX(
                    CASE
                        WHEN CAST(i.itemid AS BIGINT) IN
                            ({','.join(map(str, allids))})
                        THEN 1 ELSE 0
                    END
                ) AS vasoactive_any_by_landmark
            FROM target_union t
            LEFT JOIN read_csv_auto(
                '{qpath(inputevents_path)}',
                header=true,
                sample_size=200000,
                ignore_errors=true
            ) i
              ON CAST(i.stay_id AS BIGINT)=t.stay_id
             AND CAST(i.itemid AS BIGINT) IN
                ({','.join(map(str, allids))})
             AND TRY_CAST(i.starttime AS TIMESTAMP)
                 <= t.landmark_time
             AND TRY_CAST(i.endtime AS TIMESTAMP)
                 >= t.intime
             AND TRY_CAST(i.storetime AS TIMESTAMP)
                 <= t.landmark_time
             AND (
                    i.statusdescription IS NULL
                    OR LOWER(
                        CAST(i.statusdescription AS VARCHAR)
                    ) <> 'rewritten'
                 )
            GROUP BY t.stay_id
            """
        ).df()
        con.close()
        vaso["vasoactive_any_by_landmark"] = (
            pd.to_numeric(
                vaso["vasoactive_any_by_landmark"],
                errors="coerce",
            )
            .fillna(0)
            .astype(float)
        )
        vaso.to_parquet(vaso_cache, index=False)

    vaso_map = vaso.set_index(
        "stay_id"
    )["vasoactive_any_by_landmark"]
    backbone["vasoactive_any_by_landmark"] = (
        backbone["stay_id"].map(vaso_map).fillna(0).astype(float)
    )

    if backbone["stay_id"].duplicated().any():
        raise AssertionError("Predictor backbone duplicate stay_id.")

    for c in NON_STAGE_PREDICTORS:
        backbone[c] = pd.to_numeric(backbone[c], errors="coerce")
        arr = backbone[c].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(arr).any():
            raise AssertionError(f"Infinite reconstructed values in {c}")

    # Exact canonical primary-reference proof for all 17 nonstage predictors.
    reconstructed_primary = (
        primary_ref[["subject_id", "stay_id"]]
        .merge(
            backbone,
            on="stay_id",
            how="left",
            validate="1:1",
        )
    )

    predictor_repro_rows = []
    for c in NON_STAGE_PREDICTORS:
        ref = primary_ref[c].to_numpy(float)
        got = reconstructed_primary[c].to_numpy(float)
        ok = np.isclose(
            ref,
            got,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
        finite = np.isfinite(ref) & np.isfinite(got)
        maxdiff = (
            float(np.max(np.abs(ref[finite] - got[finite])))
            if finite.any()
            else np.nan
        )
        predictor_repro_rows.append(
            {
                "predictor": c,
                "mismatch_n": int((~ok).sum()),
                "max_abs_diff": maxdiff,
            }
        )
        if not ok.all():
            raise AssertionError(
                f"Canonical predictor reconstruction mismatch: {c}; "
                f"n={int((~ok).sum())}; maxdiff={maxdiff}"
            )


    print("17 NONRENAL PREDICTOR RAW RECONSTRUCTION: EXACT PRIMARY MATCH PASS")

    # Harmonized 19-predictor matrix.
    harm_matrix = harm_pre[
        [
            "subject_id",
            "stay_id",
            "kdigo_stage_2_indicator",
            "kdigo_stage_3_indicator",
        ]
    ].merge(
        backbone,
        on="stay_id",
        how="left",
        validate="1:1",
    )

    harm_matrix = harm_matrix[
        ["subject_id", "stay_id"] + PRIMARY_PREDICTORS
    ].copy()

    assert_preoutcome(
        harm_matrix,
        "harmonized raw predictor matrix",
    )
    assert_unique(
        harm_matrix,
        "stay_id",
        "harmonized raw predictor matrix",
    )

    for c in PRIMARY_PREDICTORS:
        harm_matrix[c] = pd.to_numeric(
            harm_matrix[c],
            errors="coerce",
        )
        arr = harm_matrix[c].to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        if np.isinf(arr).any():
            raise AssertionError(
                f"Harmonized matrix has infinity in {c}"
            )

    for c in (
        "sex_female",
        "vasoactive_any_by_landmark",
        "kdigo_stage_2_indicator",
        "kdigo_stage_3_indicator",
    ):
        if harm_matrix[c].isna().any():
            raise AssertionError(
                f"Binary predictor {c} contains missing values."
            )
        if not set(
            harm_matrix[c].unique().tolist()
        ).issubset({0, 1, 0.0, 1.0}):
            raise AssertionError(
                f"Binary predictor {c} has nonbinary values."
            )

    p_harm_matrix = (
        out_dir
        / "harmonized_cr_rrt_predictor_matrix_preoutcome.parquet"
    )
    harm_matrix.to_parquet(p_harm_matrix, index=False)
    preoutcome_matrix_sha = sha256_file(p_harm_matrix)

    missing_rows = []
    for c in PRIMARY_PREDICTORS:
        nmiss = int(harm_matrix[c].isna().sum())
        missing_rows.append(
            {
                "predictor": c,
                "missing_n": nmiss,
                "missing_pct": 100.0 * nmiss / len(harm_matrix),
            }
        )

    missingness = pd.DataFrame(missing_rows)
    missingness.to_csv(
        qc_dir / "HARMONIZED_MIMIC_PREDICTOR_MISSINGNESS.csv",
        index=False,
    )

    print("HARMONIZED PRE-OUTCOME PREDICTOR MATRIX: PASS")
    print(
        "  membership SHA256:",
        preoutcome_membership_sha,
    )
    print(
        "  predictor matrix SHA256:",
        preoutcome_matrix_sha,
    )

    # ------------------------------------------------------------------
    # 4) AFTER membership + predictor hash: attach MIMIC outcome.
    # ------------------------------------------------------------------
    admissions = pd.read_csv(
        admissions_path,
        usecols=[
            "subject_id",
            "hadm_id",
            "hospital_expire_flag",
        ],
    )
    assert_unique(admissions, "hadm_id", "MIMIC admissions")

    admissions[OUTCOME] = pd.to_numeric(
        admissions[OUTCOME],
        errors="raise",
    ).astype(int)

    if not set(admissions[OUTCOME].unique()).issubset({0, 1}):
        raise AssertionError("MIMIC outcome is not binary.")

    outcome_map = admissions.set_index("hadm_id")[OUTCOME]

    harm_dev = harm_matrix.copy()
    harm_hadm_map = harm_pre.set_index("stay_id")["hadm_id"]
    harm_dev["hadm_id"] = harm_dev["stay_id"].map(harm_hadm_map)
    harm_dev[OUTCOME] = harm_dev["hadm_id"].map(outcome_map)

    if harm_dev[OUTCOME].isna().any():
        raise AssertionError(
            "Missing MIMIC outcome after harmonized membership lock."
        )

    harm_dev[OUTCOME] = harm_dev[OUTCOME].astype(int)

    # Membership must remain byte-for-byte identical in ID order.
    if not harm_dev["stay_id"].eq(harm_matrix["stay_id"]).all():
        raise AssertionError("Outcome attachment changed row membership/order.")

    p_dev = (
        out_dir
        / "harmonized_cr_rrt_development_matrix_with_outcome.parquet"
    )
    harm_dev[
        ["subject_id", "stay_id", OUTCOME] + PRIMARY_PREDICTORS
    ].to_parquet(p_dev, index=False)

    harmonized_events = int(harm_dev[OUTCOME].sum())

    # Canonical primary outcome attachment, preserving exact matrix row order.
    target_hadm_map = target.set_index("stay_id")["hadm_id"]

    primary_for_repro = primary_ref.copy()
    primary_for_repro["hadm_id"] = (
        primary_for_repro["stay_id"].map(target_hadm_map)
    )
    primary_for_repro[OUTCOME] = (
        primary_for_repro["hadm_id"].map(outcome_map)
    )

    if primary_for_repro[OUTCOME].isna().any():
        raise AssertionError(
            "Missing canonical primary outcome after source attachment."
        )

    primary_for_repro[OUTCOME] = (
        primary_for_repro[OUTCOME].astype(int)
    )

    if int(primary_for_repro[OUTCOME].sum()) != EXPECTED_PRIMARY_EVENTS:
        raise AssertionError(
            "Canonical primary event count is not 707."
        )

    # ------------------------------------------------------------------
    # 5) Reproduce canonical primary MI model implementation.
    # ------------------------------------------------------------------
    print("Reproducing canonical primary MI30 model...")

    repro_imps = mice_pmm_imputations(
        primary_for_repro,
        PRIMARY_PREDICTORS,
        OUTCOME,
        N_IMPUTATIONS,
        MICE_CYCLES,
        PMM_DONORS,
        MODEL_SEED,
        verbose=True,
    )
    repro_fits, _, _ = fit_logistic_mi(
        repro_imps,
        PRIMARY_PREDICTORS,
        OUTCOME,
    )
    repro_pool = rubin_pool(
        repro_fits,
        "PRIMARY_REPRODUCTION",
    )

    reference_pool = pd.read_csv(p_primary_coef)
    comp = repro_pool[
        ["term", "pooled_coef"]
    ].merge(
        reference_pool[
            ["term", "pooled_coef"]
        ],
        on="term",
        suffixes=("_repro", "_reference"),
        validate="1:1",
    )
    comp["abs_diff"] = (
        comp["pooled_coef_repro"]
        - comp["pooled_coef_reference"]
    ).abs()


    maxdiff = float(comp["abs_diff"].max())
    if maxdiff > 1e-8:
        raise AssertionError(
            "Canonical primary model reproduction failed; "
            f"max pooled coefficient abs diff={maxdiff}"
        )

    del repro_imps, repro_fits

    print(
        "CANONICAL MODEL IMPLEMENTATION REPRODUCTION: PASS "
        f"(max abs diff={maxdiff:.3e})"
    )

    # ------------------------------------------------------------------
    # 6) Formal harmonized MIMIC Cr/RRT model refit.
    # ------------------------------------------------------------------
    print("Fitting harmonized Cr/RRT MI30 model...")

    harm_fit_df = harm_dev[
        ["subject_id", "stay_id", OUTCOME] + PRIMARY_PREDICTORS
    ].copy()

    harm_imps = mice_pmm_imputations(
        harm_fit_df,
        PRIMARY_PREDICTORS,
        OUTCOME,
        N_IMPUTATIONS,
        MICE_CYCLES,
        PMM_DONORS,
        MODEL_SEED,
        verbose=True,
    )

    harm_fits, harm_pred_by_imp, harm_coef_by_imp = fit_logistic_mi(
        harm_imps,
        PRIMARY_PREDICTORS,
        OUTCOME,
    )

    harm_pool = rubin_pool(
        harm_fits,
        "HARMONIZED_MIMIC_CR_RRT",
    )

    harm_coef_by_imp.to_csv(
        out_dir
        / "harmonized_cr_rrt_coefficients_by_imputation.csv",
        index=False,
    )
    harm_pool.to_csv(
        out_dir
        / "harmonized_cr_rrt_pooled_coefficients.csv",
        index=False,
    )

    locked_vector = harm_pool[
        ["term", "pooled_coef"]
    ].rename(
        columns={"pooled_coef": "locked_coefficient"}
    )
    locked_vector.to_csv(
        out_dir
        / "harmonized_cr_rrt_locked_coefficient_vector.csv",
        index=False,
    )

    apparent_prob = harm_pred_by_imp.mean(axis=1)
    apparent = harm_fit_df[
        ["subject_id", "stay_id", OUTCOME]
    ].copy()
    apparent["apparent_probability_mi_mean"] = apparent_prob
    apparent.to_csv(
        out_dir
        / "harmonized_cr_rrt_apparent_predictions.csv",
        index=False,
    )

    canonical_coef = reference_pool[
        ["term", "pooled_coef"]
    ].rename(
        columns={"pooled_coef": "canonical_pooled_coef"}
    )
    coef_compare = locked_vector.merge(
        canonical_coef,
        on="term",
        how="left",
        validate="1:1",
    )
    coef_compare["harmonized_minus_canonical"] = (
        coef_compare["locked_coefficient"]
        - coef_compare["canonical_pooled_coef"]
    )
    coef_compare["abs_difference"] = (
        coef_compare["harmonized_minus_canonical"].abs()
    )
    coef_compare.to_csv(
        qc_dir / "HARMONIZED_VS_CANONICAL_COEFFICIENTS.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 7) Fixed FCE development artifacts.
    # ------------------------------------------------------------------
    print("Building fixed-FCE development artifacts...")

    fce_coef_rows = []
    fce_init_rows = []
    fce_spec_rows = []

    for target_name in FCE_TARGETS:
        covariates = [
            c for c in PRIMARY_PREDICTORS
            if c != target_name
        ] + [OUTCOME]

        fce_spec_rows.append(
            {
                "target_predictor": target_name,
                "conditional_model_family":
                    "ordinary_linear_regression",
                "training_covariates":
                    " | ".join(covariates),
                "external_outcome_surrogate":
                    "current_locked_model_predicted_probability",
                "initialization":
                    "pooled_MIMIC_completed_development_marginal_mean",
                "external_update_order":
                    "canonical_model_column_order",
                "external_convergence_tolerance":
                    FCE_CONVERGENCE_TOL,
                "external_max_full_cycles":
                    FCE_MAX_CYCLES,
                "external_stochastic_draws": False,
                "external_pmm_donor_sampling": False,
            }
        )

        for imp_no, dat in enumerate(harm_imps, start=1):
            X = dat[covariates].astype(float)
            X = sm.add_constant(X, has_constant="add")
            y = dat[target_name].astype(float)

            arr = X.to_numpy(float)
            rank = int(np.linalg.matrix_rank(arr))
            if rank != arr.shape[1]:
                raise RuntimeError(
                    f"FCE conditional model rank deficient: "
                    f"target={target_name}, imputation={imp_no}, "
                    f"rank={rank}, p={arr.shape[1]}"
                )

            beta, *_ = np.linalg.lstsq(
                arr,
                y.to_numpy(float),
                rcond=None,
            )

            for term, value in zip(X.columns, beta):
                fce_coef_rows.append(
                    {
                        "target_predictor": target_name,
                        "imputation": imp_no,
                        "term": term,
                        "coefficient": float(value),
                    }
                )

            fce_init_rows.append(
                {
                    "target_predictor": target_name,
                    "imputation": imp_no,
                    "completed_development_mean":
                        float(y.mean()),
                }
            )

    fce_by_imp = pd.DataFrame(fce_coef_rows)
    fce_init_by_imp = pd.DataFrame(fce_init_rows)
    fce_spec = pd.DataFrame(fce_spec_rows)

    fce_pooled = (
        fce_by_imp.groupby(
            ["target_predictor", "term"],
            as_index=False,
            sort=False,
        )
        .agg(
            pooled_coefficient=("coefficient", "mean"),
            between_imputation_sd=("coefficient", "std"),
            imputations=("imputation", "nunique"),
        )
    )

    if not fce_pooled["imputations"].eq(N_IMPUTATIONS).all():
        raise AssertionError(
            "FCE pooled coefficient does not contain all 30 imputations."
        )

    fce_init = (
        fce_init_by_imp.groupby(
            "target_predictor",
            as_index=False,
            sort=False,
        )
        .agg(
            initialization_value=(
                "completed_development_mean",
                "mean",
            ),
            between_imputation_sd=(
                "completed_development_mean",
                "std",
            ),
            imputations=("imputation", "nunique"),
        )
    )

    if not fce_init["imputations"].eq(N_IMPUTATIONS).all():
        raise AssertionError(
            "FCE initialization does not contain all 30 imputations."
        )

    fce_by_imp.to_csv(
        out_dir / "fce_conditional_coefficients_by_imputation.csv",
        index=False,
    )
    fce_pooled.to_csv(
        out_dir / "fce_pooled_conditional_coefficients.csv",
        index=False,
    )
    fce_init_by_imp.to_csv(
        out_dir / "fce_initialization_by_imputation.csv",
        index=False,
    )
    fce_init.to_csv(
        out_dir / "fce_initialization_values.csv",
        index=False,
    )
    fce_spec.to_csv(
        out_dir / "fce_conditional_model_specification.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 8) Model/refit summary and deterministic hash manifest.
    # ------------------------------------------------------------------
    stage_counts = (
        harm_pre["cr_rrt_stage"]
        .astype(int)
        .value_counts()
        .sort_index()
    )

    model_summary = pd.DataFrame(
        [
            {
                "model": "HARMONIZED_MIMIC_CR_RRT",
                "n": len(harm_fit_df),
                "events": harmonized_events,
                "event_rate": (
                    harmonized_events / len(harm_fit_df)
                ),
                "kdigo_stage_1_n":
                    int(stage_counts.get(1, 0)),
                "kdigo_stage_2_n":
                    int(stage_counts.get(2, 0)),
                "kdigo_stage_3_n":
                    int(stage_counts.get(3, 0)),
                "parameters_excluding_intercept":
                    len(PRIMARY_PREDICTORS),
                "imputations": N_IMPUTATIONS,
                "mice_cycles": MICE_CYCLES,
                "pmm_donors": PMM_DONORS,
                "model_seed": MODEL_SEED,
            }
        ]
    )
    model_summary.to_csv(
        qc_dir / "HARMONIZED_MODEL_SUMMARY.csv",
        index=False,
    )


    output_files = sorted(
        [
            p
            for root in (out_dir, qc_dir)
            for p in root.rglob("*")
            if p.is_file()
            and p.name != "HARMONIZED_MODEL_MANIFEST.json"
        ],
        key=lambda p: str(p),
    )

    output_hashes = {
        str(p.relative_to(project)).replace("\\", "/"):
            sha256_file(p)
        for p in output_files
    }

    manifest = {
        "status": "PASS",
        "gate": "Harmonized model — harmonized MIMIC Cr/RRT transportability/refit",
        "analysis_scope": "development_model_fitting",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_mimic_science_changed": False,
        "eicu_data_read": False,
        "eicu_outcome_accessed": False,
        "mimic_outcome_used_for_development_refit": True,
        "scientific_definition": {
            "hf_krt_target":
                "canonical outcome-blind MIMIC prior-HF/full-KRT-eligible target",
            "creatinine":
                "rolling 48h/7d canonical MIMIC creatinine component only",
            "strict_outpatient_historical_creatinine_used": False,
            "urine_output_kdigo_used": False,
            "acute_rrt":
                "canonical active_rrt_any_by_landmark; forces Stage 3",
            "selection":
                "first Cr/RRT-positive ICU episode per subject by intime, stay_id",
        },
        "canonical_reference": {
            "n": EXPECTED_PRIMARY_N,
            "events": EXPECTED_PRIMARY_EVENTS,
        },
        "harmonized_refit": {
            "n": int(len(harm_fit_df)),
            "events": harmonized_events,
            "event_rate": float(
                harmonized_events / len(harm_fit_df)
            ),
            "stage_1_n": int(stage_counts.get(1, 0)),
            "stage_2_n": int(stage_counts.get(2, 0)),
            "stage_3_n": int(stage_counts.get(3, 0)),
            "subject_set_is_subset_of_canonical": True,
            "canonical_only_dropped_subjects": int(
                len(csub - hsub)
            ),
            "different_selected_stay_subjects": int(
                shifted.sum()
            ),
        },
        "preoutcome_lock_inside_harmonized_model": {
            "cohort_path":
                str(p_harm_cohort.relative_to(project)).replace("\\", "/"),
            "cohort_sha256": preoutcome_membership_sha,
            "subject_id_set_sha256_sorted_newline":
                sha256_sorted_ids(
                    harm_pre["subject_id"].astype("int64")
                ),
            "stay_id_set_sha256_sorted_newline":
                sha256_sorted_ids(
                    harm_pre["stay_id"].astype("int64")
                ),
            "predictor_matrix_path":
                str(p_harm_matrix.relative_to(project)).replace("\\", "/"),
            "predictor_matrix_sha256": preoutcome_matrix_sha,
        },
        "modeling": {
            "family":
                "multivariable binary logistic regression, maximum likelihood",
            "predictors": PRIMARY_PREDICTORS,
            "number_of_imputations": N_IMPUTATIONS,
            "mice_cycles": MICE_CYCLES,
            "pmm_donors": PMM_DONORS,
            "outcome_in_mimic_development_imputation": True,
            "model_seed": MODEL_SEED,
            "canonical_model_reproduction_max_abs_coef_diff":
                maxdiff,
        },
        "fce": {
            "targets": FCE_TARGETS,
            "conditional_model_family":
                "ordinary linear regression",
            "training_covariates":
                "all other fixed model predictors plus MIMIC development outcome",
            "pooling":
                "mean of 30 conditional coefficient point estimates",
            "initialization":
                "mean of 30 completed-development marginal means",
            "external_outcome_surrogate":
                "current locked-model predicted probability",
            "external_update_order":
                "canonical model-column order",
            "external_convergence_tolerance":
                FCE_CONVERGENCE_TOL,
            "external_max_full_cycles":
                FCE_MAX_CYCLES,
            "external_stochastic_draws": False,
            "external_pmm_donor_sampling": False,
        },
        "canonical_input_artifact_hashes": input_artifact_hashes,
        "predictor_reference_checks": predictor_repro_rows,
        "generated_output_hashes": output_hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": sm.__version__,
            "duckdb": duckdb.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "next_action":
            "Independent assessment of model outputs is recorded separately",
        "outcome_attachment_authorized": False,
    }

    p_manifest = qc_dir / "HARMONIZED_MODEL_MANIFEST.json"
    p_manifest.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    manifest_sha = sha256_file(p_manifest)

    print()
    print("=" * 84)
    print("HARMONIZED_MODEL EXECUTION: PASS")
    print("=" * 84)
    print(f"Canonical primary N             : {len(canonical):,}")
    print(f"Harmonized Cr/RRT N             : {len(harm_pre):,}")
    print(f"Canonical-only dropped          : {len(csub - hsub):,}")
    print(f"Different selected stay         : {int(shifted.sum()):,}")
    print(f"Harmonized deaths               : {harmonized_events:,}")
    print(
        "Harmonized stages              : "
        f"S1={int(stage_counts.get(1, 0)):,} "
        f"S2={int(stage_counts.get(2, 0)):,} "
        f"S3={int(stage_counts.get(3, 0)):,}"
    )
    print(
        "Canonical model reproduction   : "
        f"PASS (max abs coef diff={maxdiff:.3e})"
    )
    print(f"Harmonized model manifest SHA256          : {manifest_sha}")
    print("eICU data/outcome accessed       : NO")
    print("Verification scope              : model fitting and built-in integrity checks")
    print("Outcome attachment authorized    : NO")
    print("=" * 84)


if __name__ == "__main__":
    main()
