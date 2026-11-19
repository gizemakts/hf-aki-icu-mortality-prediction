#!/usr/bin/env python3
"""
Final outcome-blind eICU pre-outcome cohort/predictor build.

Scientific role
---------------
This script implements the already-fixed eICU external-validation contracts
AFTER the harmonized MIMIC Cr/RRT refit (Harmonized model) has been fixed.

It builds, without reading eICU mortality/discharge-status fields:
  * the final stay-level Cr/RRT phenotype after chronic-KRT exclusion;
  * the final first-qualifying episode per uniquePid;
  * the final 19-column raw predictor matrix;
  * deterministic MIMIC-trained fixed chained-equation (FCE) imputation;
  * the final prediction-ready 19-column matrix;
  * site, episode, renal, missingness, FCE, provenance, and hash QC.

It does NOT attach eICU outcome, fit/recalibrate a model in eICU, or compute
AUROC/Brier/calibration/other performance. A PASS execution is NOT itself a
lock. Independent artifact-level review is required before the final
pre-outcome SHA lock and before any eICU outcome attachment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow
import scipy
from scipy.special import expit


# -----------------------------------------------------------------------------
# Study cohort and source-data constants
# -----------------------------------------------------------------------------
EXPECTED_SITE_LIST_SHA = (
    "d517dd26275495d4c68e17872812a9cbc53c4d5056a666b2d9ef92da6fbb6905"
)
EXPECTED_SITE_ID_SET_SHA = (
    "c34a413540c99d1bec9ca030e20606582e8e9084ccac802c912fb62bef656786"
)
EXPECTED_SITE_N = 145
EXPECTED_ADULT_PLUS24_SITE_N = 98298
EXPECTED_HF_PLUS24_SITE_N = 15034

# Exact eICU v2.0 raw-file fingerprints already fixed in repository QC.
EXPECTED_RAW_SHA256 = {
    "patient.csv.gz":
        "2acf25f75144c5fc8a2a5b68c6741edac2dc7ca51021127f73e12f7428dc392b",
    "pastHistory.csv.gz":
        "c8d05cd14ce2f0730750354b8dcd395905ad6a958be4f42b45cbeac02e0aaf84",
    "nurseCharting.csv.gz":
        "d10444dc6b530dfd198b42f2841de7b76045570fbf10b08e991c133006661c2c",
    "lab.csv.gz":
        "91b55ed330015d31c424499cdae3b68e3b4cfe0c10e542548b5461520d93bc7a",
    "infusionDrug.csv.gz":
        "f1c958b92a676b932626638f52d0cd5bb13fa53bf5a0ec25606ff4beefc9fc3b",
    "treatment.csv.gz":
        "115ce152107aeab0f89f4c50c9b4b03a127a2d80618f70584d39e4b3fca4bd6b",
    "intakeOutput.csv.gz":
        "f809b425ea89b25c735fa8a7486e204717a3bb4d5f256af0170318ce9baa5121",
    "diagnosis.csv.gz":
        "1bc15709cfe9b376ef1da8b1ea47f61993e3d7dcc942f7c1eccbf162e5f2e0c8",
}


# -----------------------------------------------------------------------------
# Fixed scientific constants
# -----------------------------------------------------------------------------
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

UF_TREATMENT_LABELS = {
    "renal|dialysis|ultrafiltration (fluid removal only)",
    "renal|dialysis|ultrafiltration (fluid removal only)|emergent",
    "renal|dialysis|ultrafiltration (fluid removal only)|for acute renal failure",
}

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

NONRENAL_PREDICTORS = PRIMARY_PREDICTORS[:17]
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
FCE_TOL = 1e-6
FCE_MAX_CYCLES = 50

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

DRUG_PATTERNS = [
    ("norepinephrine", re.compile(
        r"\b(?:norepinephrine|noradrenaline|levophed)\b"
        r"|^nss with levo \(ml/hr\)$"
        r"|^nss w/ levo/vaso \(ml/hr\)$", re.I)),
    ("epinephrine", re.compile(
        r"\b(?:epinephrine|epinepherine|adrenalin|adrenaline)\b"
        r"|^epi \(mcg/min\)$", re.I)),
    ("phenylephrine", re.compile(
        r"\bphenylephrine\b|\bneo[- ]?synephrine\b|\bneosynephrine\b|\bneosynsprine\b",
        re.I)),
    ("vasopressin", re.compile(r"\bvasopressin\b", re.I)),
    ("dopamine", re.compile(r"\bdopamine\b", re.I)),
    ("dobutamine", re.compile(r"\bdobu\w*\b", re.I)),
    ("milrinone", re.compile(r"\b(?:milrinone|milronone|primacor|primacore)\b", re.I)),
]

PROHIBITED_EXTERNAL_COLUMN_TOKENS = (
    "hospitaldischargestatus",
    "unitdischargestatus",
    "hospitaldischargelocation",
    "unitdischargelocation",
    "mortality",
    "death",
    "expire",
    "survival",
    "auroc",
    "auc",
    "brier",
    "calibration",
)

# Fixed admissible transformed scales for the 14 externally imputed targets.
# Open raw bounds are represented here with the limiting closed boundary; a
# tiny numerical tolerance is allowed when checking deterministic OLS output.
FCE_ADMISSIBLE = {
    "heart_rate_max_per10": (0.0, 30.0, False, False),
    "map_min_per10": (0.0, 30.0, False, False),
    "resp_rate_max_per5": (0.0, 14.0, False, False),
    "temperature_max_abs_deviation_from_37c": (0.0, 27.0, True, False),
    "spo2_deficit_per5": (0.0, 20.0, True, False),
    "gcs_deficit_from15": (0.0, 12.0, True, True),
    "hemoglobin_min_gdl": (0.0, 50.0, False, True),
    "wbc_max_log2": (None, math.log2(1000.0), True, True),
    "platelet_min_log2": (None, math.log2(10000.0), True, True),
    "bun_max_log2": (None, math.log2(300.0), True, True),
    "bicarbonate_min_per5": (0.0, 2000.0, False, True),
    "sodium_max_abs_deviation_from140_per5": (0.0, 28.0, True, False),
    "potassium_max_mmol_l": (0.0, 30.0, False, True),
    "glucose_max_log2": (None, math.log2(10000.0), True, True),
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_sorted_strings(values) -> str:
    vals = sorted({str(x).strip() for x in values if pd.notna(x)})
    payload = "\n".join(vals) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_sorted_ints(values) -> str:
    vals = sorted({int(x) for x in values if pd.notna(x)})
    payload = "\n".join(str(x) for x in vals) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_file(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_hash(path: Path, expected: str, label: str) -> str:
    got = sha256_file(require_file(path))
    if got.lower() != expected.lower():
        raise AssertionError(
            f"{label} SHA-256 mismatch\nexpected={expected}\nobserved={got}\npath={path}"
        )
    return got


def discover(root: Path, stem: str) -> Path:
    wanted = {f"{stem}.csv".lower(), f"{stem}.csv.gz".lower()}
    matches = sorted(
        [p.resolve() for p in root.rglob("*") if p.is_file() and p.name.lower() in wanted],
        key=lambda p: p.as_posix().lower(),
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"{stem}: expected exactly one {stem}.csv(.gz), found {len(matches)}"
        )
    return matches[0]


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def assert_unique(df: pd.DataFrame, key: str, label: str) -> None:
    if df[key].isna().any():
        raise AssertionError(f"{label}: missing {key}")
    if df[key].duplicated().any():
        raise AssertionError(f"{label}: duplicate {key}")


def assert_safe_columns(columns, label: str) -> None:
    bad = [
        str(c) for c in columns
        if any(tok in str(c).lower() for tok in PROHIBITED_EXTERNAL_COLUMN_TOKENS)
    ]
    if bad:
        raise AssertionError(f"{label}: prohibited external columns detected: {bad}")


def age_for_eligibility(x):
    if x is None:
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    if s.startswith(">"):
        return 90.0
    try:
        return float(s)
    except Exception:
        return np.nan


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


def normalized_icd_tokens(value):
    if value is None:
        return set()
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return set()
    out = set()
    for token in s.split(","):
        token = re.sub(r"[^A-Z0-9]", "", token.strip().upper())
        if token:
            out.add(token)
    return out


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


def relative(project: Path, path: Path) -> str:
    return str(path.resolve().relative_to(project.resolve())).replace("\\", "/")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    assert_safe_columns(df.columns, path.name)
    df.to_csv(path, index=False)


def safe_predict_probability(df: pd.DataFrame, coef: dict[str, float]) -> np.ndarray:
    lp = np.full(len(df), float(coef["const"]), dtype=float)
    for c in PRIMARY_PREDICTORS:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if not np.isfinite(v).all():
            raise AssertionError(f"Non-finite predictor passed to locked model: {c}")
        lp += float(coef[c]) * v
    if not np.isfinite(lp).all():
        raise AssertionError("Non-finite locked-model linear predictor")
    p = expit(lp)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AssertionError("Invalid locked-model probability")
    return p


def check_fce_scale(series: pd.Series, target: str, mask: pd.Series) -> tuple[int, float, float]:
    vals = pd.to_numeric(series.loc[mask], errors="coerce").to_numpy(float)
    if len(vals) == 0:
        return 0, np.nan, np.nan
    if not np.isfinite(vals).all():
        return int((~np.isfinite(vals)).sum()), float(np.nanmin(vals)), float(np.nanmax(vals))
    lo, hi, lo_closed, hi_closed = FCE_ADMISSIBLE[target]
    tol = 1e-10
    bad = np.zeros(len(vals), dtype=bool)
    if lo is not None:
        bad |= vals < lo - tol if lo_closed else vals <= lo
    if hi is not None:
        bad |= vals > hi + tol if hi_closed else vals >= hi
    return int(bad.sum()), float(vals.min()), float(vals.max())


def apply_fce(
    raw: pd.DataFrame,
    coef: dict[str, float],
    fce_pool: pd.DataFrame,
    init: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Apply the fixed deterministic external FCE algorithm."""
    cur = raw.copy()
    original = raw.copy()
    masks = {t: original[t].isna() for t in FCE_TARGETS}

    init_map = dict(zip(init["target_predictor"], init["initialization_value"]))
    if list(init["target_predictor"]) != FCE_TARGETS:
        # Enforce canonical update order explicitly, not just set equality.
        raise AssertionError("FCE initialization target order differs from fixed canonical order")

    for t in FCE_TARGETS:
        if t not in init_map or not np.isfinite(float(init_map[t])):
            raise AssertionError(f"Missing/non-finite FCE initialization: {t}")
        cur.loc[masks[t], t] = float(init_map[t])

    if cur[PRIMARY_PREDICTORS].isna().any().any():
        missing = cur[PRIMARY_PREDICTORS].isna().sum()
        missing = missing[missing.gt(0)].to_dict()
        raise AssertionError(
            "Missing model inputs remain after fixed FCE initialization; "
            f"non-FCE or unexpected missingness={missing}"
        )

    # Parse each fixed conditional vector exactly once.
    beta_by_target: dict[str, dict[str, float]] = {}
    for target in FCE_TARGETS:
        g = fce_pool.loc[fce_pool["target_predictor"].eq(target)].copy()
        if len(g) != 20 or g["term"].duplicated().any():
            raise AssertionError(f"FCE pooled vector malformed for {target}: rows={len(g)}")
        beta_by_target[target] = dict(zip(g["term"], g["pooled_coefficient"]))
        expected_terms = {"const", "hospital_expire_flag"} | (set(PRIMARY_PREDICTORS) - {target})
        if set(beta_by_target[target]) != expected_terms:
            raise AssertionError(f"FCE term set mismatch for {target}")

    p_old = safe_predict_probability(cur, coef)
    cycle_rows = []
    converged = False

    for cycle in range(1, FCE_MAX_CYCLES + 1):
        # Fixed contract: use the current locked-model probability as the
        # unavailable outcome surrogate throughout this full cycle.
        outcome_surrogate = p_old.copy()

        for target in FCE_TARGETS:
            mask = masks[target]
            if not mask.any():
                continue
            beta = beta_by_target[target]
            pred = np.full(len(cur), float(beta["const"]), dtype=float)
            for term in PRIMARY_PREDICTORS:
                if term == target:
                    continue
                x = pd.to_numeric(cur[term], errors="coerce").to_numpy(float)
                if not np.isfinite(x).all():
                    raise AssertionError(
                        f"Non-finite FCE covariate {term} while updating {target}"
                    )
                pred += float(beta[term]) * x
            pred += float(beta["hospital_expire_flag"]) * outcome_surrogate
            if not np.isfinite(pred).all():
                raise AssertionError(f"Non-finite FCE update for {target}")
            cur.loc[mask, target] = pred[mask.to_numpy()]

        p_new = safe_predict_probability(cur, coef)
        delta = np.abs(p_new - p_old)
        cycle_rows.append({
            "cycle": cycle,
            "max_abs_probability_delta": float(delta.max()) if len(delta) else 0.0,
            "mean_abs_probability_delta": float(delta.mean()) if len(delta) else 0.0,
            "patients_at_or_above_tolerance": int((delta >= FCE_TOL).sum()),
            "converged_all_patients": bool((delta < FCE_TOL).all()),
        })
        p_old = p_new
        if (delta < FCE_TOL).all():
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"Fixed external FCE failed to converge within {FCE_MAX_CYCLES} cycles"
        )

    # Observed values must be exactly preserved.
    preserve_rows = []
    for t in FCE_TARGETS:
        obs = ~masks[t]
        before = pd.to_numeric(original.loc[obs, t], errors="coerce").to_numpy(float)
        after = pd.to_numeric(cur.loc[obs, t], errors="coerce").to_numpy(float)
        if len(before):
            diff = np.abs(before - after)
            maxdiff = float(diff.max())
            changed = int((diff > 0).sum())
        else:
            maxdiff = 0.0
            changed = 0
        bad_scale_n, imp_min, imp_max = check_fce_scale(cur[t], t, masks[t])
        preserve_rows.append({
            "target_predictor": t,
            "observed_n": int(obs.sum()),
            "missing_imputed_n": int(masks[t].sum()),
            "observed_values_changed_n": changed,
            "observed_max_abs_difference": maxdiff,
            "imputed_scale_violation_n": bad_scale_n,
            "imputed_minimum": imp_min,
            "imputed_maximum": imp_max,
        })
        if changed != 0 or maxdiff != 0.0:
            raise AssertionError(f"FCE overwrote observed values for {t}")
        if bad_scale_n:
            raise RuntimeError(
                f"Fixed FCE generated {bad_scale_n} inadmissible imputed values for {t}; "
                "no post-hoc clipping rule is authorized"
            )

    if cur[PRIMARY_PREDICTORS].isna().any().any():
        raise AssertionError("Missing model inputs remain after FCE convergence")
    arr = cur[PRIMARY_PREDICTORS].to_numpy(float)
    if not np.isfinite(arr).all():
        raise AssertionError("Non-finite model inputs remain after FCE convergence")

    return (
        cur,
        pd.DataFrame(cycle_rows),
        pd.DataFrame(preserve_rows),
        p_old,
    )


# -----------------------------------------------------------------------------
# Main build
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--eicu-root", required=True)
    args = ap.parse_args()

    project = Path(args.project_root).expanduser().resolve()
    eicu_root = Path(args.eicu_root).expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(project)
    if not eicu_root.is_dir():
        raise FileNotFoundError(eicu_root)

    out_dir = project / "outputs" / "08_external_transport" / "eicu_final_preoutcome"
    qc_dir = (
        project / "quality_control" / "eicu_external_validation" / "eicu_final_preoutcome"
    )
    for d in (out_dir, qc_dir):
        if d.exists() and any(d.iterdir()):
            raise FileExistsError(
                f"{d} already exists and is non-empty. Successful pre-outcome "
                "build artifacts are immutable; do not overwrite them."
            )
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 92)
    print("FINAL eICU PRE-OUTCOME COHORT / PREDICTOR BUILD")
    print("=" * 92)
    print("Project root :", project)
    print("eICU root    :", eicu_root)
    print("eICU outcome : NOT ACCESSED")
    print()

    # ------------------------------------------------------------------
    # 0) Fixed-governance preflight.
    # ------------------------------------------------------------------
    # Harmonized MIMIC model artifacts created by fit_harmonized_model.py.
    model_dir = project / "outputs" / "08_external_transport" / "mimic_cr_rrt_harmonized_refit"
    model_files = (
        "harmonized_cr_rrt_locked_coefficient_vector.csv",
        "fce_pooled_conditional_coefficients.csv",
        "fce_initialization_values.csv",
        "fce_conditional_model_specification.csv",
    )
    consumed_hashes = {}
    for name in model_files:
        source = model_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        consumed_hashes[relative(project, source)] = sha256_file(source)

    site_list_path = (
        project / "quality_control" / "eicu_external_validation"
        / "amended_site_eligibility_audit" / "AMENDED_SITE_ELIGIBLE_HOSPITALS.csv"
    )
    verify_hash(site_list_path, EXPECTED_SITE_LIST_SHA, "Fixed 145-site list")
    site_df = pd.read_csv(site_list_path)
    if list(site_df.columns) != ["hospital_id"]:
        raise AssertionError("Fixed site-list schema changed")
    site_ids = sorted(pd.to_numeric(site_df["hospital_id"], errors="raise").astype(int).tolist())
    if len(site_ids) != EXPECTED_SITE_N or len(set(site_ids)) != EXPECTED_SITE_N:
        raise AssertionError("Fixed site-list count/uniqueness drift")
    if sha256_sorted_ints(site_ids) != EXPECTED_SITE_ID_SET_SHA:
        raise AssertionError("Fixed 145-site ID-set SHA drift")

    print("MODEL INPUT CHECK: PASS")

    # ------------------------------------------------------------------
    # 1) Discover and directly fingerprint every raw eICU file used here.
    # ------------------------------------------------------------------
    raw = {
        "patient": discover(eicu_root, "patient"),
        "pastHistory": discover(eicu_root, "pastHistory"),
        "nurseCharting": discover(eicu_root, "nurseCharting"),
        "lab": discover(eicu_root, "lab"),
        "infusionDrug": discover(eicu_root, "infusionDrug"),
        "treatment": discover(eicu_root, "treatment"),
        "intakeOutput": discover(eicu_root, "intakeOutput"),
        "diagnosis": discover(eicu_root, "diagnosis"),
    }

    raw_fingerprints = {}
    print("Hashing directly used eICU v2.0 raw inputs...")
    for key, path in raw.items():
        expected = EXPECTED_RAW_SHA256.get(path.name)
        if expected is None:
            raise AssertionError(
                f"Raw file name differs from fixed eICU v2.0 identity: {path.name}"
            )
        print("  SHA256", path.name, flush=True)
        got = sha256_file(path)
        if got != expected:
            raise AssertionError(
                f"Raw eICU SHA mismatch for {path.name}\nexpected={expected}\nobserved={got}"
            )
        raw_fingerprints[path.name] = {
            "sha256": got,
            "size_bytes": int(path.stat().st_size),
        }
    print("RAW INPUT SHA-256 PREFLIGHT: PASS")

    PATIENT = raw["patient"]
    PH = raw["pastHistory"]
    NURSE = raw["nurseCharting"]
    LAB = raw["lab"]
    INF = raw["infusionDrug"]
    TREAT = raw["treatment"]
    IO = raw["intakeOutput"]
    DX = raw["diagnosis"]

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4")

    # ------------------------------------------------------------------
    # 2) Outcome-safe patient skeleton, adult +24 h, fixed site rule.
    # ------------------------------------------------------------------
    patient = con.execute(f"""
        SELECT
            patientUnitStayID AS stay_id,
            patientHealthSystemStayID AS hstay_id,
            uniquePid AS unique_pid,
            hospitalID AS hospital_id,
            age AS age,
            gender AS gender,
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
    assert_safe_columns(patient.columns, "patient_safe")
    assert_unique(patient, "stay_id", "patient table")
    if patient["unique_pid"].isna().any() or patient["hstay_id"].isna().any():
        raise AssertionError("Missing uniquePid or patientHealthSystemStayID")

    patient["age_num_eligibility"] = patient["age"].map(age_for_eligibility)
    unit_off = pd.to_numeric(patient["unit_discharge_offset"], errors="coerce")
    hosp_off = pd.to_numeric(patient["hospital_discharge_offset"], errors="coerce")
    adult24 = patient.loc[
        patient["age_num_eligibility"].fillna(-1).ge(18)
        & unit_off.gt(1440)
        & hosp_off.gt(1440)
    ].copy()
    site_adult24 = adult24.loc[adult24["hospital_id"].isin(site_ids)].copy()
    if len(site_adult24) != EXPECTED_ADULT_PLUS24_SITE_N:
        raise AssertionError(
            f"Fixed site-eligible adult +24h count drift: "
            f"expected {EXPECTED_ADULT_PLUS24_SITE_N:,}, observed {len(site_adult24):,}"
        )

    con.register("adult24_df", site_adult24[["stay_id"]])
    con.execute("CREATE TEMP TABLE adult24_ids AS SELECT * FROM adult24_df")
    values_sql = ",".join("'" + v.replace("'", "''") + "'" for v in HF_VALUES)
    hf_ids = con.execute(f"""
        SELECT DISTINCT ph.patientUnitStayID AS stay_id
        FROM read_csv_auto(
            '{qpath(PH)}',
            header=true,
            sample_size=200000
        ) ph
        JOIN adult24_ids a ON ph.patientUnitStayID=a.stay_id
        WHERE ph.pastHistoryOffset<=1440
          AND ph.pastHistoryEnteredOffset<=1440
          AND ph.pastHistoryValue IN ({values_sql})
          AND ph.pastHistoryValueText=ph.pastHistoryValue
          AND ph.pastHistoryPath=
              '{HF_PATH_PREFIX.replace("'", "''")}' || ph.pastHistoryValue
    """).df()

    cand = site_adult24.merge(hf_ids.assign(hf_proxy=1), on="stay_id", how="inner")
    assert_unique(cand, "stay_id", "site-eligible HF +24h candidate stays")
    if len(cand) != EXPECTED_HF_PLUS24_SITE_N:
        raise AssertionError(
            f"Fixed site-eligible HF +24h count drift: expected "
            f"{EXPECTED_HF_PLUS24_SITE_N:,}, observed {len(cand):,}"
        )
    con.register("cand_df", cand)
    con.execute("CREATE TEMP TABLE cand AS SELECT * FROM cand_df")
    print("ADULT +24H / HF / SITE RECONSTRUCTION: PASS")
    print(f"  site-eligible adult +24h stays : {len(site_adult24):,}")
    print(f"  site-eligible HF +24h stays    : {len(cand):,}")

    # ------------------------------------------------------------------
    # 3) Fixed chronic-KRT evidence.
    # ------------------------------------------------------------------
    ph_renal = con.execute(f"""
        SELECT
            p.patientUnitStayID AS stay_id,
            p.pastHistoryValue AS history_value,
            p.pastHistoryPath AS history_path
        FROM read_csv_auto(
            '{qpath(PH)}',
            header=true,
            sample_size=200000
        ) p
        JOIN cand c ON p.patientUnitStayID=c.stay_id
        WHERE p.pastHistoryOffset<=1440
          AND p.pastHistoryEnteredOffset<=1440
          AND p.pastHistoryValue IN (
              '{PH_HD_VALUE}',
              '{PH_PD_VALUE}',
              '{PH_TRANSPLANT_VALUE}'
          )
    """).df()
    ph_path_norm = ph_renal["history_path"].fillna("").astype(str).str.strip().str.lower()
    ph_value_norm = ph_renal["history_value"].fillna("").astype(str).str.strip().str.lower()
    hd_path = (RENAL_FAILURE_PATH_PREFIX + PH_HD_VALUE).strip().lower()
    pd_path = (RENAL_FAILURE_PATH_PREFIX + PH_PD_VALUE).strip().lower()
    ph_hd = set(ph_renal.loc[ph_value_norm.eq(PH_HD_VALUE.lower()) & ph_path_norm.eq(hd_path), "stay_id"])
    ph_pd = set(ph_renal.loc[ph_value_norm.eq(PH_PD_VALUE.lower()) & ph_path_norm.eq(pd_path), "stay_id"])
    transplant = set(ph_renal.loc[ph_value_norm.eq(PH_TRANSPLANT_VALUE.lower()), "stay_id"])

    dx = con.execute(f"""
        SELECT
            d.patientUnitStayID AS stay_id,
            d.diagnosisString AS diagnosis_string,
            d.ICD9Code AS icd_code
        FROM read_csv_auto(
            '{qpath(DX)}',
            header=true,
            sample_size=200000
        ) d
        JOIN cand c ON d.patientUnitStayID=c.stay_id
        WHERE d.diagnosisOffset<=1440
    """).df()
    diagnosis_norm = dx["diagnosis_string"].fillna("").astype(str).str.strip().str.lower()
    dx["_icd_tokens"] = dx["icd_code"].map(normalized_icd_tokens)
    dx_path_mask = diagnosis_norm.eq("renal|disorder of kidney|esrd (end stage renal disease)")
    dx_code_mask = dx["_icd_tokens"].map(lambda s: bool(s & {"5856", "N186"}))
    dx_dep_mask = dx["_icd_tokens"].map(lambda s: bool(s & {"V4511", "Z992"}))
    dx_chronic = set(dx.loc[dx_path_mask | dx_code_mask | dx_dep_mask, "stay_id"])

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
        JOIN cand c ON t.patientUnitStayID=c.stay_id
        WHERE t.treatmentOffset<=1440
    """).df()
    treatment["_norm"] = treatment["treatment_string"].fillna("").astype(str).str.strip().str.lower()
    treatment["_offset"] = pd.to_numeric(treatment["treatment_offset"], errors="coerce")
    chronic_treatment_norm = {x.lower() for x in CHRONIC_TREATMENT_ALLOWLIST}
    acute_treatment_norm = {x.lower() for x in ACUTE_RRT_TREATMENT_ALLOWLIST}
    tr_chronic = set(treatment.loc[treatment["_norm"].isin(chronic_treatment_norm), "stay_id"])

    renal = cand[
        [
            "stay_id", "hstay_id", "unique_pid", "hospital_id", "age", "gender",
            "hospital_admit_offset", "hospital_discharge_offset",
            "hospital_discharge_year", "unit_discharge_offset", "unit_visit_number",
            "unit_stay_type", "unit_type", "hf_proxy",
        ]
    ].copy()
    renal["chronic_krt_ph_hd"] = renal["stay_id"].isin(ph_hd).astype("int8")
    renal["chronic_krt_ph_pd"] = renal["stay_id"].isin(ph_pd).astype("int8")
    renal["chronic_krt_current_esrd_or_dependence"] = renal["stay_id"].isin(dx_chronic).astype("int8")
    renal["chronic_krt_explicit_treatment"] = renal["stay_id"].isin(tr_chronic).astype("int8")
    renal["renal_transplant_history_qc"] = renal["stay_id"].isin(transplant).astype("int8")
    chronic_cols = [
        "chronic_krt_ph_hd", "chronic_krt_ph_pd",
        "chronic_krt_current_esrd_or_dependence", "chronic_krt_explicit_treatment",
    ]
    renal["definite_chronic_krt"] = renal[chronic_cols].max(axis=1).astype("int8")

    # ------------------------------------------------------------------
    # 4) Fixed rolling creatinine component.
    # ------------------------------------------------------------------
    con.execute(f"""
        CREATE TEMP TABLE cr_raw AS
        SELECT
            l.patientUnitStayID AS stay_id,
            TRY_CAST(l.labResultOffset AS DOUBLE) AS result_offset,
            TRY_CAST(l.labResultRevisedOffset AS DOUBLE) AS revised_offset,
            TRY_CAST(l.labResult AS DOUBLE) AS creatinine
        FROM read_csv_auto(
            '{qpath(LAB)}',
            header=true,
            sample_size=200000
        ) l
        JOIN cand c ON l.patientUnitStayID=c.stay_id
        WHERE lower(l.labName)='creatinine'
          AND l.labTypeID=1
          AND TRY_CAST(l.labResult AS DOUBLE)>0
          AND TRY_CAST(l.labResult AS DOUBLE)<=150
          AND TRY_CAST(l.labResultRevisedOffset AS DOUBLE)<=1440
    """)
    con.execute("""
        CREATE TEMP TABLE cr_same_offset AS
        SELECT
            stay_id,
            result_offset,
            AVG(creatinine) AS cr_avg,
            COUNT(*) AS raw_rows,
            COUNT(DISTINCT creatinine) AS distinct_values
        FROM cr_raw
        GROUP BY stay_id, result_offset
    """)
    con.execute("""
        CREATE TEMP TABLE cr_rolling AS
        SELECT
            *,
            MIN(cr_avg) OVER (
                PARTITION BY stay_id ORDER BY result_offset
                RANGE BETWEEN 2880 PRECEDING AND 1 PRECEDING
            ) AS low48,
            MIN(cr_avg) OVER (
                PARTITION BY stay_id ORDER BY result_offset
                RANGE BETWEEN 10080 PRECEDING AND 1 PRECEDING
            ) AS low7d
        FROM cr_same_offset
        WHERE result_offset>=-525600 AND result_offset<=1440
    """)
    con.execute("""
        CREATE TEMP TABLE cr_events AS
        SELECT
            *,
            CASE
                WHEN low48 IS NULL AND low7d IS NULL THEN NULL
                WHEN low7d IS NOT NULL AND cr_avg>=3.0*low7d THEN 3
                WHEN cr_avg>=4.0 AND (
                    (low48 IS NOT NULL AND cr_avg>=low48+0.3)
                    OR (low7d IS NOT NULL AND cr_avg>=1.5*low7d)
                ) THEN 3
                WHEN low7d IS NOT NULL AND cr_avg>=2.0*low7d THEN 2
                WHEN low48 IS NOT NULL AND cr_avg>=low48+0.3 THEN 1
                WHEN low7d IS NOT NULL AND cr_avg>=1.5*low7d THEN 1
                ELSE 0
            END AS creatinine_stage
        FROM cr_rolling
        WHERE result_offset>=0 AND result_offset<=1440
    """)
    cr_stay = con.execute("""
        SELECT
            c.stay_id,
            MAX(e.creatinine_stage) AS creatinine_stage,
            MAX(CASE WHEN e.creatinine_stage IS NOT NULL THEN 1 ELSE 0 END)
                AS creatinine_evaluable,
            COUNT(e.result_offset) AS current_creatinine_events,
            SUM(CASE WHEN e.raw_rows>1 THEN 1 ELSE 0 END) AS same_offset_duplicate_groups
        FROM cand c
        LEFT JOIN cr_events e USING(stay_id)
        GROUP BY c.stay_id
    """).df()
    renal = renal.merge(cr_stay, on="stay_id", how="left", validate="1:1")
    renal["creatinine_evaluable"] = pd.to_numeric(
        renal["creatinine_evaluable"], errors="coerce"
    ).fillna(0).astype("int8")
    renal["creatinine_stage"] = pd.to_numeric(renal["creatinine_stage"], errors="coerce")

    # ------------------------------------------------------------------
    # 5) Fixed acute RRT positive evidence + isolated-UF flag.
    # ------------------------------------------------------------------
    acute_treat = treatment["_norm"].isin(acute_treatment_norm) & treatment["_offset"].between(
        0, 1440, inclusive="both"
    )
    tr_primary = set(treatment.loc[acute_treat, "stay_id"])
    tr_uf = set(treatment.loc[acute_treat & treatment["_norm"].isin(UF_TREATMENT_LABELS), "stay_id"])
    tr_nonuf = set(treatment.loc[acute_treat & ~treatment["_norm"].isin(UF_TREATMENT_LABELS), "stay_id"])

    io = con.execute(f"""
        SELECT
            i.patientUnitStayID AS stay_id,
            i.intakeOutputOffset AS io_offset,
            i.intakeOutputEntryOffset AS entry_offset,
            i.cellPath AS cell_path,
            i.cellValueNumeric AS cell_value_numeric
        FROM read_csv_auto(
            '{qpath(IO)}',
            header=true,
            sample_size=200000
        ) i
        JOIN cand c ON i.patientUnitStayID=c.stay_id
        WHERE i.intakeOutputOffset<=1440
          AND i.intakeOutputEntryOffset<=1440
    """).df()
    io["_path"] = io["cell_path"].fillna("").astype(str).str.strip().str.lower()
    io["_offset"] = pd.to_numeric(io["io_offset"], errors="coerce")
    io["_value"] = pd.to_numeric(io["cell_value_numeric"], errors="coerce")
    io_paths = {x.lower() for x in IO_RRT_PATH_ALLOWLIST}
    io["_explicit"] = io["_path"].isin(io_paths)
    io["_acute"] = (
        io["_explicit"]
        & io["_value"].gt(0)
        & io["_offset"].between(0, 1440, inclusive="both")
    )
    io["_uf"] = (
        io["_path"].str.contains("ultrafiltration", regex=False)
        | io["_path"].str.contains("uf removed", regex=False)
    )
    io_primary = set(io.loc[io["_acute"], "stay_id"])
    io_uf = set(io.loc[io["_acute"] & io["_uf"], "stay_id"])
    io_nonuf = set(io.loc[io["_acute"] & ~io["_uf"], "stay_id"])

    renal["rrt_treatment_positive_0_24"] = renal["stay_id"].isin(tr_primary).astype("int8")
    renal["rrt_intakeoutput_positive_0_24"] = renal["stay_id"].isin(io_primary).astype("int8")
    renal["primary_rrt_positive_raw"] = renal[
        ["rrt_treatment_positive_0_24", "rrt_intakeoutput_positive_0_24"]
    ].max(axis=1).astype("int8")
    renal["primary_rrt_positive_after_chronic_exclusion"] = (
        renal["primary_rrt_positive_raw"].eq(1) & renal["definite_chronic_krt"].eq(0)
    ).astype("int8")
    renal["uf_evidence_0_24"] = renal["stay_id"].isin(tr_uf | io_uf).astype("int8")
    renal["nonuf_rrt_evidence_0_24"] = renal["stay_id"].isin(tr_nonuf | io_nonuf).astype("int8")
    renal["isolated_uf_after_chronic_exclusion"] = (
        renal["uf_evidence_0_24"].eq(1)
        & renal["nonuf_rrt_evidence_0_24"].eq(0)
        & renal["definite_chronic_krt"].eq(0)
    ).astype("int8")

    renal["eicu_cr_rrt_stage"] = renal["creatinine_stage"]
    renal.loc[
        renal["primary_rrt_positive_after_chronic_exclusion"].eq(1),
        "eicu_cr_rrt_stage",
    ] = 3.0
    renal.loc[renal["definite_chronic_krt"].eq(1), "eicu_cr_rrt_stage"] = np.nan
    renal["cr_rrt_aki_positive"] = (
        renal["definite_chronic_krt"].eq(0)
        & pd.to_numeric(renal["eicu_cr_rrt_stage"], errors="coerce").ge(1)
    )

    # UF suppression state for the later fixed-membership sensitivity.
    def uf_suppressed_state(row) -> str:
        if not bool(row["isolated_uf_after_chronic_exclusion"]):
            s = row["eicu_cr_rrt_stage"]
            return f"STAGE_{int(s)}" if pd.notna(s) else "RENAL_UNEVALUABLE"
        cs = row["creatinine_stage"]
        if pd.isna(cs):
            return "RENAL_UNEVALUABLE"
        return f"STAGE_{int(cs)}"

    renal["uf_suppressed_fixed_cohort_state"] = renal.apply(uf_suppressed_state, axis=1)

    # Save stay-level phenotype evidence BEFORE episode selection.
    renal_path = out_dir / "eicu_renal_stay_level_preoutcome.parquet"
    assert_safe_columns(renal.columns, "renal stay-level artifact")
    renal.sort_values("stay_id").to_parquet(renal_path, index=False)

    # ------------------------------------------------------------------
    # 6) Final episode selection AFTER all stay-level cohort eligibility.
    # ------------------------------------------------------------------
    qual = renal.loc[renal["cr_rrt_aki_positive"]].copy()
    if len(qual) == 0:
        raise RuntimeError("No qualifying Cr/RRT AKI stays after fixed stay-level gates")

    # Hospital-admission invariants among qualifying stays.
    hstay_qc = (
        qual.groupby(["unique_pid", "hstay_id"], dropna=False)
        .agg(
            qualifying_unit_stays=("stay_id", "size"),
            hospital_count=("hospital_id", "nunique"),
            discharge_year_count=("hospital_discharge_year", "nunique"),
            unit_visit_missing=("unit_visit_number", lambda x: int(pd.to_numeric(x, errors="coerce").isna().sum())),
            unit_visit_unique=("unit_visit_number", lambda x: int(pd.to_numeric(x, errors="coerce").nunique(dropna=True))),
        )
        .reset_index()
    )
    if (hstay_qc["hospital_count"] != 1).any():
        raise AssertionError("Qualifying patientHealthSystemStayID maps to multiple hospitals")
    if (hstay_qc["discharge_year_count"] != 1).any():
        raise AssertionError("Qualifying patientHealthSystemStayID has inconsistent/missing discharge year")
    bad_visit = (
        hstay_qc["unit_visit_missing"].gt(0)
        | hstay_qc["unit_visit_unique"].ne(hstay_qc["qualifying_unit_stays"])
    )
    if bad_visit.any():
        raise RuntimeError(
            "Fixed episode rule encountered missing/tied unitVisitNumber among qualifying "
            f"stays in {int(bad_visit.sum())} hospital admissions; explicit pre-outcome "
            "adjudication is required before cohort lock"
        )

    qual["_year"] = pd.to_numeric(qual["hospital_discharge_year"], errors="coerce")
    qual["_visit"] = pd.to_numeric(qual["unit_visit_number"], errors="coerce")
    if qual["_year"].isna().any():
        raise RuntimeError(
            "Qualifying stay has missing hospitalDischargeYear; chronology cannot be fixed"
        )

    # One row per qualifying hospital admission for cross-admission chronology.
    qadm = (
        qual.groupby(["unique_pid", "hstay_id"], as_index=False)
        .agg(
            hospital_id=("hospital_id", "first"),
            discharge_year=("_year", "first"),
            qualifying_unit_stays=("stay_id", "size"),
        )
    )

    patient_episode_rows = []
    selected_rows = []
    ambiguous_pids = set()
    for pid, gadm in qadm.groupby("unique_pid", sort=False):
        min_year = gadm["discharge_year"].min()
        earliest_adm = gadm.loc[gadm["discharge_year"].eq(min_year)]
        ambiguous = len(earliest_adm) > 1
        if ambiguous:
            ambiguous_pids.add(str(pid))
            patient_episode_rows.append({
                "unique_pid": str(pid),
                "qualifying_hospital_admissions": int(len(gadm)),
                "earliest_qualifying_discharge_year": int(min_year),
                "qualifying_hospital_admissions_in_earliest_year": int(len(earliest_adm)),
                "chronology_ambiguous": True,
                "selected_stay_id": np.nan,
            })
            continue

        hstay = earliest_adm.iloc[0]["hstay_id"]
        gh = qual.loc[
            qual["unique_pid"].eq(pid) & qual["hstay_id"].eq(hstay)
        ].copy()
        min_visit = gh["_visit"].min()
        firsts = gh.loc[gh["_visit"].eq(min_visit)]
        if len(firsts) != 1:
            raise RuntimeError(
                f"Unexpected unresolved within-admission visit tie for uniquePid={pid}"
            )
        row = firsts.iloc[0]
        selected_rows.append(row)
        patient_episode_rows.append({
            "unique_pid": str(pid),
            "qualifying_hospital_admissions": int(len(gadm)),
            "earliest_qualifying_discharge_year": int(min_year),
            "qualifying_hospital_admissions_in_earliest_year": 1,
            "chronology_ambiguous": False,
            "selected_stay_id": int(row["stay_id"]),
        })

    selected = pd.DataFrame(selected_rows).drop(columns=["_year", "_visit"], errors="ignore")
    episode_patient_qc = pd.DataFrame(patient_episode_rows)
    if len(selected) == 0:
        raise RuntimeError("Episode selection produced an empty cohort")
    assert_unique(selected, "unique_pid", "final selected cohort patient")
    assert_unique(selected, "stay_id", "final selected cohort stay")
    if len(selected) + len(ambiguous_pids) != qual["unique_pid"].nunique():
        raise AssertionError("Episode selection accounting does not close")

    selected["kdigo_stage_2_indicator"] = selected["eicu_cr_rrt_stage"].eq(2).astype("int8")
    selected["kdigo_stage_3_indicator"] = selected["eicu_cr_rrt_stage"].eq(3).astype("int8")
    if (selected[["kdigo_stage_2_indicator", "kdigo_stage_3_indicator"]].sum(axis=1) > 1).any():
        raise AssertionError("Invalid mutually exclusive KDIGO dummy encoding")

    selected = selected.sort_values(["unique_pid", "stay_id"], kind="mergesort").reset_index(drop=True)
    selected_cohort_cols = [
        "unique_pid", "hstay_id", "stay_id", "hospital_id",
        "age", "hf_proxy", "unit_discharge_offset", "hospital_discharge_offset",
        "hospital_discharge_year", "unit_visit_number", "unit_type", "unit_stay_type",
        "definite_chronic_krt", "cr_rrt_aki_positive",
        "creatinine_evaluable", "creatinine_stage",
        "primary_rrt_positive_after_chronic_exclusion", "eicu_cr_rrt_stage",
        "kdigo_stage_2_indicator", "kdigo_stage_3_indicator",
        "isolated_uf_after_chronic_exclusion", "uf_suppressed_fixed_cohort_state",
    ]
    cohort_pre = selected[selected_cohort_cols].copy()
    assert_safe_columns(cohort_pre.columns, "final cohort preoutcome")
    cohort_path = out_dir / "eicu_final_cohort_preoutcome.parquet"
    cohort_pre.to_parquet(cohort_path, index=False)

    print("FINAL EPISODE SELECTION: PASS")
    print(f"  qualifying renal-positive stays      : {len(qual):,}")
    print(f"  qualifying patients                  : {qual['unique_pid'].nunique():,}")
    print(f"  chronology-ambiguous patients        : {len(ambiguous_pids):,}")
    print(f"  final selected patients/episodes     : {len(selected):,}")

    # ------------------------------------------------------------------
    # 7) Final 17 nonrenal predictors, exact fixed mappings.
    # ------------------------------------------------------------------
    selected_identity = selected[
        ["unique_pid", "hstay_id", "stay_id", "hospital_id", "age", "gender"]
    ].copy()
    con.register("selected_df", selected_identity[["stay_id"]])
    con.execute("CREATE TEMP TABLE selected_ids AS SELECT * FROM selected_df")

    pred = selected_identity[["unique_pid", "hstay_id", "stay_id", "hospital_id"]].copy()
    pred["age_per10y"] = selected_identity["age"].map(age_for_model).to_numpy()
    gender = selected_identity["gender"].fillna("").astype(str).str.strip().str.lower()
    pred["sex_female"] = np.where(
        gender.eq("female"), 1.0,
        np.where(gender.eq("male"), 0.0, np.nan),
    )

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
            JOIN selected_ids s ON n.patientUnitStayID=s.stay_id
            WHERE n.nursingChartOffset BETWEEN 0 AND 1440
              AND n.nursingChartEntryOffset<=1440
              AND n.nursingChartCellTypeCat IN
                  ('Vital Signs','Scores','Other Vital Signs and Infusions')
        )
        SELECT
            stay_id,
            MAX(CASE WHEN vallabel='Heart Rate' AND valname='Heart Rate'
                      AND v>0 AND v<300 THEN v END) AS hr_max,
            MIN(CASE
                WHEN vallabel='Non-Invasive BP' AND valname='Non-Invasive BP Mean'
                     AND v>0 AND v<300 THEN v
                WHEN vallabel='Invasive BP' AND valname='Invasive BP Mean'
                     AND v>0 AND v<300 THEN v
                WHEN vallabel='MAP (mmHg)' AND valname='Value'
                     AND v>0 AND v<300 THEN v
                WHEN vallabel='Arterial Line MAP (mmHg)' AND valname='Value'
                     AND v>0 AND v<300 THEN v END) AS map_min,
            MAX(CASE WHEN vallabel='Respiratory Rate' AND valname='Respiratory Rate'
                      AND v>0 AND v<70 THEN v END) AS rr_max,
            MIN(CASE
                WHEN vallabel='Temperature' AND valname='Temperature (C)'
                     AND v>10 AND v<50 THEN v
                WHEN vallabel='Temperature' AND valname='Temperature (F)'
                     AND v>70 AND v<120 THEN (v-32.0)/1.8 END) AS temp_min_c,
            MAX(CASE
                WHEN vallabel='Temperature' AND valname='Temperature (C)'
                     AND v>10 AND v<50 THEN v
                WHEN vallabel='Temperature' AND valname='Temperature (F)'
                     AND v>70 AND v<120 THEN (v-32.0)/1.8 END) AS temp_max_c,
            MIN(CASE
                WHEN vallabel='O2 Saturation' AND valname='O2 Saturation'
                     AND v>0 AND v<=100 THEN v
                WHEN vallabel='SpO2' AND valname='Value'
                     AND v>0 AND v<=100 THEN v END) AS spo2_min,
            MIN(CASE
                WHEN vallabel='Glasgow coma score' AND valname='GCS Total'
                     AND v BETWEEN 3 AND 15 THEN v
                WHEN vallabel='Score (Glasgow Coma Scale)' AND valname='Value'
                     AND v BETWEEN 3 AND 15 THEN v END) AS gcs_min
        FROM x
        GROUP BY stay_id
    """).df()
    pred = pred.merge(vitals, on="stay_id", how="left", validate="1:1")
    pred["heart_rate_max_per10"] = pred["hr_max"] / 10.0
    pred["map_min_per10"] = pred["map_min"] / 10.0
    pred["resp_rate_max_per5"] = pred["rr_max"] / 5.0
    pred["temperature_max_abs_deviation_from_37c"] = pd.concat(
        [(pred["temp_min_c"] - 37.0).abs(), (pred["temp_max_c"] - 37.0).abs()],
        axis=1,
    ).max(axis=1, skipna=True)
    pred.loc[
        pred[["temp_min_c", "temp_max_c"]].isna().all(axis=1),
        "temperature_max_abs_deviation_from_37c",
    ] = np.nan
    pred["spo2_deficit_per5"] = (100.0 - pred["spo2_min"]) / 5.0
    pred["gcs_deficit_from15"] = 15.0 - pred["gcs_min"]
    pred.drop(
        columns=[
            "hr_max", "map_min", "rr_max", "temp_min_c", "temp_max_c",
            "spo2_min", "gcs_min",
        ],
        inplace=True,
    )

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
        JOIN selected_ids s ON l.patientUnitStayID=s.stay_id
        WHERE l.labResultOffset BETWEEN 0 AND 1440
          AND l.labResultRevisedOffset<=1440
          AND l.labName IN ({names_sql})
    """).df()

    for model_col, spec in LAB_SPECS.items():
        x = lab.loc[
            lab["lab_name"].eq(spec["name"])
            & pd.to_numeric(lab["lab_type_id"], errors="coerce").eq(spec["type"])
        ].copy()
        x = x.loc[x["value"].gt(spec["lo"]) & x["value"].le(spec["hi"])]
        if spec["agg"] == "min":
            vals = x.groupby("stay_id")["value"].min()
        elif spec["agg"] == "max":
            vals = x.groupby("stay_id")["value"].max()
        elif spec["agg"] == "minmax":
            gg = x.groupby("stay_id")["value"].agg(["min", "max"])
            vals = pd.concat(
                [(gg["min"] - 140.0).abs(), (gg["max"] - 140.0).abs()], axis=1
            ).max(axis=1)
        else:
            raise RuntimeError(spec["agg"])

        if model_col in {"wbc_max_log2", "platelet_min_log2", "bun_max_log2", "glucose_max_log2"}:
            vals = np.log2(vals)
        elif model_col == "bicarbonate_min_per5":
            vals = vals / 5.0
        elif model_col == "sodium_max_abs_deviation_from140_per5":
            vals = vals / 5.0

        pred = pred.merge(
            vals.rename(model_col).reset_index(), on="stay_id", how="left", validate="1:1"
        )

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
        JOIN selected_ids s ON i.patientUnitStayID=s.stay_id
        WHERE i.infusionOffset BETWEEN 0 AND 1440
    """).df()
    if len(vaso):
        vaso["drug_class"] = vaso["drug_name"].map(classify_drug)
        vaso["positive_rate"] = [
            finite_positive(a) or finite_positive(b)
            for a, b in zip(vaso["drug_rate"], vaso["infusion_rate"])
        ]
        vaso_pos = set(
            vaso.loc[vaso["drug_class"].notna() & vaso["positive_rate"], "stay_id"].unique()
        )
    else:
        vaso_pos = set()
    pred["vasoactive_any_by_landmark"] = pred["stay_id"].isin(vaso_pos).astype(float)

    stage_map = selected.set_index("stay_id")[["kdigo_stage_2_indicator", "kdigo_stage_3_indicator"]]
    pred = pred.merge(stage_map.reset_index(), on="stay_id", how="left", validate="1:1")

    pred = pred[["unique_pid", "hstay_id", "stay_id", "hospital_id"] + PRIMARY_PREDICTORS].copy()
    pred = pred.sort_values(["unique_pid", "stay_id"], kind="mergesort").reset_index(drop=True)
    assert_unique(pred, "unique_pid", "raw final predictor matrix patient")
    assert_unique(pred, "stay_id", "raw final predictor matrix stay")
    assert_safe_columns(pred.columns, "raw final predictor matrix")

    for c in PRIMARY_PREDICTORS:
        pred[c] = pd.to_numeric(pred[c], errors="coerce")
        finite_vals = pred[c].dropna().to_numpy(float)
        if not np.isfinite(finite_vals).all():
            raise AssertionError(f"Infinite/non-finite observed values in {c}")

    # Non-imputable model terms must already be complete.
    non_fce = [c for c in PRIMARY_PREDICTORS if c not in FCE_TARGETS]
    non_fce_missing = pred[non_fce].isna().sum()
    if non_fce_missing.gt(0).any():
        raise RuntimeError(
            "Final cohort has missing non-FCE model predictors: "
            f"{non_fce_missing[non_fce_missing.gt(0)].to_dict()}"
        )

    for c in [
        "sex_female", "vasoactive_any_by_landmark",
        "kdigo_stage_2_indicator", "kdigo_stage_3_indicator",
    ]:
        vals = set(pred[c].dropna().unique().tolist())
        if not vals.issubset({0, 1, 0.0, 1.0}):
            raise AssertionError(f"Nonbinary values in {c}: {vals}")

    preimp_path = out_dir / "eicu_predictor_matrix_preimputation.parquet"
    pred.to_parquet(preimp_path, index=False)

    # ------------------------------------------------------------------
    # 8) Final missingness / sensitivity flags BEFORE imputation.
    # ------------------------------------------------------------------
    miss_rows = []
    for c in PRIMARY_PREDICTORS:
        nmiss = int(pred[c].isna().sum())
        pct = 100.0 * nmiss / len(pred)
        if pct > 75:
            category = "NOT_INTERPRETABLE"
        elif pct > 50:
            category = "SENSITIVITY_ONLY"
        elif pct > 25:
            category = "ENHANCED_REPORTING"
        else:
            category = "PRIMARY_INTERPRETABLE_RANGE"
        miss_rows.append({
            "model_term": c,
            "n": int(len(pred)),
            "missing_n": nmiss,
            "missing_pct": pct,
            "governance_category": category,
        })
    missingness = pd.DataFrame(miss_rows)
    write_csv(missingness, qc_dir / "FINAL_PREDICTOR_MISSINGNESS.csv")

    if (missingness["missing_pct"] > 75).any():
        raise RuntimeError("At least one required predictor exceeds the fixed >75% not-interpretable threshold")
    if (missingness["missing_pct"] > 50).any():
        interpretability = "SENSITIVITY_ONLY"
    else:
        interpretability = "INTERPRETABLE_WITH_LIMITATIONS"

    missing14 = pred[FCE_TARGETS].isna().sum(axis=1)
    missing17 = pred[NONRENAL_PREDICTORS].isna().sum(axis=1)
    sensitivity = pred[["unique_pid", "hstay_id", "stay_id", "hospital_id"]].copy()
    sensitivity["missing_imputable_predictor_count_14"] = missing14.astype(int)
    sensitivity["high_imputation_burden_ge7of14"] = missing14.ge(7)
    sensitivity["complete_case_17_nonrenal"] = missing17.eq(0)
    iso_map = selected.set_index("stay_id")[[
        "isolated_uf_after_chronic_exclusion", "uf_suppressed_fixed_cohort_state"
    ]]
    sensitivity = sensitivity.merge(iso_map.reset_index(), on="stay_id", how="left", validate="1:1")
    sensitivity_path = out_dir / "eicu_prespecified_sensitivity_flags_preoutcome.parquet"
    assert_safe_columns(sensitivity.columns, "sensitivity flags")
    sensitivity.to_parquet(sensitivity_path, index=False)

    # ------------------------------------------------------------------
    # 9) Verify and apply locked Harmonized model FCE artifacts.
    # ------------------------------------------------------------------
    g8_dir = project / "outputs" / "08_external_transport" / "mimic_cr_rrt_harmonized_refit"
    locked = pd.read_csv(g8_dir / "harmonized_cr_rrt_locked_coefficient_vector.csv")
    fce_pool = pd.read_csv(g8_dir / "fce_pooled_conditional_coefficients.csv")
    fce_init = pd.read_csv(g8_dir / "fce_initialization_values.csv")
    fce_spec = pd.read_csv(g8_dir / "fce_conditional_model_specification.csv")

    expected_terms = ["const"] + PRIMARY_PREDICTORS
    if locked["term"].tolist() != expected_terms or len(locked) != 20:
        raise AssertionError("Locked harmonized coefficient vector schema/order drift")
    locked_coef = dict(zip(locked["term"], locked["locked_coefficient"]))
    if not np.isfinite(np.asarray(list(locked_coef.values()), float)).all():
        raise AssertionError("Non-finite locked harmonized coefficient")

    if fce_spec["target_predictor"].tolist() != FCE_TARGETS:
        raise AssertionError("FCE specification target/update order drift")
    for _, r in fce_spec.iterrows():
        if r["conditional_model_family"] != "ordinary_linear_regression":
            raise AssertionError("FCE conditional family drift")
        if r["external_outcome_surrogate"] != "current_locked_model_predicted_probability":
            raise AssertionError("FCE outcome-surrogate rule drift")
        if r["initialization"] != "pooled_MIMIC_completed_development_marginal_mean":
            raise AssertionError("FCE initialization rule drift")
        if r["external_update_order"] != "canonical_model_column_order":
            raise AssertionError("FCE update-order rule drift")
        if not np.isclose(float(r["external_convergence_tolerance"]), FCE_TOL):
            raise AssertionError("FCE convergence tolerance drift")
        if int(r["external_max_full_cycles"]) != FCE_MAX_CYCLES:
            raise AssertionError("FCE max-cycle drift")
        if bool(r["external_stochastic_draws"]) or bool(r["external_pmm_donor_sampling"]):
            raise AssertionError("FCE external stochastic/PMM rule drift")
        target = r["target_predictor"]
        expected_covariates = [c for c in PRIMARY_PREDICTORS if c != target] + ["hospital_expire_flag"]
        observed_covariates = [x.strip() for x in str(r["training_covariates"]).split("|")]
        if observed_covariates != expected_covariates:
            raise AssertionError(f"FCE training-covariate order/content drift for {target}")

    if fce_init["target_predictor"].tolist() != FCE_TARGETS:
        raise AssertionError("FCE initialization artifact order drift")
    if not fce_init["imputations"].eq(30).all():
        raise AssertionError("FCE initialization does not reflect all 30 development imputations")
    if not fce_pool["imputations"].eq(30).all():
        raise AssertionError("FCE pooled coefficients do not reflect all 30 development imputations")

    ready, fce_cycles, _, _ = apply_fce(
        pred, locked_coef, fce_pool, fce_init
    )

    ready = ready[["unique_pid", "hstay_id", "stay_id", "hospital_id"] + PRIMARY_PREDICTORS].copy()
    ready_path = out_dir / "eicu_prediction_ready_matrix_preoutcome.parquet"
    assert_safe_columns(ready.columns, "prediction-ready matrix")
    ready.to_parquet(ready_path, index=False)

    # ------------------------------------------------------------------
    # 10) Cohort/site/renal/predictor QC.
    # ------------------------------------------------------------------
    stage_counts_final = pd.to_numeric(selected["eicu_cr_rrt_stage"], errors="coerce").value_counts().sort_index()
    flow = pd.DataFrame([
        {"metric": "fixed_site_eligible_hospitals", "value": EXPECTED_SITE_N},
        {"metric": "site_eligible_adult_plus24h_stays", "value": len(site_adult24)},
        {"metric": "site_eligible_hf_plus24h_stays", "value": len(cand)},
        {"metric": "definite_chronic_krt_stays_excluded", "value": int(renal["definite_chronic_krt"].sum())},
        {"metric": "cr_rrt_positive_qualifying_stays", "value": len(qual)},
        {"metric": "cr_rrt_positive_qualifying_patients", "value": qual["unique_pid"].nunique()},
        {"metric": "chronology_ambiguous_patients_excluded", "value": len(ambiguous_pids)},
        {"metric": "final_selected_patients", "value": len(selected)},
        {"metric": "final_contributing_hospitals", "value": selected["hospital_id"].nunique()},
        {"metric": "final_stage1", "value": int(stage_counts_final.get(1, 0))},
        {"metric": "final_stage2", "value": int(stage_counts_final.get(2, 0))},
        {"metric": "final_stage3", "value": int(stage_counts_final.get(3, 0))},
        {"metric": "final_complete_case_17_nonrenal", "value": int(sensitivity["complete_case_17_nonrenal"].sum())},
        {"metric": "final_high_imputation_burden_ge7of14", "value": int(sensitivity["high_imputation_burden_ge7of14"].sum())},
        {"metric": "final_isolated_uf_primary_cases", "value": int(sensitivity["isolated_uf_after_chronic_exclusion"].sum())},
    ])
    write_csv(flow, qc_dir / "FINAL_COHORT_FLOW.csv")

    unit_dist = (
        selected.groupby(["unit_type", "unit_stay_type"], dropna=False)
        .size().rename("patients").reset_index().sort_values("patients", ascending=False)
    )
    unit_dist["pct"] = 100.0 * unit_dist["patients"] / len(selected)
    write_csv(unit_dist, qc_dir / "FINAL_SELECTED_UNIT_DISTRIBUTION.csv")

    cohort_by_hospital = (
        selected.groupby("hospital_id", as_index=False)
        .agg(
            final_patients=("stay_id", "size"),
            stage1=("eicu_cr_rrt_stage", lambda x: int(pd.to_numeric(x, errors="coerce").eq(1).sum())),
            stage2=("eicu_cr_rrt_stage", lambda x: int(pd.to_numeric(x, errors="coerce").eq(2).sum())),
            stage3=("eicu_cr_rrt_stage", lambda x: int(pd.to_numeric(x, errors="coerce").eq(3).sum())),
        )
        .sort_values("hospital_id")
    )
    write_csv(cohort_by_hospital, qc_dir / "FINAL_COHORT_BY_HOSPITAL.csv")

    by_hosp_rows = []
    for hospital_id, g in pred.groupby("hospital_id", sort=True):
        for c in PRIMARY_PREDICTORS:
            by_hosp_rows.append({
                "hospital_id": int(hospital_id),
                "model_term": c,
                "patients": int(len(g)),
                "missing_n": int(g[c].isna().sum()),
                "missing_pct": float(100.0 * g[c].isna().mean()),
            })
    write_csv(
        pd.DataFrame(by_hosp_rows),
        qc_dir / "FINAL_PREDICTOR_MISSINGNESS_BY_HOSPITAL.csv",
    )

    # Do not persist uniquePid-level chronology QC; aggregate it to protect the
    # QC surface while the canonical cohort parquet already contains identifiers.
    chronology_dist = (
        episode_patient_qc.groupby(
            ["qualifying_hospital_admissions", "qualifying_hospital_admissions_in_earliest_year", "chronology_ambiguous"],
            dropna=False,
        ).size().rename("patients").reset_index().sort_values("patients", ascending=False)
    )
    write_csv(chronology_dist, qc_dir / "FINAL_EPISODE_MULTIPLICITY_DISTRIBUTION.csv")

    availability_out = pd.DataFrame({"hospital_id": site_ids})
    write_csv(availability_out, qc_dir / "APPLIED_AVAILABILITY_ELIGIBLE_HOSPITALS.csv")
    contributing_ids = sorted(pd.to_numeric(selected["hospital_id"], errors="raise").astype(int).unique().tolist())
    contributing = pd.DataFrame({"hospital_id": contributing_ids})
    write_csv(contributing, qc_dir / "FINAL_CONTRIBUTING_HOSPITALS.csv")
    if not set(contributing_ids).issubset(set(site_ids)):
        raise AssertionError("Final contributing hospital violates fixed 145-site rule")

    sens_summary = pd.DataFrame([{
        "final_primary_patients": int(len(sensitivity)),
        "complete_case_17_nonrenal_n": int(sensitivity["complete_case_17_nonrenal"].sum()),
        "complete_case_17_nonrenal_pct": float(100.0 * sensitivity["complete_case_17_nonrenal"].mean()),
        "high_imputation_burden_ge7of14_n": int(sensitivity["high_imputation_burden_ge7of14"].sum()),
        "high_imputation_burden_ge7of14_pct": float(100.0 * sensitivity["high_imputation_burden_ge7of14"].mean()),
        "isolated_uf_primary_n": int(sensitivity["isolated_uf_after_chronic_exclusion"].sum()),
        "primary_interpretability_category_preoutcome": interpretability,
    }])
    write_csv(sens_summary, qc_dir / "PRESPECIFIED_SENSITIVITY_SUMMARY.csv")

    # ------------------------------------------------------------------
    # 11) Deterministic identities and execution manifest.
    # ------------------------------------------------------------------
    identity = {
        "availability_eligible_hospital_set_sha256_sorted_newline": sha256_sorted_ints(site_ids),
        "final_contributing_hospital_set_sha256_sorted_newline": sha256_sorted_ints(contributing_ids),
        "final_uniquePid_set_sha256_sorted_newline": sha256_sorted_strings(selected["unique_pid"]),
        "final_hstay_id_set_sha256_sorted_newline": sha256_sorted_ints(selected["hstay_id"]),
        "final_stay_id_set_sha256_sorted_newline": sha256_sorted_ints(selected["stay_id"]),
    }
    identity_df = pd.DataFrame([{"identity": k, "sha256": v} for k, v in identity.items()])
    write_csv(identity_df, qc_dir / "FINAL_IDENTITY_HASHES.csv")

    raw_fp_df = pd.DataFrame([
        {
            "raw_file": name,
            "size_bytes": info["size_bytes"],
            "sha256": info["sha256"],
        }
        for name, info in sorted(raw_fingerprints.items())
    ])
    write_csv(raw_fp_df, qc_dir / "RAW_EICU_INPUT_FINGERPRINTS.csv")

    runtime_df = pd.DataFrame([{
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "duckdb": duckdb.__version__,
        "pyarrow": pyarrow.__version__,
    }])
    write_csv(runtime_df, qc_dir / "RUNTIME_ENVIRONMENT.csv")

    # Scientific/QC output hashes before writing the manifest itself.
    generated_files = sorted(
        [p for base in (out_dir, qc_dir) for p in base.rglob("*") if p.is_file()],
        key=lambda p: str(p),
    )
    generated_hashes = {relative(project, p): sha256_file(p) for p in generated_files}

    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)

    manifest = {
        "status": "PASS",
        "gate": "Final eICU pre-outcome cohort/predictor build",
        "analysis_scope": "preoutcome_cohort_construction",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_mimic_science_changed": False,
        "eicu_outcome_accessed": False,
        "eicu_discharge_status_or_location_accessed": False,
        "model_fit_or_recalibration_performed_in_eicu": False,
        "performance_metrics_computed": False,
        "outcome_attachment_authorized": False,
        "source_script": {
            "path": relative(project, script_path),
            "sha256": script_sha,
        },
        "harmonized_model_inputs": consumed_hashes,
        "raw_eicu_input_fingerprints": raw_fingerprints,
        "site_lock": {
            "fixed_site_list_path": relative(project, site_list_path),
            "fixed_site_list_sha256": EXPECTED_SITE_LIST_SHA,
            "availability_eligible_hospitals_n": len(site_ids),
            "availability_eligible_id_set_sha256_sorted_newline": identity[
                "availability_eligible_hospital_set_sha256_sorted_newline"
            ],
            "final_contributing_hospitals_n": len(contributing_ids),
            "final_contributing_id_set_sha256_sorted_newline": identity[
                "final_contributing_hospital_set_sha256_sorted_newline"
            ],
        },
        "cohort": {
            "adult_plus24_site_eligible_stays": int(len(site_adult24)),
            "hf_plus24_site_eligible_stays": int(len(cand)),
            "definite_chronic_krt_stays": int(renal["definite_chronic_krt"].sum()),
            "qualifying_cr_rrt_positive_stays": int(len(qual)),
            "qualifying_cr_rrt_positive_patients": int(qual["unique_pid"].nunique()),
            "chronology_ambiguous_patients_excluded": int(len(ambiguous_pids)),
            "final_selected_patients": int(len(selected)),
            "final_stage1": int(stage_counts_final.get(1, 0)),
            "final_stage2": int(stage_counts_final.get(2, 0)),
            "final_stage3": int(stage_counts_final.get(3, 0)),
            "membership_hashes": identity,
        },
        "predictor_schema": PRIMARY_PREDICTORS,
        "preimputation": {
            "matrix_path": relative(project, preimp_path),
            "matrix_sha256": sha256_file(preimp_path),
            "complete_case_17_nonrenal_n": int(sensitivity["complete_case_17_nonrenal"].sum()),
            "high_imputation_burden_ge7of14_n": int(sensitivity["high_imputation_burden_ge7of14"].sum()),
            "maximum_predictor_missing_pct": float(missingness["missing_pct"].max()),
            "interpretability_category": interpretability,
        },
        "fce": {
            "targets": FCE_TARGETS,
            "convergence_tolerance": FCE_TOL,
            "maximum_full_cycles": FCE_MAX_CYCLES,
            "cycles_to_global_convergence": int(fce_cycles["cycle"].max()),
            "final_max_abs_probability_delta": float(fce_cycles.iloc[-1]["max_abs_probability_delta"]),
            "observed_values_overwritten": False,
            "imputed_scale_violations": 0,
            "actual_eicu_outcome_used": False,
            "eicu_parameters_estimated": False,
        },
        "prediction_ready": {
            "matrix_path": relative(project, ready_path),
            "matrix_sha256": sha256_file(ready_path),
            "rows": int(len(ready)),
            "missing_model_inputs": int(ready[PRIMARY_PREDICTORS].isna().sum().sum()),
            "nonfinite_model_inputs": int((~np.isfinite(ready[PRIMARY_PREDICTORS].to_numpy(float))).sum()),
        },
        "generated_artifact_hashes": generated_hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "duckdb": duckdb.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "next_action": (
            "independent artifact-level pre-outcome audit; do not attach eICU outcome "
            "until final pre-outcome SHA lock passes"
        ),
    }

    manifest_path = qc_dir / "EICU_PREOUTCOME_BUILD_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)

    # Compact post-manifest execution receipt anchors the immutable manifest.
    receipt = {
        "status": "PASS",
        "build_manifest_path": relative(project, manifest_path),
        "build_manifest_sha256": manifest_sha,
        "source_script_sha256": script_sha,
        "eicu_outcome_accessed": False,
        "outcome_attachment_authorized": False,
    }
    receipt_path = qc_dir / "EICU_PREOUTCOME_EXECUTION_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    receipt_sha = sha256_file(receipt_path)

    con.close()

    print()
    print("=" * 92)
    print("FINAL eICU PRE-OUTCOME BUILD: PASS")
    print("=" * 92)
    print(f"Final selected patients/episodes : {len(selected):,}")
    print(f"Final contributing hospitals     : {len(contributing_ids):,}")
    print(
        "Final stages                     : "
        f"S1={int(stage_counts_final.get(1,0)):,} "
        f"S2={int(stage_counts_final.get(2,0)):,} "
        f"S3={int(stage_counts_final.get(3,0)):,}"
    )
    print(f"Chronology ambiguity exclusions  : {len(ambiguous_pids):,}")
    print(f"Max predictor missingness        : {float(missingness['missing_pct'].max()):.4f}%")
    print(f"FCE cycles to convergence        : {int(fce_cycles['cycle'].max())}")
    print(f"Final probability delta          : {float(fce_cycles.iloc[-1]['max_abs_probability_delta']):.3e}")
    print("Build manifest SHA-256           :", manifest_sha)
    print("Execution receipt SHA-256        :", receipt_sha)
    print("eICU outcome accessed            : NO")
    print("Verification scope              : data preparation and built-in integrity checks")
    print("Outcome attachment authorized    : NO")
    print("=" * 92)


if __name__ == "__main__":
    main()
