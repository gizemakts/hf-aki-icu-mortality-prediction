#!/usr/bin/env python3
"""Post-hoc supplementary internal validation of fixed harmonized MIMIC Cr/RRT.

Modes are deliberately gated: --preflight, --pilot 5 (or 10), then --full.
Bootstrap model development imports and uses the fixed Harmonized model MI/PMM and
logistic-regression functions.  Assessment imputation is trained only on the
bootstrap sample and never receives original-cohort test outcomes.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score


RUN_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
HARMONIZED_MODEL_SCRIPT = REPO / "scripts/modeling/harmonized/fit_harmonized_model.py"
QC_DIR = REPO / "quality_control/eicu_external_validation/mimic_cr_rrt_harmonized_refit"
MODEL_DIR = REPO / "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit"
DEV_PATH = MODEL_DIR / "harmonized_cr_rrt_development_matrix_with_outcome.parquet"
LOCKED_PATH = MODEL_DIR / "harmonized_cr_rrt_locked_coefficient_vector.csv"
POOLED_PATH = MODEL_DIR / "harmonized_cr_rrt_pooled_coefficients.csv"
BY_IMP_PATH = MODEL_DIR / "harmonized_cr_rrt_coefficients_by_imputation.csv"
APP_PRED_PATH = MODEL_DIR / "harmonized_cr_rrt_apparent_predictions.csv"
HARMONIZED_MODEL_MANIFEST_PATH = QC_DIR / "HARMONIZED_MODEL_MANIFEST.json"
CANONICAL_IV_PATH = REPO / "scripts/evaluation/mimic_iv/bootstrap_internal_validation.py"

EXPECTED_N = 2098
EXPECTED_EVENTS = 537
EXPECTED_PARAMETERS = 19
STRICT_TOL = 1e-12
N_IMPUTATIONS = 30
MICE_CYCLES = 10
PMM_DONORS = 5
DEVELOPMENT_SEED = 2026091701
BOOTSTRAP_INDEX_SEED = 2026101001
BOOTSTRAP_MI_SEED = 2026101002
ASSESSMENT_MI_SEED = 2026101003
FULL_REPLICATES = 500
PARALLEL_WORKERS = 4
OUTCOME = "hospital_expire_flag"
TITLE = "Post-hoc supplementary internal validation of the fixed harmonized MIMIC Cr/RRT model."
METRICS = ["auroc", "brier", "calibration_intercept", "calibration_slope", "observed_expected_ratio"]


def load_harmonized_model_module():
    spec = importlib.util.spec_from_file_location("fixed_harmonized_model_refit", HARMONIZED_MODEL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import fixed Harmonized model source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


G8 = load_harmonized_model_module()
PREDICTORS = list(G8.PRIMARY_PREDICTORS)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def protected_files() -> list[Path]:
    roots = [
        REPO / "outputs/04_modeling",
        REPO / "outputs/05_validation",
        MODEL_DIR,
        QC_DIR,
    ]
    files = [HARMONIZED_MODEL_SCRIPT]
    for root in roots:
        if root.is_dir():
            files.extend(p for p in root.rglob("*") if p.is_file())
    return sorted(set(files))


def protected_snapshot(apparent_predictions: Path = APP_PRED_PATH) -> dict[str, str]:
    paths = {*protected_files(), apparent_predictions}
    return {p.relative_to(REPO).as_posix(): sha256(p) for p in sorted(paths)}


def verify_fixed_hashes(apparent_predictions: Path = APP_PRED_PATH) -> dict[str, str]:
    """Check the development artifacts against the original fit's output record."""
    gate = json.loads(HARMONIZED_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise AssertionError("Harmonized development model did not complete successfully")
    expected = gate.get("generated_output_hashes", {})
    required = [DEV_PATH, LOCKED_PATH, POOLED_PATH, BY_IMP_PATH, apparent_predictions]
    checked = {}
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
        rel = str(path.relative_to(REPO)).replace("\\", "/")
        observed = sha256(path)
        reference = expected.get(rel)
        if reference is not None and reference != observed:
            raise AssertionError(f"Model development output changed: {rel}")
        checked[rel] = observed
    return checked


def load_and_validate_development() -> tuple[pd.DataFrame, dict]:
    gate = json.loads(HARMONIZED_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_cols = ["subject_id", "stay_id", OUTCOME] + PREDICTORS
    df = pd.read_parquet(DEV_PATH)
    if df.columns.tolist() != expected_cols:
        raise AssertionError(f"Development schema/order mismatch: {df.columns.tolist()}")
    if len(df) != EXPECTED_N or df.subject_id.nunique() != EXPECTED_N or df.stay_id.nunique() != EXPECTED_N:
        raise AssertionError("N, unique-subject, or unique-stay contract failed.")
    if int(df[OUTCOME].sum()) != EXPECTED_EVENTS or set(df[OUTCOME].unique()) != {0, 1}:
        raise AssertionError("Outcome/event contract failed.")
    if len(PREDICTORS) != EXPECTED_PARAMETERS:
        raise AssertionError("Predictor parameter count is not 19.")
    model = gate.get("modeling", {})
    expected_model = (N_IMPUTATIONS, MICE_CYCLES, PMM_DONORS, DEVELOPMENT_SEED)
    got_model = (model.get("number_of_imputations"), model.get("mice_cycles"), model.get("pmm_donors"), model.get("model_seed"))
    if got_model != expected_model:
        raise AssertionError(f"Fixed MI contract mismatch: {got_model} != {expected_model}")
    if model.get("predictors") != PREDICTORS or G8.OUTCOME != OUTCOME:
        raise AssertionError("Fixed predictor/order or endpoint mismatch.")
    for c in PREDICTORS:
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        if np.isinf(x).any():
            raise AssertionError(f"Infinite values in {c}.")
    for c in ("sex_female", "vasoactive_any_by_landmark", "kdigo_stage_2_indicator", "kdigo_stage_3_indicator"):
        vals = set(df[c].dropna().astype(float).unique())
        if not vals.issubset({0.0, 1.0}):
            raise AssertionError(f"Binary coding mismatch in {c}: {vals}")
    if ((df.kdigo_stage_2_indicator + df.kdigo_stage_3_indicator) > 1).any():
        raise AssertionError("KDIGO indicators are not mutually exclusive.")
    contract = {
        "n": len(df), "unique_subjects": int(df.subject_id.nunique()),
        "unique_stays": int(df.stay_id.nunique()), "events": int(df[OUTCOME].sum()),
        "endpoint": "database-native in-hospital mortality (hospital_expire_flag)",
        "time_zero": "ICU admission +24 h landmark",
        "episode": "first Cr/RRT-positive ICU episode per subject",
        "predictors": PREDICTORS,
        "non_intercept_parameters": len(PREDICTORS),
        "imputations": N_IMPUTATIONS, "mice_cycles": MICE_CYCLES,
        "pmm_donors": PMM_DONORS, "development_seed": DEVELOPMENT_SEED,
        "missing_by_predictor": {c: int(df[c].isna().sum()) for c in PREDICTORS},
    }
    return df, contract


def coefficient_comparison(reproduced: pd.DataFrame, reference: pd.DataFrame, reproduction_col: str, ref_col: str, check: str) -> list[dict]:
    left = reproduced[["term", reproduction_col]].rename(columns={reproduction_col: "_reproduced_value"})
    right = reference[["term", ref_col]].rename(columns={ref_col: "_reference_value"})
    q = left.merge(right, on="term", validate="1:1")
    rows = []
    for _, r in q.iterrows():
        diff = abs(float(r._reproduced_value) - float(r._reference_value))
        rows.append({"check": check, "term": r.term, "reproduced": float(r._reproduced_value), "fixed_reference": float(r._reference_value), "abs_diff": diff, "tolerance": STRICT_TOL, "status": "PASS" if diff <= STRICT_TOL else "FAIL"})
    return rows


def reproduction_baseline(
    df: pd.DataFrame, apparent_predictions: Path = APP_PRED_PATH,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    print("Baseline reproduction: exact Harmonized model MI30/MICE10/PMM5 ...", flush=True)
    fit_df = df[["subject_id", "stay_id", OUTCOME] + PREDICTORS].copy()
    imps = G8.mice_pmm_imputations(fit_df, PREDICTORS, OUTCOME, N_IMPUTATIONS, MICE_CYCLES, PMM_DONORS, DEVELOPMENT_SEED, verbose=False)
    fits, pred, by_imp = G8.fit_logistic_mi(imps, PREDICTORS, OUTCOME)
    pool = G8.rubin_pool(fits, "HARMONIZED_MIMIC_CR_RRT")
    rows = coefficient_comparison(pool, pd.read_csv(POOLED_PATH), "pooled_coef", "pooled_coef", "pooled_coefficient")
    locked_reproduction = pool[["term", "pooled_coef"]].rename(columns={"pooled_coef": "locked_coefficient"})
    rows += coefficient_comparison(locked_reproduction, pd.read_csv(LOCKED_PATH), "locked_coefficient", "locked_coefficient", "locked_coefficient")
    fixed_by_imp = pd.read_csv(BY_IMP_PATH)
    q = by_imp.merge(fixed_by_imp, on=["imputation", "term"], suffixes=("_reproduction", "_fixed"), validate="1:1")
    for _, r in q.iterrows():
        diff = abs(float(r.coefficient_reproduction) - float(r.coefficient_fixed))
        rows.append({"check": "coefficient_by_imputation", "term": f"MI{int(r.imputation):02d}:{r.term}", "reproduced": float(r.coefficient_reproduction), "fixed_reference": float(r.coefficient_fixed), "abs_diff": diff, "tolerance": STRICT_TOL, "status": "PASS" if diff <= STRICT_TOL else "FAIL"})
    p = pred.mean(axis=1)
    fixed_pred = pd.read_csv(apparent_predictions)
    if not df.subject_id.eq(fixed_pred.subject_id).all() or not df.stay_id.eq(fixed_pred.stay_id).all():
        raise AssertionError("Fixed apparent prediction identity/order mismatch.")
    pdiff = np.abs(p - fixed_pred.apparent_probability_mi_mean.to_numpy(float))
    rows.append({"check": "apparent_probability_mi_mean", "term": "all_rows_max_abs_diff", "reproduced": float(pdiff.max()), "fixed_reference": 0.0, "abs_diff": float(pdiff.max()), "tolerance": STRICT_TOL, "status": "PASS" if pdiff.max() <= STRICT_TOL else "FAIL"})
    qc = pd.DataFrame(rows)
    if not qc.status.eq("PASS").all() or len(fits) != N_IMPUTATIONS or not all(getattr(f, "converged", True) for f in fits):
        raise AssertionError("Strict baseline reproduction failed.")
    info = {"imputations": len(fits), "converged_fits": int(sum(bool(getattr(f, "converged", True)) for f in fits)), "max_abs_difference": float(qc.abs_diff.max()), "status": "PASS"}
    print(f"Baseline reproduction PASS; max abs difference={info['max_abs_difference']:.3e}", flush=True)
    return qc, info, p


def safe_logit(prob: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(np.asarray(prob, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def metrics(y, prob) -> dict[str, float]:
    y = np.asarray(y, float)
    prob = np.asarray(prob, float)
    lp = safe_logit(prob)
    int_fit = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit(maxiter=200, tol=1e-10)
    slope_fit = sm.GLM(y, sm.add_constant(lp, has_constant="add"), family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
    if not getattr(int_fit, "converged", True) or not getattr(slope_fit, "converged", True):
        raise RuntimeError("Calibration model did not converge.")
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
        "calibration_intercept": float(int_fit.params[0]),
        "calibration_slope": float(slope_fit.params[1]),
        "observed_expected_ratio": float(y.sum() / prob.sum()),
    }


def replicate_seed(master: int, rep: int) -> int:
    return int(np.random.SeedSequence([master, rep]).generate_state(1, dtype=np.uint64)[0])


def bootstrap_indices(rep: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_INDEX_SEED, rep]))
    return rng.integers(0, EXPECTED_N, EXPECTED_N)


def pmm_values(pred_obs, pred_mis, y_obs, rng, donors=5):
    pred_obs = np.asarray(pred_obs, float)
    pred_mis = np.asarray(pred_mis, float)
    y_obs = np.asarray(y_obs, float)
    out = np.empty(len(pred_mis), float)
    k = min(int(donors), len(y_obs))
    for j, value in enumerate(pred_mis):
        d = np.abs(pred_obs - value)
        near = np.argpartition(d, k - 1)[:k]
        out[j] = y_obs[int(rng.choice(near))]
    return out


def leakage_free_test_imputations(train_raw: pd.DataFrame, train_imps: list[pd.DataFrame], test_raw: pd.DataFrame, seed: int) -> list[pd.DataFrame]:
    """Impute test predictors using bootstrap-training data only, never test y.

    Each of the 30 exact Harmonized model training imputations supplies completed
    conditioning predictors. Conditional models contain predictors only because
    the original-cohort outcomes are unavailable to the assessment imputer.
    """
    original_train = train_raw[PREDICTORS].copy()
    original_test = test_raw[PREDICTORS].copy()
    train_masks = {c: original_train[c].isna().to_numpy() for c in PREDICTORS}
    test_masks = {c: original_test[c].isna().to_numpy() for c in PREDICTORS}
    incomplete = [c for c in PREDICTORS if test_masks[c].any()]
    children = np.random.SeedSequence(seed).spawn(N_IMPUTATIONS)
    outputs = []
    for mi, (train_imp, child) in enumerate(zip(train_imps, children), start=1):
        rng = np.random.default_rng(child)
        cur = original_test.copy()
        for c in incomplete:
            observed = ~train_masks[c]
            donor = original_train.loc[observed, c].to_numpy(float)
            if len(donor) < 20:
                raise RuntimeError(f"Too few bootstrap-training PMM donors for assessment {c}.")
            cur.loc[test_masks[c], c] = rng.choice(donor, size=int(test_masks[c].sum()), replace=True)
        for _ in range(MICE_CYCLES):
            for target in list(rng.permutation(incomplete)):
                observed = ~train_masks[target]
                other = [c for c in PREDICTORS if c != target]
                Xo = train_imp.loc[observed, other].to_numpy(float)
                Xo = np.column_stack([np.ones(len(Xo)), Xo])
                yo = original_train.loc[observed, target].to_numpy(float)
                mask = test_masks[target]
                Xm = cur.loc[mask, other].to_numpy(float)
                Xm = np.column_stack([np.ones(len(Xm)), Xm])
                beta = G8._draw_linear_beta(Xo, yo, rng)
                cur.loc[mask, target] = pmm_values(Xo @ beta, Xm @ beta, yo, rng, PMM_DONORS)
        if cur.isna().any().any() or not np.isfinite(cur.to_numpy(float)).all():
            raise AssertionError(f"Assessment imputation {mi} incomplete/non-finite.")
        outputs.append(cur)
    return outputs


def run_replicate(rep: int, df: pd.DataFrame) -> dict:
    idx = bootstrap_indices(rep)
    train = df.iloc[idx].reset_index(drop=True)
    if train[OUTCOME].nunique() != 2:
        raise RuntimeError("Bootstrap sample contains one outcome class.")
    train_seed = replicate_seed(BOOTSTRAP_MI_SEED, rep)
    assess_seed = replicate_seed(ASSESSMENT_MI_SEED, rep)
    train_imps = G8.mice_pmm_imputations(train, PREDICTORS, OUTCOME, N_IMPUTATIONS, MICE_CYCLES, PMM_DONORS, train_seed, verbose=False)
    fits, train_pred_by_mi, _ = G8.fit_logistic_mi(train_imps, PREDICTORS, OUTCOME)
    pool = G8.rubin_pool(fits, f"BOOTSTRAP_{rep:04d}")
    test_imps = leakage_free_test_imputations(train, train_imps, df, assess_seed)
    test_pred = np.zeros((len(df), N_IMPUTATIONS), float)
    for j, (fit, imp) in enumerate(zip(fits, test_imps)):
        X = sm.add_constant(imp[PREDICTORS].astype(float), has_constant="add")
        test_pred[:, j] = np.asarray(fit.predict(X), float)
    app = metrics(train[OUTCOME], train_pred_by_mi.mean(axis=1))
    test = metrics(df[OUTCOME], test_pred.mean(axis=1))
    counts = np.bincount(idx, minlength=EXPECTED_N)
    row = {
        "replicate": rep, "status": "SUCCESS", "error_type": "", "error_message": "",
        "bootstrap_index_seed": BOOTSTRAP_INDEX_SEED,
        "bootstrap_mi_seed": f"u64:{train_seed}",
        "assessment_mi_seed": f"u64:{assess_seed}", "bootstrap_rows": len(idx),
        "bootstrap_unique_subjects": int((counts > 0).sum()),
        "bootstrap_omitted_subjects": int((counts == 0).sum()),
        "bootstrap_duplicate_draws": int(len(idx) - (counts > 0).sum()),
        "bootstrap_max_multiplicity": int(counts.max()),
        "bootstrap_events": int(train[OUTCOME].sum()),
        "converged_model_fits": int(sum(bool(getattr(f, "converged", True)) for f in fits)),
        "pooled_coefficients": int(len(pool)),
        "pooled_coefficient_l2_norm": float(np.linalg.norm(pool.pooled_coef.to_numpy(float))),
    }
    for metric in METRICS:
        row[f"bootstrap_apparent_{metric}"] = app[metric]
        row[f"test_original_{metric}"] = test[metric]
        row[f"optimism_{metric}"] = app[metric] - test[metric]
    return row


def progress_frame() -> pd.DataFrame:
    path = RUN_DIR / "BOOTSTRAP_REPLICATES.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


_WORKER_DF = None


def _worker_init(df: pd.DataFrame) -> None:
    global _WORKER_DF
    _WORKER_DF = df


def _worker_run(rep: int) -> dict:
    if _WORKER_DF is None:
        raise RuntimeError("Worker development frame was not initialized.")
    started = time.time()
    row = run_replicate(rep, _WORKER_DF)
    row["elapsed_seconds"] = time.time() - started
    return row


def execute_replicates(target: int, df: pd.DataFrame) -> pd.DataFrame:
    path = RUN_DIR / "BOOTSTRAP_REPLICATES.csv"
    out = progress_frame()
    if not out.empty and (out.status != "SUCCESS").any():
        raise RuntimeError("A prior failed replicate exists; fail-closed policy requires review, not automatic skipping/retry.")
    done = set(out.replicate.astype(int)) if not out.empty else set()
    if any(r < 1 or r > FULL_REPLICATES for r in done):
        raise AssertionError("Invalid replicate IDs in checkpoint.")
    print(f"Checkpoint: {len(done)} successful replicates; target={target}.", flush=True)
    session = time.time()
    initial = len(done)
    remaining_replicates = [rep for rep in range(1, target + 1) if rep not in done]
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=min(PARALLEL_WORKERS, max(len(remaining_replicates), 1)),
        initializer=_worker_init,
        initargs=(df,),
    )
    futures = {executor.submit(_worker_run, rep): rep for rep in remaining_replicates}
    try:
        for future in concurrent.futures.as_completed(futures):
            rep = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"replicate": rep, "status": "FAILURE", "error_type": type(exc).__name__, "error_message": str(exc)}
                out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
                out = out.drop_duplicates("replicate", keep="last").sort_values("replicate")
                atomic_csv(out, path)
                (RUN_DIR / f"FAILED_REPLICATE_{rep:04d}.txt").write_text(traceback.format_exc(), encoding="utf-8")
                for f in futures:
                    f.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise RuntimeError(f"Replicate {rep} failed; analysis stopped prospectively without discarding it.") from exc
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
            out = out.drop_duplicates("replicate", keep="last").sort_values("replicate")
            atomic_csv(out, path)
            complete = int((out.status == "SUCCESS").sum())
            avg = (time.time() - session) / max(complete - initial, 1)
            print(f"replicate {rep:03d}: SUCCESS | {row['elapsed_seconds']:.1f}s | {complete}/{target} | ETA {(target-complete)*avg*PARALLEL_WORKERS/3600:.2f} worker-h", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    valid = out.loc[(out.status == "SUCCESS") & (out.replicate <= target)].copy()
    if len(valid) != target or set(valid.replicate.astype(int)) != set(range(1, target + 1)):
        raise AssertionError(f"Completeness failure: {len(valid)}/{target} valid.")
    return out.sort_values("replicate").reset_index(drop=True)


def summarize(original_prob: np.ndarray, df: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    original = metrics(df[OUTCOME], original_prob)
    valid = boot.loc[(boot.status == "SUCCESS") & boot.replicate.between(1, FULL_REPLICATES)]
    if len(valid) != FULL_REPLICATES:
        raise AssertionError("Cannot summarize: 500 valid replicates are required.")
    rows = []
    for metric in METRICS:
        optimism = valid[f"optimism_{metric}"].to_numpy(float)
        mean_opt = float(optimism.mean())
        rows.append({
            "analysis": TITLE.rstrip("."), "metric": metric,
            "original_apparent": original[metric], "mean_optimism_apparent_minus_test": mean_opt,
            "optimism_corrected": original[metric] - mean_opt,
            "valid_replicates": len(valid), "failed_replicates": 0,
            "correction_formula": "original apparent - mean(bootstrap apparent - bootstrap test)",
            "confidence_interval": "NA",
        })
    summary = pd.DataFrame(rows)
    atomic_csv(summary, RUN_DIR / "INTERNAL_VALIDATION_SUMMARY.csv")
    return summary


def output_hashes() -> dict[str, str]:
    names = ["harmonized_internal_validation.py", "BOOTSTRAP_REPLICATES.csv", "INTERNAL_VALIDATION_SUMMARY.csv"]
    return {name: sha256(RUN_DIR / name) for name in names if (RUN_DIR / name).is_file()}


def manifest(mode: str, status: str, fixed_hashes: dict, contract: dict, reproduction: dict, target: int, completed: int, protected_before: dict, protected_after: dict) -> dict:
    return {
        "analysis_label": TITLE, "status": status, "mode": mode, "updated_utc": utcnow(),
        "repository_root": str(REPO), "run_directory": str(RUN_DIR),
        "scientific_scope": {"post_hoc_supplementary": True, "canonical_model_changed": False, "eicu_data_read": False, "eicu_outcome_accessed": False},
        "contract": contract,
        "rng": {"generator": "numpy.random.default_rng / PCG64", "development_seed": DEVELOPMENT_SEED, "bootstrap_index_master_seed": BOOTSTRAP_INDEX_SEED, "bootstrap_training_mi_master_seed": BOOTSTRAP_MI_SEED, "assessment_mi_master_seed": ASSESSMENT_MI_SEED, "replicate_derivation": "SeedSequence([master_seed, replicate])", "parallel_workers": PARALLEL_WORKERS, "parallel_invariance": "replicate-specific deterministic streams; scheduling order cannot change a replicate"},
        "method": {"bootstrap_protocol": "patient-level resampling with replacement; apparent minus original-cohort test optimism", "nested_training_imputation": "exact fixed Harmonized model MI30/MICE10/PMM5, outcome included in bootstrap development imputation", "assessment_imputation": "30 training-derived predictor-only PMM chains; original test outcome unavailable to imputer", "prediction_combination": "arithmetic mean of probabilities across 30 imputation-specific models", "calibration_intercept": "binomial GLM intercept with logit(prediction) offset fixed at slope 1", "calibration_slope": "binomial GLM of outcome on logit(prediction) with free intercept", "oe": "sum(observed)/sum(predicted); corrected on raw O/E scale per requested common formula", "failure_policy": "fail immediately, record failed replicate, do not skip or replace"},
        "target_replicates": target, "valid_replicates": completed, "baseline_reproduction": reproduction,
        "fixed_input_hashes_verified": fixed_hashes,
        "canonical_internal_validation_source": {"path": str(CANONICAL_IV_PATH.relative_to(REPO)).replace("\\", "/"), "sha256": sha256(CANONICAL_IV_PATH), "reused_conventions": ["patient bootstrap", "probability averaging across MI", "AUROC", "Brier", "calibration intercept/slope", "O/E", "apparent-minus-test optimism"], "intentional_safety_difference": "canonical test imputation exposes test outcome; this supplementary implementation does not"},
        "software_versions": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__, "scikit_learn": sklearn.__version__},
        "protected_files_unchanged": protected_before == protected_after,
        "protected_hashes_before": protected_before, "protected_hashes_after": protected_after,
        "output_hashes_excluding_manifest": output_hashes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apparent-predictions", type=Path, default=APP_PRED_PATH,
                        help="Apparent prediction table within the study project.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--pilot", type=int, choices=[5, 10])
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    apparent_predictions = (REPO / args.apparent_predictions).resolve()
    existing_manifest_path = RUN_DIR / "RUN_MANIFEST.json"
    if existing_manifest_path.is_file():
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("status") == "COMPLETE":
            raise FileExistsError(
                "This timestamped run is already COMPLETE and sealed; create a new run directory rather than overwriting it."
            )
    mode = "preflight" if args.preflight else (f"pilot-{args.pilot}" if args.pilot else "full")
    target = 0 if args.preflight else (args.pilot if args.pilot else FULL_REPLICATES)
    print(TITLE, flush=True)
    fixed_hashes = verify_fixed_hashes(apparent_predictions)
    df, contract = load_and_validate_development()
    protected_before = protected_snapshot(apparent_predictions)
    _, reproduction, original_prob = reproduction_baseline(df, apparent_predictions)
    boot = progress_frame()
    if target:
        boot = execute_replicates(target, df)
    summary = None
    if args.full:
        summary = summarize(original_prob, df, boot)
    protected_after = protected_snapshot(apparent_predictions)
    if protected_before != protected_after:
        raise AssertionError("Protected fixed files changed during execution.")
    completed = int(((boot.status == "SUCCESS") & (boot.replicate <= target)).sum()) if target and not boot.empty else 0
    status = "COMPLETE" if args.full and completed == FULL_REPLICATES else ("PASS" if completed == target else "FAIL")
    man = manifest(mode, status, fixed_hashes, contract, reproduction, target, completed, protected_before, protected_after)
    atomic_json(man, RUN_DIR / "RUN_MANIFEST.json")
    print(f"{mode}: {status}; valid replicates={completed}/{target}", flush=True)
    if summary is not None:
        print(summary[["metric", "original_apparent", "mean_optimism_apparent_minus_test", "optimism_corrected"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
