"""Shared, isolated safeguards for the JICM post hoc analyses."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

WORKSPACE = Path(__file__).resolve().parents[0]
REPO = WORKSPACE.parents[2]
RESULTS = WORKSPACE / "results"
CHECKPOINTS = WORKSPACE / "checkpoints"
LOGS = WORKSPACE / "logs"

PRIMARY_PREDICTORS = [
    "age_per10y", "sex_female", "heart_rate_max_per10", "map_min_per10",
    "resp_rate_max_per5", "temperature_max_abs_deviation_from_37c",
    "spo2_deficit_per5", "gcs_deficit_from15", "hemoglobin_min_gdl",
    "wbc_max_log2", "platelet_min_log2", "bun_max_log2",
    "bicarbonate_min_per5", "sodium_max_abs_deviation_from140_per5",
    "potassium_max_mmol_l", "glucose_max_log2", "vasoactive_any_by_landmark",
    "kdigo_stage_2_indicator", "kdigo_stage_3_indicator",
]
REDUCED_PREDICTORS = PRIMARY_PREDICTORS[:-2]
OUTCOME = "hospital_expire_flag"

# High-value anchors verified during discovery. Specialized lock manifests remain
# the full authority; this list is the minimum preflight protection set.
PROTECTED_HASHES = {
    "data/raw/mimic-iv-3.1/hosp/patients.csv.gz": "4c7507de7ecad8c5ba7647ea4d26018e88ff6672c5fc22ab59c2a5697d687cdd",
    "outputs/01_cohort/primary_cohort_preoutcome.parquet": "b627bc76a9adf759a60fce991b33f653dbaddb2db80729055b4ff703d79b8c40",
    "outputs/01_cohort/primary_cohort_with_outcome.parquet": "0d11ccab56a220f31eecd705e1f7d9a58abd8221f67bbf41aabd2168b23ff9c1",
    "outputs/03_predictors/primary_predictor_matrix.parquet": "15b3223ca3358c68d204c12ac3eb00a00d29aa12af93546fd867f1f075c471e1",
    "outputs/03_predictors/parameter_budget.csv": "4fadb9dd7b292cfbf4beda3e28bb5b336ad0e1670590177648bfdbcef7996047",
    "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_development_matrix_with_outcome.parquet": "f93967a2f7a523e3b65464764cd232065956a7afbfec199c324ed709ebd4013b",
    "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_locked_coefficient_vector.csv": "3773d8ce03655e014e0e0ea578ba85888470ae248070552b089fd94c27afb513",
    "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/fce_pooled_conditional_coefficients.csv": "40aba6b6158b1e4f534bb4dc68d3cc7e0d802d099935a37367c60c75423045d2",
    "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/fce_initialization_values.csv": "544ada93e615da32f44ff083349ae6fe647e955d9b1e5c1ca7c53cea6343ada5",
    "outputs/08_external_transport/eicu_final_preoutcome/eicu_predictor_matrix_preimputation.parquet": "43faf1beced67ea3e6cdd4ec3733160d9fd61b14932aba885fd7998aa14a168e",
    "outputs/08_external_transport/eicu_final_preoutcome/eicu_prediction_ready_matrix_preoutcome.parquet": "9269e4c8eb858ea3a4e7c559d533827ea72788e44d30060a5502433b55591e66",
    "outputs/08_external_transport/eicu_outcome_attachment/eicu_prediction_ready_with_outcome_availability.parquet": "7c7c23d5ec740c495723cb0444b36a328920db6417e38c4ba60d6a0986e6f1dd",
    "outputs/08_external_transport/eicu_outcome_attachment/eicu_external_validation_evaluable.parquet": "3971f51cd11e83ad85b38fcd7a4cf7e3e5943aca52e668911f61783b08052e8a",
    "outputs/08_external_transport/eicu_outcome_attachment/eicu_outcome_unavailable_rows.parquet": "a8e8b616810187735d6c16ae692882d283ffc1261d09e14077e42122736eb1a4",
    "outputs/08_external_transport/eicu_locked_external_validation/EICU_LOCKED_PRIMARY_PERFORMANCE.json": "96a0d04c7cf2ccdf3747e94837e8206fa59928d5b90198c4b5561fdfe7d600a4",
    "outputs/04_modeling/apparent_predictions.csv": "c9313fd1566d1b0f1e3ba80372df6fd7c2a7800dc96d08ecaa7081c99205b72a",
    "outputs/04_modeling/apparent_performance.csv": "059d05f5204d67b1f817a0f2211047281c0b02e3e5112d47a55f1f4da1ff9b0f",
    "outputs/04_modeling/primary_pooled_coefficients.csv": "25535764ab82f8256c60931812bf163f912b17d7e2d019735ef575e8f0ce8d35",
    "outputs/04_modeling/model_fit_manifest.json": "19af50b63f97b42f240fe0555f3be8ae76f0fbe13dcd29672289bf5ab5f81960",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_protected(paths: list[str] | None = None) -> dict[str, str]:
    chosen = paths or list(PROTECTED_HASHES)
    observed: dict[str, str] = {}
    for rel in chosen:
        path = REPO / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        got = sha256(path)
        expected = PROTECTED_HASHES[rel]
        if got != expected:
            raise RuntimeError(f"Protected hash mismatch: {rel}; expected {expected}, observed {got}")
        observed[rel] = got
    return observed


def atomic_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def execution_identity(payload: dict, modules: list[Path] | None = None) -> dict:
    """Identity of every implementation/protocol surface driving a run."""
    default_modules=[Path(sys.argv[0]).resolve(),Path(__file__).resolve(),WORKSPACE/"scripts/mi_validation.py",
                     WORKSPACE/"protocol/POSTHOC_ANALYSIS_PLAN.md",WORKSPACE/"protocol/PLAN_LOCK.json",
                     WORKSPACE/"protocol/AMENDMENT_2026-10-29_INDEPENDENT_AUDIT_CORRECTIONS.md",
                     WORKSPACE/"config/INPUT_PATHS.json"]
    selected=modules or default_modules
    missing=[str(p) for p in selected if not p.is_file()]
    if missing: raise FileNotFoundError(f"Execution-identity files missing: {missing}")
    def label(path: Path) -> str:
        try: return str(path.relative_to(REPO))
        except ValueError: return str(path)
    return {"files":{label(p):sha256(p) for p in selected},"environment":environment(),"payload":payload}


def verify_recorded_execution_identity(identity: dict) -> None:
    """Reject a receipt if any bound source/protocol/config or dependency changed."""
    if not isinstance(identity,dict) or not isinstance(identity.get("files"),dict):
        raise RuntimeError("Receipt execution identity is absent or malformed")
    for label,expected in identity["files"].items():
        path=Path(label)
        if not path.is_absolute(): path=REPO/path
        if not path.is_file() or sha256(path)!=expected:
            raise RuntimeError(f"Receipt execution-identity mismatch: {label}")
    if identity.get("environment")!=environment():
        raise RuntimeError("Receipt dependency environment differs from current environment")


def ensure_checkpoint_contract(name: str, payload: dict, checkpoint_paths: list[Path] | None = None,
                               modules: list[Path] | None = None) -> Path:
    """Bind resumable state to all code, protocol, inputs and configuration."""
    contract={"name":name,"execution_identity":execution_identity(payload,modules)}
    path=CHECKPOINTS/f"{name}_contract.json"
    if path.exists():
        observed=json.loads(path.read_text(encoding="utf-8"))
        if observed!=contract: raise RuntimeError(f"Checkpoint contract mismatch: {path}; preserve it and start a separately named governed run")
    else:
        if checkpoint_paths and any(p.exists() for p in checkpoint_paths):
            raise RuntimeError("Checkpoint data exist without their governing contract; refuse unsafe resume")
        atomic_json(contract,path)
    return path


def validate_checkpoint(frame: pd.DataFrame, expected_columns: list[str], target_replicates: int) -> set[int]:
    if frame.empty:return set()
    if frame.columns.tolist()!=expected_columns:raise RuntimeError("Checkpoint schema/order mismatch")
    ids=pd.to_numeric(frame["replicate"],errors="raise").astype(int)
    if ids.duplicated().any() or not ids.between(1,target_replicates).all():raise RuntimeError("Checkpoint replicate IDs invalid/duplicated")
    if not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(float)).all():raise RuntimeError("Checkpoint has non-finite numeric values")
    return set(ids.tolist())


def snapshot(paths: list[str]) -> dict[str,str]:
    result={}
    for rel in paths:
        p=REPO/rel
        if not p.is_file():raise FileNotFoundError(p)
        result[rel]=sha256(p)
    return result


def verify_postflight(before: dict[str,str]) -> dict:
    after=snapshot(list(before)); changed={k:{"before":v,"after":after[k]} for k,v in before.items() if after[k]!=v}
    if changed:raise RuntimeError(f"Fixed artifacts changed during execution: {changed}")
    return {"performed":True,"unchanged":True,"checked":len(before),"hashes_after":after}


def environment() -> dict[str, str]:
    import scipy, sklearn, statsmodels
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__, "statsmodels": statsmodels.__version__,
    }


def manifest(analysis: str, status: str, parameters: dict, inputs: dict[str, str], outputs: list[Path], postflight: dict, disclosure: dict) -> dict:
    if not postflight.get("performed") or not postflight.get("unchanged"):raise RuntimeError("Cannot claim completion without successful fixed-artifact postflight")
    if disclosure.get("status")!="PASS":raise RuntimeError("Cannot claim completion without disclosure PASS")
    return {
        "analysis": analysis, "post_hoc_supplementary": True, "status": status,
        "updated_utc": utcnow(), "repository_root": str(REPO),
        "code": {str(Path(__file__).relative_to(REPO)): sha256(Path(__file__)),
                 str(Path(sys.argv[0]).resolve().relative_to(REPO)): sha256(Path(sys.argv[0]).resolve())},
        "parameters": parameters, "input_hashes": inputs,
        "output_hashes": {str(p.relative_to(REPO)): sha256(p) for p in outputs if p.is_file()},
        "environment": environment(), "postflight_integrity":postflight,
        "fixed_artifacts_modified": not postflight["unchanged"],"disclosure_review":disclosure,
    }


def disclosure_check(outputs: list[Path], small_cell_threshold: int = 11) -> dict:
    prohibited={"subject_id","stay_id","hadm_id","unique_pid","hstay_id","hospital_id","patientunitstayid"}
    reviewed=[]
    for p in outputs:
        if p.suffix.lower()==".csv" and p.is_file():
            cols=set(pd.read_csv(p,nrows=0).columns.str.lower())
            bad=sorted(cols&prohibited)
            if bad:raise RuntimeError(f"Disclosure failure, identifier columns in aggregate output {p}: {bad}")
        reviewed.append(str(p.relative_to(REPO)))
    return {"status":"PASS","aggregate_only":True,"identifier_columns_prohibited":sorted(prohibited),
            "small_cell_threshold":small_cell_threshold,"files_reviewed":reviewed}


def safe_logit(prob: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(np.asarray(prob, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def scalar_metrics(y, prob) -> dict[str, float]:
    import statsmodels.api as sm

    y = np.asarray(y, float); p = np.asarray(prob, float)
    if len(y) != len(p) or set(np.unique(y)) != {0.0, 1.0}:
        raise ValueError("Metrics require aligned binary outcomes with both classes")
    eta = safe_logit(p)
    intercept = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=eta).fit(maxiter=200, tol=1e-10)
    slope = sm.GLM(y, sm.add_constant(eta, has_constant="add"), family=sm.families.Binomial()).fit(maxiter=200, tol=1e-10)
    return {
        "auroc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)),
        "calibration_intercept": float(intercept.params[0]),
        "calibration_slope": float(slope.params[1]),
        "observed_over_expected": float(y.sum() / p.sum()),
    }


def load_locked_coefficients() -> dict[str, float]:
    p = REPO / "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_locked_coefficient_vector.csv"
    frame = pd.read_csv(p)
    if frame["term"].tolist() != ["const"] + PRIMARY_PREDICTORS:
        raise RuntimeError("Locked coefficient order mismatch")
    return dict(zip(frame.term, frame.locked_coefficient.astype(float)))


def locked_probability(frame: pd.DataFrame, coefficients: dict[str, float] | None = None) -> np.ndarray:
    coef = coefficients or load_locked_coefficients()
    x = frame[PRIMARY_PREDICTORS].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite locked model inputs")
    eta = float(coef["const"]) + x @ np.array([coef[c] for c in PRIMARY_PREDICTORS])
    return 1.0 / (1.0 + np.exp(-eta))


def external_baseline_concordance(frame: pd.DataFrame, probability: np.ndarray, tolerance: float = 5e-12) -> dict:
    """Stop unless reproduced locked eICU identity and scalar results are authoritative."""
    reference_path=REPO/"outputs/08_external_transport/eicu_locked_external_validation/EICU_LOCKED_PRIMARY_PERFORMANCE.json"
    reference=json.loads(reference_path.read_text(encoding="utf-8"))
    evaluable=frame.loc[frame["outcome_evaluable"].eq(1)].copy()
    if len(frame)!=1918 or len(evaluable)!=1897:raise RuntimeError("eICU target/evaluable count discordance")
    if int(evaluable[OUTCOME].sum())!=433 or evaluable["hospital_id"].nunique()!=123:raise RuntimeError("eICU event/hospital discordance")
    p=np.asarray(probability,float)
    if len(p)==len(frame):p=p[frame["outcome_evaluable"].eq(1).to_numpy()]
    if len(p)!=1897:raise RuntimeError("Locked probability identity length mismatch")
    observed=scalar_metrics(evaluable[OUTCOME].to_numpy(float),p)
    expected={k:float(v["point"]) for k,v in reference["metrics"].items()}
    delta={k:abs(observed[k]-expected[k]) for k in expected}
    if any(v>tolerance for v in delta.values()):raise RuntimeError(f"Locked eICU scalar reproduction discordance: {delta}")
    if abs(float(p.mean())-float(reference["mean_locked_predicted_risk"]))>tolerance:raise RuntimeError("Locked mean risk discordance")
    return {"status":"PASS","target_n":1918,"evaluable_n":1897,"events":433,"hospitals":123,
            "tolerance":tolerance,"metrics_observed":observed,"metrics_expected":expected,"absolute_differences":delta,
            "reference_sha256":sha256(reference_path)}


def hospital_cluster_bootstrap(y, p_a, hospitals, replicates: int, seed: int, p_b=None, checkpoint: Path | None = None) -> pd.DataFrame:
    y = np.asarray(y, float); p_a = np.asarray(p_a, float); hospitals = np.asarray(hospitals)
    p_b = None if p_b is None else np.asarray(p_b, float)
    unique = np.unique(hospitals)
    metric_names=["auroc","brier","calibration_intercept","calibration_slope","observed_over_expected"]
    expected=["replicate"]+[f"a_{k}" for k in metric_names]
    if p_b is not None:expected += [f"b_{k}" for k in metric_names]+[f"difference_{k}" for k in metric_names]
    rows = pd.read_csv(checkpoint) if checkpoint and checkpoint.exists() else pd.DataFrame(columns=expected)
    done = validate_checkpoint(rows,expected,replicates)
    for rep in range(1, replicates + 1):
        if rep in done: continue
        rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
        draw = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(hospitals == h) for h in draw])
        if np.unique(y[idx]).size != 2:
            raise RuntimeError(f"Hospital bootstrap replicate {rep} has one outcome class")
        row = {"replicate": rep, **{f"a_{k}": v for k, v in scalar_metrics(y[idx], p_a[idx]).items()}}
        if p_b is not None:
            mb = scalar_metrics(y[idx], p_b[idx])
            row.update({f"b_{k}": v for k, v in mb.items()})
            row.update({f"difference_{k}": row[f"b_{k}"] - row[f"a_{k}"] for k in mb})
        addition=pd.DataFrame([row],columns=expected)
        rows = addition if rows.empty else pd.concat([rows,addition],ignore_index=True)
        rows=rows[expected].sort_values("replicate")
        if checkpoint: atomic_csv(rows, checkpoint)
    validate_checkpoint(rows,expected,replicates)
    if len(rows) != replicates or set(rows.replicate.astype(int)) != set(range(1, replicates + 1)):
        raise RuntimeError("Bootstrap checkpoint is incomplete or inconsistent")
    return rows.reset_index(drop=True)


def pmsampsize_binary(rsquared: float, parameters: int, prevalence: float, shrinkage: float, available_n: int) -> dict:
    r2a, p, prev, s = float(rsquared), int(parameters), float(prevalence), float(shrinkage)
    if not (0 < r2a < 1 and 0 < prev < 1 and 0 < s < 1): raise ValueError("Invalid Riley inputs")
    n1 = math.ceil(p / ((s - 1.0) * math.log(1.0 - r2a / s)))
    e1 = n1 * prev
    ln_null = e1 * math.log(e1 / n1) + (n1 - e1) * math.log(1 - e1 / n1)
    max_r2 = 1 - math.exp(2 * ln_null / n1)
    nag = r2a / max_r2
    s_small = r2a / (r2a + 0.05 * max_r2)
    n2 = math.ceil(p / ((s_small - 1.0) * math.log(1.0 - r2a / s_small)))
    n3 = math.ceil((1.96 / 0.05) ** 2 * prev * (1 - prev))
    required = max(n1, n2, n3)
    return {"parameters": p, "prevalence": prev, "anticipated_cox_snell_r2": r2a,
            "target_shrinkage": s, "criterion1_n_shrinkage": n1,
            "criterion2_n_r2_difference": n2, "criterion3_n_risk_precision": n3,
            "required_n": required, "available_n": available_n,
            "margin_n": available_n-required, "passes": available_n >= required,
            "max_cox_snell_r2": max_r2, "nagelkerke_r2": nag,
            "required_epp": required*prev/p}
