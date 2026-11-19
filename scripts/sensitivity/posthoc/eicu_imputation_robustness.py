#!/usr/bin/env python3
"""Post hoc eICU imputation robustness; dry-run unless --execute is supplied."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from common import (CHECKPOINTS, PRIMARY_PREDICTORS, REPO, RESULTS, atomic_csv,
                    atomic_json, disclosure_check, ensure_checkpoint_contract, external_baseline_concordance,
                    hospital_cluster_bootstrap, load_locked_coefficients, locked_probability, manifest,
                    scalar_metrics, sha256, snapshot, verify_postflight, verify_protected,
                    verify_recorded_execution_identity)

FCE_TARGETS = PRIMARY_PREDICTORS[2:16]
TOL = 1e-6
MAX_CYCLES = 50
MIMIC_DEVELOPMENT_PREVALENCE = 537 / 2098
KEYS = ["unique_pid", "hstay_id", "stay_id", "hospital_id"]
BASELINE_RECEIPT = CHECKPOINTS / "EXTERNAL_BASELINE_CONCORDANCE.json"
BOUNDS = {
    "heart_rate_max_per10": (0.0, 30.0, False, False), "map_min_per10": (0.0, 30.0, False, False),
    "resp_rate_max_per5": (0.0, 14.0, False, False),
    "temperature_max_abs_deviation_from_37c": (0.0, 27.0, True, False),
    "spo2_deficit_per5": (0.0, 20.0, True, False), "gcs_deficit_from15": (0.0, 12.0, True, True),
    "hemoglobin_min_gdl": (0.0, 50.0, False, True), "wbc_max_log2": (None, math.log2(1000), True, True),
    "platelet_min_log2": (None, math.log2(10000), True, True), "bun_max_log2": (None, math.log2(300), True, True),
    "bicarbonate_min_per5": (0.0, 2000.0, False, True),
    "sodium_max_abs_deviation_from140_per5": (0.0, 28.0, True, False),
    "potassium_max_mmol_l": (0.0, 30.0, False, True), "glucose_max_log2": (None, math.log2(10000), True, True),
}


def _admissible(values: np.ndarray, target: str) -> bool:
    lo, hi, lc, hc = BOUNDS[target]
    if not np.isfinite(values).all(): return False
    if lo is not None and np.any(values < lo if lc else values <= lo): return False
    if hi is not None and np.any(values > hi if hc else values >= hi): return False
    return True


def iterative_imputer(raw: pd.DataFrame, init_values: dict[str, float], feedback: bool) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Apply fixed conditionals with or without the predicted-outcome term."""
    pool = pd.read_csv(REPO / "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/fce_pooled_conditional_coefficients.csv")
    coef = load_locked_coefficients(); original = raw.copy(); cur = raw.copy()
    masks = {t: original[t].isna().to_numpy() for t in FCE_TARGETS}
    beta = {t: dict(zip(g.term, g.pooled_coefficient.astype(float))) for t, g in pool.groupby("target_predictor", sort=False)}
    for t in FCE_TARGETS:
        cur.loc[masks[t], t] = float(init_values[t])
    if cur[PRIMARY_PREDICTORS].isna().any().any(): raise RuntimeError("Unexpected non-FCE missing predictor")
    p_old = locked_probability(cur, coef); trace = []
    converged = False
    for cycle in range(1, MAX_CYCLES + 1):
        previous = cur[FCE_TARGETS].to_numpy(float).copy()
        surrogate = p_old.copy()
        for target in FCE_TARGETS:
            mask = masks[target]
            if not mask.any(): continue
            b = beta[target]; pred = np.full(len(cur), float(b["const"]))
            for term in PRIMARY_PREDICTORS:
                if term != target: pred += float(b[term]) * cur[term].to_numpy(float)
            # Feedback-free sensitivity marginalizes the unavailable outcome term
            # by a prespecified MIMIC-only constant prevalence. It neither sets all
            # patients to survivor nor uses an eICU outcome/probability.
            outcome_value = surrogate if feedback else MIMIC_DEVELOPMENT_PREVALENCE
            pred += float(b["hospital_expire_flag"]) * outcome_value
            cur.loc[mask, target] = pred[mask]
        p_new = locked_probability(cur, coef)
        p_delta = np.abs(p_new-p_old)
        value_delta = np.abs(cur[FCE_TARGETS].to_numpy(float)-previous)
        changed = np.concatenate([value_delta[:, j][masks[t]] for j, t in enumerate(FCE_TARGETS) if masks[t].any()])
        max_value_delta = float(changed.max()) if len(changed) else 0.0
        trace.append({"cycle": cycle, "feedback": feedback, "max_probability_delta": float(p_delta.max()),
                      "mean_probability_delta": float(p_delta.mean()), "max_imputed_value_delta": max_value_delta})
        p_old = p_new
        # Fixed feedback rule stops on probability. Feedback-free alternative
        # uses the stricter all-imputed-values rule fixed in the protocol.
        converged = bool(np.all(p_delta < TOL)) if feedback else max_value_delta < TOL
        if converged: break
    if not converged: raise RuntimeError(f"Imputer did not converge within {MAX_CYCLES} cycles; feedback={feedback}")
    for t in FCE_TARGETS:
        obs = ~masks[t]
        if not np.array_equal(cur.loc[obs, t].to_numpy(), original.loc[obs, t].to_numpy()):
            raise RuntimeError(f"Observed values changed: {t}")
        if not _admissible(cur.loc[masks[t], t].to_numpy(float), t):
            raise RuntimeError(f"Inadmissible imputed values: {t}; no clipping is authorized")
    return cur, pd.DataFrame(trace), p_old


def summarize_shifts(reference: np.ndarray, candidate: np.ndarray, label: str) -> dict:
    d = candidate-reference
    return {"candidate": label, "n": len(d), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "minimum": float(d.min()), "p01": float(np.quantile(d,.01)), "p05": float(np.quantile(d,.05)),
            "median": float(np.median(d)), "p95": float(np.quantile(d,.95)), "p99": float(np.quantile(d,.99)),
            "maximum": float(d.max()), "mean_absolute": float(np.abs(d).mean()),
            "maximum_absolute": float(np.abs(d).max())}


def require_external_baseline(input_hashes: dict[str, str]) -> dict:
    if not BASELINE_RECEIPT.is_file():
        raise RuntimeError("Required external baseline receipt absent; run external_baseline_concordance.py --execute first")
    gate = json.loads(BASELINE_RECEIPT.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("scientific_module_sha256") != sha256(REPO / "scripts/sensitivity/posthoc/common.py"):
        raise RuntimeError("External baseline receipt is not PASS for the current scientific module")
    verify_recorded_execution_identity(gate.get("execution_identity"))
    for rel, expected in gate.get("input_hashes", {}).items():
        if sha256(REPO / rel) != expected:
            raise RuntimeError(f"External baseline receipt input changed: {rel}")
    for rel in [needed for needed in input_hashes if needed in gate.get("input_hashes", {})]:
        if input_hashes[rel] != gate["input_hashes"][rel]:
            raise RuntimeError(f"External baseline receipt differs from run input: {rel}")
    return gate


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--execute", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000); args = ap.parse_args()
    if args.bootstrap < 1: raise ValueError("--bootstrap must be positive")
    needed = [
        "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_development_matrix_with_outcome.parquet",
        "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_locked_coefficient_vector.csv",
        "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/fce_pooled_conditional_coefficients.csv",
        "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/fce_initialization_values.csv",
        "outputs/08_external_transport/eicu_final_preoutcome/eicu_predictor_matrix_preimputation.parquet",
        "outputs/08_external_transport/eicu_final_preoutcome/eicu_prediction_ready_matrix_preoutcome.parquet",
        "outputs/08_external_transport/eicu_outcome_attachment/eicu_prediction_ready_with_outcome_availability.parquet",
        "outputs/08_external_transport/eicu_locked_external_validation/EICU_LOCKED_PRIMARY_PERFORMANCE.json",
    ]
    hashes = verify_protected(needed)
    # Metadata/schema preflight only in default mode.
    import pyarrow.parquet as pq
    expected = KEYS + PRIMARY_PREDICTORS
    for rel in needed[4:7]:
        names = pq.ParquetFile(REPO/rel).schema_arrow.names
        if not set(expected).issubset(names): raise RuntimeError(f"Schema mismatch: {rel}")
    if not args.execute:
        state="REQUIRED_NOT_PRESENT"
        if BASELINE_RECEIPT.exists():
            try: require_external_baseline(hashes); state="PASS/current"
            except RuntimeError: state="REQUIRED_NOT_CURRENT"
        print(f"DRY-RUN PASS: fixed hashes, schemas, methods, and output locations validated; external concordance={state}. No patient rows loaded.")
        return
    receipt = require_external_baseline(hashes)
    before=snapshot(needed)
    raw = pd.read_parquet(REPO/needed[-4]).sort_values(KEYS).reset_index(drop=True)
    fixed = pd.read_parquet(REPO/needed[-3]).sort_values(KEYS).reset_index(drop=True)
    attached = pd.read_parquet(REPO/needed[-2]).sort_values(KEYS).reset_index(drop=True)
    if not raw[KEYS].equals(fixed[KEYS]) or not raw[KEYS].equals(attached[KEYS]): raise RuntimeError("eICU identity mismatch")
    dev = pd.read_parquet(REPO/needed[0])
    fixed_init = pd.read_csv(REPO/needed[3]).set_index("target_predictor").initialization_value.to_dict()
    starts = {
        "fixed_mean": fixed_init,
        "development_median": {t: float(dev[t].median()) for t in FCE_TARGETS},
        "development_p10": {t: float(dev[t].quantile(.10)) for t in FCE_TARGETS},
        "development_p90": {t: float(dev[t].quantile(.90)) for t in FCE_TARGETS},
    }
    evaluable = attached.outcome_evaluable.eq(1).to_numpy()
    y = attached.loc[evaluable, "hospital_expire_flag"].to_numpy(float)
    hospitals = attached.loc[evaluable, "hospital_id"].to_numpy()
    fixed_prob = locked_probability(fixed)[evaluable]
    baseline=external_baseline_concordance(attached,locked_probability(fixed))
    traces=[]; shifts=[]; completed={}; probability_by_start={}; imputed_differences=[]
    for label, values in starts.items():
        comp, trace, prob = iterative_imputer(raw, values, feedback=True)
        trace.insert(0, "initialization", label); traces.append(trace)
        probability_by_start[label] = prob[evaluable]
        shifts.append(summarize_shifts(fixed_prob, prob[evaluable], label))
        completed[label] = comp
        for target in FCE_TARGETS:
            mask=raw[target].isna().to_numpy(); delta=comp.loc[mask,target].to_numpy(float)-fixed.loc[mask,target].to_numpy(float)
            imputed_differences.append({"candidate":label,"target":target,"imputed_n":int(mask.sum()),
                "mean_difference":float(delta.mean()) if len(delta) else 0.0,"mean_absolute_difference":float(np.abs(delta).mean()) if len(delta) else 0.0,
                "maximum_absolute_difference":float(np.abs(delta).max()) if len(delta) else 0.0})
    if not np.allclose(probability_by_start["fixed_mean"], fixed_prob, rtol=0, atol=1e-12):
        raise RuntimeError("Fixed-start reproduction does not reproduce fixed predictions")
    if not np.allclose(completed["fixed_mean"][PRIMARY_PREDICTORS].to_numpy(float),
                       fixed[PRIMARY_PREDICTORS].to_numpy(float),rtol=0,atol=1e-10):
        raise RuntimeError("Fixed-start reproduction does not reproduce the fixed completed predictor matrix")
    alt, alt_trace, alt_prob_all = iterative_imputer(raw, fixed_init, feedback=False)
    alt_trace.insert(0, "initialization", "feedback_free_fixed_mean"); traces.append(alt_trace)
    alt_prob = alt_prob_all[evaluable]; shifts.append(summarize_shifts(fixed_prob, alt_prob, "feedback_free"))
    for target in FCE_TARGETS:
        mask=raw[target].isna().to_numpy(); delta=alt.loc[mask,target].to_numpy(float)-fixed.loc[mask,target].to_numpy(float)
        imputed_differences.append({"candidate":"feedback_free","target":target,"imputed_n":int(mask.sum()),
            "mean_difference":float(delta.mean()) if len(delta) else 0.0,"mean_absolute_difference":float(np.abs(delta).mean()) if len(delta) else 0.0,
            "maximum_absolute_difference":float(np.abs(delta).max()) if len(delta) else 0.0})
    out = RESULTS / "analysis3"; out.mkdir(parents=True, exist_ok=True)
    trace_path=out/"IMPUTATION_CONVERGENCE.csv"; shift_path=out/"PREDICTION_SHIFTS.csv"; metrics_path=out/"PERFORMANCE_COMPARISON.csv"; imp_path=out/"IMPUTED_VALUE_DIFFERENCES.csv"
    atomic_csv(pd.concat(traces, ignore_index=True), trace_path); atomic_csv(pd.DataFrame(shifts), shift_path); atomic_csv(pd.DataFrame(imputed_differences),imp_path)
    point=[]
    for label, prob in [("fixed_feedback", fixed_prob), ("feedback_free", alt_prob)]: point.append({"method": label, **scalar_metrics(y, prob)})
    atomic_csv(pd.DataFrame(point), metrics_path)
    checkpoint=CHECKPOINTS/f"analysis3_hospital_bootstrap_{args.bootstrap}_amended.csv"
    ensure_checkpoint_contract(f"analysis3_hospital_bootstrap_{args.bootstrap}_amended",{"input_hashes":hashes,"baseline":baseline,"baseline_receipt_sha256":sha256(BASELINE_RECEIPT),"bootstrap":args.bootstrap,"seed":2026102903,"tol":TOL,"max_cycles":MAX_CYCLES,"alternative_prevalence":MIMIC_DEVELOPMENT_PREVALENCE},[checkpoint])
    boot = hospital_cluster_bootstrap(y, fixed_prob, hospitals, args.bootstrap, 2026102903, alt_prob,
                                      checkpoint)
    boot_path=out/"HOSPITAL_BOOTSTRAP.csv"; atomic_csv(boot, boot_path)
    interval_rows=[]
    for col in [c for c in boot.columns if c.startswith(("a_","b_","difference_"))]:
        lo,hi=np.quantile(boot[col],[.025,.975]); interval_rows.append({"quantity":col,"p025":lo,"p975":hi,"replicates":args.bootstrap})
    interval_path=out/"HOSPITAL_BOOTSTRAP_INTERVALS.csv"; atomic_csv(pd.DataFrame(interval_rows),interval_path)
    post=verify_postflight(before);disc=disclosure_check([trace_path,shift_path,metrics_path,imp_path,boot_path,interval_path])
    baseline_path=out/"LOCKED_BASELINE_CONCORDANCE.json";atomic_json(baseline,baseline_path)
    m=manifest("analysis3_eicu_imputation_robustness","COMPLETE",{"bootstrap":args.bootstrap,"seed":2026102903,"tol":TOL,"max_cycles":MAX_CYCLES,
               "feedback_free_outcome_term":"constant harmonized-MIMIC development prevalence 537/2098","baseline_receipt_sha256":sha256(BASELINE_RECEIPT)},
               {**before,str(BASELINE_RECEIPT.relative_to(REPO)):sha256(BASELINE_RECEIPT)},[trace_path,shift_path,metrics_path,imp_path,boot_path,interval_path,baseline_path],post,disc)
    atomic_json(m,out/"RESULT_MANIFEST.json")
    print(f"COMPLETE: {out}")


if __name__ == "__main__": main()
