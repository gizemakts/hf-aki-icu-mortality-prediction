#!/usr/bin/env python3
"""Outcome-free external imputation sensitivity implementation.

This module is inert unless ``--execute`` is supplied.  Real-data preparation
and outcome evaluation are deliberately separate phases with separate approval
gates.  Unit tests use synthetic DataFrames and never load project datasets.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from common import PRIMARY_PREDICTORS, atomic_json, hospital_cluster_bootstrap, scalar_metrics


SCRIPT = Path(__file__).resolve()
WORKSPACE = SCRIPT.parents[0]
REPO = WORKSPACE.parents[2]

PLAN = WORKSPACE / "protocol" / "EXTERNAL_IMPUTATION_SENSITIVITY_PLAN.md"
READINESS = WORKSPACE / "protocol" / "EXECUTION_READINESS_CHECKLIST.md"

REFERENCE_SOURCE = REPO / "scripts" / "cohort" / "eicu_crd" / "final_preoutcome_build.py"
DEVELOPMENT_SOURCE = REPO / "scripts" / "modeling" / "harmonized" / "fit_harmonized_model.py"
DEVELOPMENT_MATRIX = (
    REPO / "outputs" / "08_external_transport" / "mimic_cr_rrt_harmonized_refit"
    / "harmonized_cr_rrt_predictor_matrix_preoutcome.parquet"
)
LOCKED_COEFFICIENTS = (
    REPO / "outputs" / "08_external_transport" / "mimic_cr_rrt_harmonized_refit"
    / "harmonized_cr_rrt_locked_coefficient_vector.csv"
)
EXTERNAL_RAW = (
    REPO / "outputs" / "08_external_transport" / "eicu_final_preoutcome"
    / "eicu_predictor_matrix_preimputation.parquet"
)
REFERENCE_COMPLETED = (
    REPO / "outputs" / "08_external_transport" / "eicu_final_preoutcome"
    / "eicu_prediction_ready_matrix_preoutcome.parquet"
)
REFERENCE_FCE_COEFFICIENTS = (
    REPO / "outputs" / "08_external_transport" / "mimic_cr_rrt_harmonized_refit"
    / "fce_pooled_conditional_coefficients.csv"
)
REFERENCE_FCE_INITIALIZATION = (
    REPO / "outputs" / "08_external_transport" / "mimic_cr_rrt_harmonized_refit"
    / "fce_initialization_values.csv"
)
OUTCOME_EVALUABLE = (
    REPO / "outputs" / "08_external_transport" / "eicu_outcome_attachment"
    / "eicu_external_validation_evaluable.parquet"
)
REFERENCE_PERFORMANCE = (
    REPO / "outputs" / "08_external_transport" / "eicu_locked_external_validation"
    / "EICU_LOCKED_PRIMARY_PERFORMANCE.json"
)

DEVELOPMENT_KEYS = ["subject_id", "stay_id"]
EXTERNAL_KEYS = ["unique_pid", "hstay_id", "stay_id", "hospital_id"]
FCE_TARGETS = PRIMARY_PREDICTORS[2:16]
NON_FCE_PREDICTORS = [c for c in PRIMARY_PREDICTORS if c not in FCE_TARGETS]

RISK_TOLERANCE = 1e-6
MAX_CYCLES = 50
TOLERANCE_POLICY_ID = "outcome_free_robust_span_v1"
TOLERANCE_FORMULA = "tau_j = 1e-6 * max(1, observed_P99_j - observed_P01_j)"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026102903
REFERENCE_PREDICTOR_REPRODUCTION_ATOL = 1e-12
REFERENCE_RISK_REPRODUCTION_ATOL = 1e-12
# Source digests and protected input digests are checked independently.
# Importing the module does not access restricted study data.
EXPECTED_PREPARE_HASHES = {
    str(REFERENCE_SOURCE.relative_to(REPO)): "7d25a2359cc541cb3fd3adf5ae0d36740c2fce24c465860cfbc809ceaf0bcca2",
    str(DEVELOPMENT_SOURCE.relative_to(REPO)): "786c938d3a7f2d2d8362f409ad190abd825f42fe2ed43fb570bb659cc4e434f8",
    str(DEVELOPMENT_MATRIX.relative_to(REPO)): "528bf2f30fa8e79a02c6246b5902c64b95c9bf5da4801e2ec2abdb190f1304bf",
    str(LOCKED_COEFFICIENTS.relative_to(REPO)): "3773d8ce03655e014e0e0ea578ba85888470ae248070552b089fd94c27afb513",
    str(EXTERNAL_RAW.relative_to(REPO)): "43faf1beced67ea3e6cdd4ec3733160d9fd61b14932aba885fd7998aa14a168e",
    str(REFERENCE_COMPLETED.relative_to(REPO)): "9269e4c8eb858ea3a4e7c559d533827ea72788e44d30060a5502433b55591e66",
    str(REFERENCE_FCE_COEFFICIENTS.relative_to(REPO)): "40aba6b6158b1e4f534bb4dc68d3cc7e0d802d099935a37367c60c75423045d2",
    str(REFERENCE_FCE_INITIALIZATION.relative_to(REPO)): "544ada93e615da32f44ff083349ae6fe647e955d9b1e5c1ca7c53cea6343ada5",
}
EXPECTED_EVALUATION_HASHES = {
    str(OUTCOME_EVALUABLE.relative_to(REPO)): "3971f51cd11e83ad85b38fcd7a4cf7e3e5943aca52e668911f61783b08052e8a",
    str(REFERENCE_PERFORMANCE.relative_to(REPO)): "96a0d04c7cf2ccdf3747e94837e8206fa59928d5b90198c4b5561fdfe7d600a4",
}
EXPECTED_HASHES = {**EXPECTED_PREPARE_HASHES, **EXPECTED_EVALUATION_HASHES}

OUTCOME_LIKE_TOKENS = (
    "expire", "mortality", "death", "outcome", "survival", "probability",
    "prediction", "auroc", "auc", "brier", "calibration",
)
CLOUD_TOKENS = (
    "onedrive", "dropbox", "google drive", "icloud", "box sync",
    "sharepoint", "syncthing",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class GuardFailure(RuntimeError):
    """Raised for a governed, recorded hard failure."""


@dataclass
class FailureRecorder:
    path: Path | None = None

    def fail(self, code: str, message: str, **details: Any) -> None:
        receipt = {
            "status": "FAILED",
            "failure_code": code,
            "message": message,
            "details": details,
            "recorded_utc": utcnow(),
        }
        if self.path is not None:
            atomic_json(receipt, self.path)
        raise GuardFailure(f"{code}: {message}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GuardFailure(f"Expected a JSON object: {path}")
    return payload


def verify_source_hashes(expected: dict[str, str] = EXPECTED_HASHES) -> dict[str, str]:
    observed: dict[str, str] = {}
    for rel, wanted in expected.items():
        path = REPO / rel
        if not path.is_file():
            raise GuardFailure(f"Required governed source is missing: {rel}")
        got = sha256(path)
        if got.lower() != wanted.lower():
            raise GuardFailure(f"SHA-256 mismatch for {rel}: expected {wanted}, observed {got}")
        observed[rel] = got
    return observed


def _safe_ast_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_safe_ast_value(x) for x in node.elts)
    if isinstance(node, ast.Dict):
        return {_safe_ast_value(k): _safe_ast_value(v) for k, v in zip(node.keys, node.values)}
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "math"
        and node.func.attr == "log2"
        and len(node.args) == 1
        and not node.keywords
    ):
        return math.log2(float(_safe_ast_value(node.args[0])))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -float(_safe_ast_value(node.operand))
    raise GuardFailure(f"Unsupported expression in canonical bounds: {ast.dump(node)}")


def load_canonical_fce_admissible(source: Path = REFERENCE_SOURCE) -> dict[str, tuple[Any, ...]]:
    """Extract FCE_ADMISSIBLE without importing or executing the reference script."""
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "FCE_ADMISSIBLE" for target in node.targets)
    ]
    if len(assignments) != 1:
        raise GuardFailure("Canonical FCE_ADMISSIBLE assignment is absent or ambiguous")
    bounds = _safe_ast_value(assignments[0].value)
    if list(bounds) != FCE_TARGETS:
        raise GuardFailure("Canonical FCE_ADMISSIBLE target order differs from the locked 14-target order")
    if any(not isinstance(bounds[t], tuple) or len(bounds[t]) != 4 for t in FCE_TARGETS):
        raise GuardFailure("Canonical FCE_ADMISSIBLE tuple schema is invalid")
    return bounds


def inadmissible_mask(values: np.ndarray, bound: tuple[Any, ...]) -> np.ndarray:
    """Apply the reference check_fce_scale boundary semantics unchanged."""
    vals = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(vals)
    lo, hi, lo_closed, hi_closed = bound
    tol = 1e-10
    if lo is not None:
        bad |= vals < lo - tol if lo_closed else vals <= lo
    if hi is not None:
        bad |= vals > hi + tol if hi_closed else vals >= hi
    return bad


def check_imputations(
    frame: pd.DataFrame,
    masks: dict[str, np.ndarray],
    bounds: dict[str, tuple[Any, ...]],
    cycle: int,
    recorder: FailureRecorder,
) -> None:
    for target in FCE_TARGETS:
        mask = masks[target]
        values = frame.loc[mask, target].to_numpy(dtype=np.float64)
        bad = inadmissible_mask(values, bounds[target])
        if bad.any():
            recorder.fail(
                "INADMISSIBLE_IMPUTATION",
                "An imputed value violated the unchanged canonical FCE_ADMISSIBLE bound",
                cycle=cycle,
                target=target,
                failure_count=int(bad.sum()),
                failed_row_positions=np.flatnonzero(mask)[bad][:20].astype(int).tolist(),
                failed_values=[float(v) for v in values[bad][:20]],
                failed_values_truncated=bool(bad.sum() > 20),
            )


def validate_predictor_frame(
    frame: pd.DataFrame,
    keys: list[str],
    label: str,
    recorder: FailureRecorder,
) -> pd.DataFrame:
    expected = keys + PRIMARY_PREDICTORS
    if frame.columns.tolist() != expected:
        recorder.fail("SCHEMA_MISMATCH", f"{label} columns/order differ", expected=expected, observed=frame.columns.tolist())
    outcome_like = [c for c in frame.columns if any(token in c.lower() for token in OUTCOME_LIKE_TOKENS)]
    if outcome_like:
        recorder.fail("OUTCOME_COLUMN_PRESENT", f"{label} contains outcome-like columns", columns=outcome_like)
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        recorder.fail("IDENTITY_FAILURE", f"{label} keys are missing or duplicated")
    for column in PRIMARY_PREDICTORS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float64)
        values = frame[column].to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            recorder.fail("NONFINITE_INPUT", f"{label} has infinite predictor values", predictor=column)
    for column in NON_FCE_PREDICTORS:
        if frame[column].isna().any():
            recorder.fail("NON_FCE_MISSING", f"{label} has missing non-FCE predictor", predictor=column)
    return frame


def load_locked_coefficients(path: Path = LOCKED_COEFFICIENTS) -> dict[str, float]:
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["term", "locked_coefficient"]:
        raise GuardFailure("Locked coefficient schema/order changed")
    if frame["term"].tolist() != ["const"] + PRIMARY_PREDICTORS:
        raise GuardFailure("Locked coefficient term order changed")
    values = frame["locked_coefficient"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise GuardFailure("Locked coefficients are nonfinite")
    return dict(zip(frame["term"], values))


def locked_risk(frame: pd.DataFrame, coefficients: dict[str, float]) -> np.ndarray:
    x = frame[PRIMARY_PREDICTORS].to_numpy(dtype=np.float64)
    beta = np.array([coefficients[c] for c in PRIMARY_PREDICTORS], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(beta).all():
        raise GuardFailure("Nonfinite locked-risk input")
    eta = np.float64(coefficients["const"]) + x @ beta
    risk = np.empty_like(eta, dtype=np.float64)
    positive = eta >= 0
    risk[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
    exp_eta = np.exp(eta[~positive])
    risk[~positive] = exp_eta / (1.0 + exp_eta)
    if not np.isfinite(risk).all() or ((risk <= 0) | (risk >= 1)).any():
        raise GuardFailure("Invalid locked-risk result")
    return risk


def propose_tolerances(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    """Create explicit outcome-free proposals; this function does not approve them."""
    rows: list[dict[str, Any]] = []
    tolerances: dict[str, float] = {}
    eps = np.finfo(np.float64).eps
    for target in FCE_TARGETS:
        values = frame[target].dropna().to_numpy(dtype=np.float64)
        if len(values) < 100 or not np.isfinite(values).all():
            raise GuardFailure(f"Tolerance diagnostics insufficient for {target}: finite observed n={len(values)}")
        p01, p25, median, p75, p99 = np.quantile(
            values, [0.01, 0.25, 0.5, 0.75, 0.99], method="linear"
        )
        if not np.isfinite([p01, p25, median, p75, p99]).all() or p99 < p01:
            raise GuardFailure(f"Invalid outcome-free tolerance quantiles for {target}")
        scale = max(1.0, float(p99 - p01))
        tolerance = 1e-6 * scale
        max_abs = max(1.0, float(np.max(np.abs(values))))
        epsilon_ratio = tolerance / (eps * max_abs)
        if epsilon_ratio < 100.0:
            raise GuardFailure(f"Proposed tolerance is below 100 floating-point spacings for {target}")
        tolerances[target] = tolerance
        rows.append({
            "target_predictor": target,
            "observed_n": int(len(values)),
            "missing_n": int(frame[target].isna().sum()),
            "missing_fraction": float(frame[target].isna().mean()),
            "p01": float(p01), "p25": float(p25), "median": float(median),
            "p75": float(p75), "p99": float(p99), "robust_span_scale": scale,
            "proposed_tolerance": tolerance, "max_abs_observed": max_abs,
            "tolerance_to_epsilon_spacing_ratio": float(epsilon_ratio),
            "policy_id": TOLERANCE_POLICY_ID,
            "approval_status": "PROPOSED_NOT_APPROVED",
        })
    return tolerances, pd.DataFrame(rows)


def _initialize_missing(frame: pd.DataFrame, means: dict[str, float]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    original = frame.copy(deep=True)
    current = frame.copy(deep=True)
    masks = {target: original[target].isna().to_numpy() for target in FCE_TARGETS}
    for target in FCE_TARGETS:
        value = float(means[target])
        if not np.isfinite(value):
            raise GuardFailure(f"Nonfinite initialization mean for {target}")
        current.loc[masks[target], target] = value
    if current[PRIMARY_PREDICTORS].isna().any().any():
        raise GuardFailure("Missing model inputs remain after initialization")
    return current, masks


def _value_deltas(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for target in FCE_TARGETS:
        mask = masks[target]
        if mask.any():
            delta = np.abs(
                current.loc[mask, target].to_numpy(np.float64)
                - previous.loc[mask, target].to_numpy(np.float64)
            )
            result[target] = float(delta.max())
        else:
            result[target] = 0.0
    return result


def _assert_observed_preserved(
    original: pd.DataFrame,
    completed: pd.DataFrame,
    masks: dict[str, np.ndarray],
    recorder: FailureRecorder,
) -> None:
    for target in FCE_TARGETS:
        observed = ~masks[target]
        before = original.loc[observed, target].to_numpy(np.float64)
        after = completed.loc[observed, target].to_numpy(np.float64)
        if not np.array_equal(before, after):
            recorder.fail("OBSERVED_VALUE_CHANGED", "Imputation changed an observed value", target=target)


def fit_predictor_only_imputer(
    development: pd.DataFrame,
    coefficients: dict[str, float],
    tolerances: dict[str, float],
    bounds: dict[str, tuple[Any, ...]],
    recorder: FailureRecorder | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Fit deterministic predictor-only OLS conditionals by chained updates."""
    recorder = recorder or FailureRecorder()
    original = development.copy(deep=True)
    means = {t: float(original[t].mean(skipna=True)) for t in FCE_TARGETS}
    current, masks = _initialize_missing(original, means)
    previous_risk = locked_risk(current, coefficients)
    trace: list[dict[str, Any]] = []
    final_models: dict[str, dict[str, float]] = {}

    for cycle in range(1, MAX_CYCLES + 1):
        previous = current.copy(deep=True)
        cycle_models: dict[str, dict[str, float]] = {}
        for target in FCE_TARGETS:
            covariates = [c for c in PRIMARY_PREDICTORS if c != target]
            observed = ~masks[target]
            design = np.column_stack([
                np.ones(int(observed.sum()), dtype=np.float64),
                current.loc[observed, covariates].to_numpy(dtype=np.float64),
            ])
            response = original.loc[observed, target].to_numpy(dtype=np.float64)
            rank = int(np.linalg.matrix_rank(design))
            if rank != design.shape[1] or len(response) <= rank:
                recorder.fail(
                    "OLS_DESIGN_FAILURE", "Predictor-only conditional equation is rank deficient or has no positive residual df",
                    cycle=cycle, target=target, observed_n=len(response), rank=rank, columns=design.shape[1],
                )
            beta, *_ = np.linalg.lstsq(design, response, rcond=None)
            if not np.isfinite(beta).all():
                recorder.fail("OLS_COEFFICIENT_FAILURE", "Nonfinite predictor-only OLS coefficient", cycle=cycle, target=target)
            terms = ["const"] + covariates
            cycle_models[target] = dict(zip(terms, beta.astype(float)))
            missing = masks[target]
            if missing.any():
                update_design = np.column_stack([
                    np.ones(int(missing.sum()), dtype=np.float64),
                    current.loc[missing, covariates].to_numpy(dtype=np.float64),
                ])
                predicted = update_design @ beta
                if not np.isfinite(predicted).all():
                    recorder.fail("OLS_UPDATE_FAILURE", "Nonfinite predictor-only conditional expectation", cycle=cycle, target=target)
                current.loc[missing, target] = predicted
                check_imputations(current, {t: masks[t] if t == target else np.zeros(len(current), bool) for t in FCE_TARGETS}, bounds, cycle, recorder)

        current_risk = locked_risk(current, coefficients)
        risk_delta = np.abs(current_risk - previous_risk)
        value_delta = _value_deltas(current, previous, masks)
        risk_pass = bool(np.all(risk_delta < RISK_TOLERANCE))
        predictor_pass = bool(all(value_delta[t] < tolerances[t] for t in FCE_TARGETS))
        trace.append({
            "cycle": cycle,
            "max_abs_risk_change": float(risk_delta.max(initial=0.0)),
            "risk_converged": risk_pass,
            "predictors_converged": predictor_pass,
            "dual_converged": risk_pass and predictor_pass,
            **{f"max_abs_change__{t}": value_delta[t] for t in FCE_TARGETS},
            **{f"tolerance__{t}": float(tolerances[t]) for t in FCE_TARGETS},
        })
        previous_risk = current_risk
        final_models = cycle_models
        if risk_pass and predictor_pass:
            _assert_observed_preserved(original, current, masks, recorder)
            check_imputations(current, masks, bounds, cycle, recorder)
            return final_models, means, current, pd.DataFrame(trace)

    recorder.fail(
        "DUAL_NONCONVERGENCE",
        "Both risk and imputed-predictor convergence were not satisfied within 50 full cycles",
        max_cycles=MAX_CYCLES,
        final_trace=trace[-1] if trace else None,
    )
    raise AssertionError("unreachable")


def apply_predictor_only_imputer(
    external: pd.DataFrame,
    models: dict[str, dict[str, float]],
    means: dict[str, float],
    coefficients: dict[str, float],
    tolerances: dict[str, float],
    bounds: dict[str, tuple[Any, ...]],
    recorder: FailureRecorder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    recorder = recorder or FailureRecorder()
    original = external.copy(deep=True)
    current, masks = _initialize_missing(original, means)
    previous_risk = locked_risk(current, coefficients)
    trace: list[dict[str, Any]] = []

    for cycle in range(1, MAX_CYCLES + 1):
        previous = current.copy(deep=True)
        for target in FCE_TARGETS:
            covariates = [c for c in PRIMARY_PREDICTORS if c != target]
            expected_terms = ["const"] + covariates
            model = models.get(target, {})
            if list(model) != expected_terms:
                recorder.fail("MODEL_SCHEMA_FAILURE", "Conditional equation terms/order differ", target=target)
            missing = masks[target]
            if missing.any():
                design = np.column_stack([
                    np.ones(int(missing.sum()), dtype=np.float64),
                    current.loc[missing, covariates].to_numpy(dtype=np.float64),
                ])
                beta = np.array([model[t] for t in expected_terms], dtype=np.float64)
                predicted = design @ beta
                if not np.isfinite(predicted).all():
                    recorder.fail("OLS_UPDATE_FAILURE", "Nonfinite external conditional expectation", cycle=cycle, target=target)
                current.loc[missing, target] = predicted
                check_imputations(current, {t: masks[t] if t == target else np.zeros(len(current), bool) for t in FCE_TARGETS}, bounds, cycle, recorder)

        current_risk = locked_risk(current, coefficients)
        risk_delta = np.abs(current_risk - previous_risk)
        value_delta = _value_deltas(current, previous, masks)
        risk_pass = bool(np.all(risk_delta < RISK_TOLERANCE))
        predictor_pass = bool(all(value_delta[t] < tolerances[t] for t in FCE_TARGETS))
        trace.append({
            "cycle": cycle,
            "max_abs_risk_change": float(risk_delta.max(initial=0.0)),
            "risk_converged": risk_pass,
            "predictors_converged": predictor_pass,
            "dual_converged": risk_pass and predictor_pass,
            **{f"max_abs_change__{t}": value_delta[t] for t in FCE_TARGETS},
            **{f"tolerance__{t}": float(tolerances[t]) for t in FCE_TARGETS},
        })
        previous_risk = current_risk
        if risk_pass and predictor_pass:
            _assert_observed_preserved(original, current, masks, recorder)
            check_imputations(current, masks, bounds, cycle, recorder)
            if current[PRIMARY_PREDICTORS].isna().any().any():
                recorder.fail("MISSING_AFTER_IMPUTATION", "Missing predictors remain after convergence")
            return current, pd.DataFrame(trace), current_risk.astype(np.float64)

    recorder.fail(
        "DUAL_NONCONVERGENCE",
        "Both risk and imputed-predictor convergence were not satisfied within 50 full cycles",
        max_cycles=MAX_CYCLES,
        final_trace=trace[-1] if trace else None,
    )
    raise AssertionError("unreachable")


def assert_exact_key_alignment(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> None:
    if len(left) != len(right):
        raise GuardFailure(f"Row-count mismatch: {len(left)} versus {len(right)}")
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise GuardFailure("Duplicate keys prevent exact row alignment")
    for key in keys:
        if not np.array_equal(left[key].to_numpy(), right[key].to_numpy()):
            raise GuardFailure(f"Exact row/order alignment failed for key {key}")


def reproduction_fixed_reference_fce(
    raw: pd.DataFrame,
    coefficients: dict[str, float],
    fce_pool: pd.DataFrame,
    initialization: pd.DataFrame,
    bounds: dict[str, tuple[Any, ...]],
    recorder: FailureRecorder | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Reconstruct the reference FCE for an immutable consistency check."""
    recorder = recorder or FailureRecorder()
    original = raw.copy(deep=True)
    current = raw.copy(deep=True)
    masks = {target: original[target].isna().to_numpy() for target in FCE_TARGETS}

    required_init_columns = ["target_predictor", "initialization_value"]
    if not set(required_init_columns).issubset(initialization.columns):
        recorder.fail("REFERENCE_INIT_SCHEMA", "Fixed FCE initialization columns are absent")
    if initialization["target_predictor"].tolist() != FCE_TARGETS:
        recorder.fail("REFERENCE_INIT_ORDER", "Fixed FCE initialization target order differs")
    init_values = dict(zip(
        initialization["target_predictor"],
        pd.to_numeric(initialization["initialization_value"], errors="raise").astype(np.float64),
    ))
    for target in FCE_TARGETS:
        current.loc[masks[target], target] = float(init_values[target])
    if current[PRIMARY_PREDICTORS].isna().any().any():
        recorder.fail("REFERENCE_INIT_MISSING", "Missing values remain after fixed FCE initialization")

    beta_by_target: dict[str, dict[str, float]] = {}
    for target in FCE_TARGETS:
        group = fce_pool.loc[fce_pool["target_predictor"].eq(target)].copy()
        expected_terms = {"const", "hospital_expire_flag"} | (set(PRIMARY_PREDICTORS) - {target})
        if len(group) != 20 or group["term"].duplicated().any() or set(group["term"]) != expected_terms:
            recorder.fail("REFERENCE_FCE_SCHEMA", "Fixed FCE conditional vector is malformed", target=target)
        values = pd.to_numeric(group["pooled_coefficient"], errors="raise").to_numpy(np.float64)
        if not np.isfinite(values).all():
            recorder.fail("REFERENCE_FCE_NONFINITE", "Fixed FCE coefficient is nonfinite", target=target)
        beta_by_target[target] = dict(zip(group["term"], values.astype(float)))

    previous_risk = locked_risk(current, coefficients)
    trace: list[dict[str, Any]] = []
    for cycle in range(1, MAX_CYCLES + 1):
        surrogate = previous_risk.copy()
        for target in FCE_TARGETS:
            missing = masks[target]
            if not missing.any():
                continue
            beta = beta_by_target[target]
            predicted = np.full(len(current), float(beta["const"]), dtype=np.float64)
            for predictor in PRIMARY_PREDICTORS:
                if predictor != target:
                    predicted += float(beta[predictor]) * current[predictor].to_numpy(np.float64)
            predicted += float(beta["hospital_expire_flag"]) * surrogate
            if not np.isfinite(predicted).all():
                recorder.fail("REFERENCE_FCE_UPDATE", "Reference FCE produced a nonfinite update", cycle=cycle, target=target)
            current.loc[missing, target] = predicted[missing]
        check_imputations(current, masks, bounds, cycle, recorder)
        current_risk = locked_risk(current, coefficients)
        delta = np.abs(current_risk - previous_risk)
        converged = bool(np.all(delta < RISK_TOLERANCE))
        trace.append({
            "cycle": cycle,
            "max_abs_probability_delta": float(delta.max(initial=0.0)),
            "mean_abs_probability_delta": float(delta.mean()) if len(delta) else 0.0,
            "patients_at_or_above_tolerance": int((delta >= RISK_TOLERANCE).sum()),
            "converged_all_patients": converged,
        })
        previous_risk = current_risk
        if converged:
            _assert_observed_preserved(original, current, masks, recorder)
            return current, pd.DataFrame(trace), current_risk
    recorder.fail(
        "REFERENCE_FCE_NONCONVERGENCE",
        "Original feedback reference failed to converge within 50 cycles",
        max_cycles=MAX_CYCLES,
    )
    raise AssertionError("unreachable")


def authoritative_completed_predictor_reproduction_gate(
    raw: pd.DataFrame,
    fixed_completed: pd.DataFrame,
    coefficients: dict[str, float],
    fce_pool: pd.DataFrame,
    initialization: pd.DataFrame,
    bounds: dict[str, tuple[Any, ...]],
    recorder: FailureRecorder | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Hard gate matching the original independent reproduction's authoritative tolerances."""
    recorder = recorder or FailureRecorder()
    assert_exact_key_alignment(raw, fixed_completed, EXTERNAL_KEYS)
    reproduction, trace, reproduction_risk = reproduction_fixed_reference_fce(
        raw, coefficients, fce_pool, initialization, bounds, recorder
    )
    predictor_differences: dict[str, float] = {}
    for predictor in PRIMARY_PREDICTORS:
        difference = np.abs(
            reproduction[predictor].to_numpy(np.float64)
            - fixed_completed[predictor].to_numpy(np.float64)
        )
        maximum = float(difference.max(initial=0.0))
        predictor_differences[predictor] = maximum
        if maximum > REFERENCE_PREDICTOR_REPRODUCTION_ATOL:
            recorder.fail(
                "REFERENCE_PREDICTOR_REPRODUCTION_MISMATCH",
                "Reference completed predictor reproduction exceeded the authoritative 1e-12 tolerance",
                predictor=predictor,
                maximum_absolute_difference=maximum,
                rtol=0.0,
                atol=REFERENCE_PREDICTOR_REPRODUCTION_ATOL,
            )
    fixed_risk = locked_risk(fixed_completed, coefficients)
    risk_difference = np.abs(reproduction_risk - fixed_risk)
    maximum_risk_difference = float(risk_difference.max(initial=0.0))
    if maximum_risk_difference > REFERENCE_RISK_REPRODUCTION_ATOL:
        recorder.fail(
            "REFERENCE_RISK_REPRODUCTION_MISMATCH",
            "Reference risk reproduction exceeded the authoritative 1e-12 tolerance",
            maximum_absolute_difference=maximum_risk_difference,
            rtol=0.0,
            atol=REFERENCE_RISK_REPRODUCTION_ATOL,
        )
    receipt = {
        "status": "PASS",
        "predictor_rtol": 0.0,
        "predictor_atol": REFERENCE_PREDICTOR_REPRODUCTION_ATOL,
        "risk_rtol": 0.0,
        "risk_atol": REFERENCE_RISK_REPRODUCTION_ATOL,
        "maximum_predictor_absolute_difference": max(predictor_differences.values(), default=0.0),
        "predictor_maximum_absolute_differences": predictor_differences,
        "maximum_risk_absolute_difference": maximum_risk_difference,
        "cycles": int(len(trace)),
        "created_utc": utcnow(),
    }
    return receipt, trace


def atomic_parquet_exact(frame: pd.DataFrame, path: Path, float_columns: list[str]) -> str:
    """Write float64 Parquet atomically and require exact binary numeric round-trip."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    implemented = frame.copy()
    for column in float_columns:
        implemented[column] = implemented[column].astype(np.float64)
    implemented.to_parquet(temp, index=False)
    reproduction = pd.read_parquet(temp)
    if reproduction.columns.tolist() != implemented.columns.tolist() or len(reproduction) != len(implemented):
        temp.unlink(missing_ok=True)
        raise GuardFailure("Parquet round-trip schema or row count differs")
    for column in float_columns:
        if reproduction[column].dtype != np.dtype("float64"):
            temp.unlink(missing_ok=True)
            raise GuardFailure(f"Parquet round-trip dtype differs for {column}: {reproduction[column].dtype}")
        if not np.array_equal(reproduction[column].to_numpy(), implemented[column].to_numpy()):
            temp.unlink(missing_ok=True)
            raise GuardFailure(f"Parquet round-trip values differ for {column}")
    os.replace(temp, path)
    return sha256(path)


def _is_confirmed_text(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    return value.strip().upper() not in {"UNKNOWN", "UNASSESSED", "ASSESSMENT_NEEDED", "N/A"}


def validate_approval_manifest(payload: dict[str, Any], phase: str) -> None:
    required_true = [
        "evidence_gap_approach_approved",
        "predictor_only_comparator_approved",
        "canonical_fce_bounds_required",
        "dual_convergence_required",
        "tolerance_policy_approved",
        "serialization_policy_approved",
        "secure_environment_approved",
        "execution_authorized",
    ]
    required_text = [
        "author_approver", "author_approval_datetime",
        "approval_document_path", "approval_document_sha256",
    ]
    if phase == "evaluate":
        required_true.extend(["preoutcome_lock_independently_approved", "outcome_access_approved"])
        required_text.extend(["preoutcome_lock_manifest_path", "preoutcome_lock_manifest_sha256"])
    missing = [field for field in required_true if payload.get(field) is not True]
    missing += [field for field in required_text if not _is_confirmed_text(payload.get(field))]
    if payload.get("status") != "APPROVED" or payload.get("tolerance_policy_id") != TOLERANCE_POLICY_ID:
        missing.append("status/tolerance_policy_id")
    if missing:
        raise GuardFailure(f"Execution approval manifest is incomplete/unapproved: {sorted(set(missing))}")
    evidence_path = Path(payload["approval_document_path"])
    if not evidence_path.is_file():
        raise GuardFailure("Approval evidence file does not exist")
    if sha256(evidence_path).lower() != payload["approval_document_sha256"].lower():
        raise GuardFailure("Approval evidence SHA-256 mismatch")


def validate_secure_environment(root: Path, payload: dict[str, Any]) -> None:
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or not resolved.is_absolute():
        raise GuardFailure("Restricted root must be an existing absolute directory")
    lowered = str(resolved).lower()
    in_named_cloud_path = any(token in lowered for token in CLOUD_TOKENS)
    if in_named_cloud_path and payload.get("named_cloud_path_reviewed_as_inactive_or_approved") is not True:
        raise GuardFailure("Named cloud path lacks documented inactive/approved sync review")
    try:
        resolved.relative_to(REPO.resolve())
        raise GuardFailure("Restricted root is inside the repository")
    except ValueError:
        pass
    required_true = [
        "physionet_licensee_confirmed", "human_subject_training_current",
        "restricted_data_not_shared", "reasonable_security_controls_verified",
        "no_active_unapproved_cloud_sync_verified", "access_control_verified",
        "local_fixed_storage_verified", "logs_and_temp_confined",
        "external_ai_transfer_prohibited",
    ]
    missing = [field for field in required_true if payload.get(field) is not True]
    required_text = [
        "approved_absolute_root", "attesting_owner", "attestation_datetime",
        "sync_verification_method", "access_review_evidence",
        "retention_policy",
    ]
    missing += [field for field in required_text if not _is_confirmed_text(payload.get(field))]
    if Path(payload.get("approved_absolute_root", "")).resolve() != resolved:
        missing.append("approved_absolute_root_match")
    for variable in ("TEMP", "TMP"):
        value = os.environ.get(variable)
        if not value:
            missing.append(f"{variable}_set")
            continue
        try:
            Path(value).resolve().relative_to(resolved)
        except ValueError:
            missing.append(f"{variable}_inside_restricted_root")
    if missing:
        raise GuardFailure(f"Secure-environment attestation is incomplete: {sorted(set(missing))}")


def execution_gate(
    execute: bool,
    phase: str,
    approval_path: Path | None,
    secure_root: Path | None,
    environment_attestation_path: Path | None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if not execute:
        raise GuardFailure("Execution is disabled by default; --execute plus all approval gates is required")
    if approval_path is None or secure_root is None or environment_attestation_path is None:
        raise GuardFailure("Approval manifest, restricted root and environment attestation are all required")
    approval = _load_json(approval_path)
    environment_attestation = _load_json(environment_attestation_path)
    validate_approval_manifest(approval, phase)
    validate_secure_environment(secure_root, environment_attestation)
    return secure_root.resolve(), approval, environment_attestation


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _model_frame(models: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for target in FCE_TARGETS:
        for term, coefficient in models[target].items():
            rows.append({"target_predictor": target, "term": term, "coefficient": float(coefficient)})
    return pd.DataFrame(rows)


def run_paired_hospital_bootstrap(
    outcome: np.ndarray,
    reference_probability: np.ndarray,
    alternative_probability: np.ndarray,
    hospitals: np.ndarray,
    checkpoint: Path,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Governed paired wrapper: identical cluster indices, alternative-reference differences."""
    return hospital_cluster_bootstrap(
        outcome, reference_probability, hospitals, replicates, seed,
        alternative_probability, checkpoint,
    )


def missingness_burden_summary(raw: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    assert_exact_key_alignment(raw, predictions, EXTERNAL_KEYS)
    missing_count = raw[FCE_TARGETS].isna().sum(axis=1).to_numpy()
    labels = np.select(
        [missing_count == 0, missing_count == 1, missing_count == 2,
         (missing_count >= 3) & (missing_count <= 6), missing_count >= 7],
        ["0", "1", "2", "3-6", "7-14"], default="INVALID",
    )
    if (labels == "INVALID").any():
        raise GuardFailure("Invalid missing-target count while forming burden groups")
    reference = predictions["reference_probability"].to_numpy(np.float64)
    alternative = predictions["alternative_probability"].to_numpy(np.float64)
    shift = alternative - reference
    rows = []
    for label in ["0", "1", "2", "3-6", "7-14"]:
        selected = labels == label
        for method, values in [("reference", reference), ("alternative", alternative)]:
            v = values[selected]
            rows.append({
                "missing_target_group": label, "method": method, "n": int(len(v)),
                "mean_risk": float(np.mean(v)) if len(v) else np.nan,
                "sd_risk": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
                "median_risk": float(np.median(v)) if len(v) else np.nan,
                "p25_risk": float(np.quantile(v, 0.25)) if len(v) else np.nan,
                "p75_risk": float(np.quantile(v, 0.75)) if len(v) else np.nan,
                "p05_risk": float(np.quantile(v, 0.05)) if len(v) else np.nan,
                "p95_risk": float(np.quantile(v, 0.95)) if len(v) else np.nan,
                "mean_absolute_shift": float(np.mean(np.abs(shift[selected]))) if selected.any() else np.nan,
                "maximum_absolute_shift": float(np.max(np.abs(shift[selected]))) if selected.any() else np.nan,
            })
    return pd.DataFrame(rows)


def bootstrap_interval_summary(replicates: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in replicates.columns if c != "replicate"]
    rows = []
    for column in columns:
        values = replicates[column].to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise GuardFailure(f"Nonfinite bootstrap values for {column}")
        rows.append({
            "quantity": column,
            "replicates": int(len(values)),
            "mean": float(np.mean(values)),
            "percentile_2_5": float(np.quantile(values, 0.025)),
            "percentile_97_5": float(np.quantile(values, 0.975)),
        })
    return pd.DataFrame(rows)


def run_prepare(secure_root: Path, approval: dict[str, Any], environment_attestation: dict[str, Any]) -> Path:
    """Real-data pre-outcome phase. Called only after execution_gate succeeds."""
    source_hashes = verify_source_hashes(EXPECTED_PREPARE_HASHES)
    bounds = load_canonical_fce_admissible()
    coefficients = load_locked_coefficients()
    run_dir = secure_root / f"imputation_sensitivity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir.mkdir(parents=False, exist_ok=False)
    recorder = FailureRecorder(run_dir / "FAILURE.json")

    development = validate_predictor_frame(pd.read_parquet(DEVELOPMENT_MATRIX), DEVELOPMENT_KEYS, "development", recorder)
    external = validate_predictor_frame(pd.read_parquet(EXTERNAL_RAW), EXTERNAL_KEYS, "external raw", recorder)
    reference = validate_predictor_frame(pd.read_parquet(REFERENCE_COMPLETED), EXTERNAL_KEYS, "reference completed", recorder)
    assert_exact_key_alignment(external, reference, EXTERNAL_KEYS)
    reference_fce_pool = pd.read_csv(REFERENCE_FCE_COEFFICIENTS)
    reference_fce_initialization = pd.read_csv(REFERENCE_FCE_INITIALIZATION)
    reproduction_receipt, reference_reproduction_trace = authoritative_completed_predictor_reproduction_gate(
        external, reference, coefficients, reference_fce_pool,
        reference_fce_initialization, bounds, recorder,
    )
    atomic_json(reproduction_receipt, run_dir / "REFERENCE_COMPLETED_PREDICTOR_REPRODUCTION_GATE.json")
    _write_csv_atomic(reference_reproduction_trace, run_dir / "REFERENCE_FCE_REPRODUCTION_TRACE.csv")
    tolerances, tolerance_diagnostics = propose_tolerances(development)
    models, means, development_completed, development_trace = fit_predictor_only_imputer(
        development, coefficients, tolerances, bounds, recorder
    )
    alternative_completed, external_trace, alternative_risk = apply_predictor_only_imputer(
        external, models, means, coefficients, tolerances, bounds, recorder
    )
    assert_exact_key_alignment(external, alternative_completed, EXTERNAL_KEYS)
    reference_risk = locked_risk(reference, coefficients).astype(np.float64)

    predictions = external[EXTERNAL_KEYS].copy()
    predictions["reference_probability"] = reference_risk
    predictions["alternative_probability"] = alternative_risk
    predictions["alternative_minus_reference"] = alternative_risk - reference_risk
    burden_summary = missingness_burden_summary(external, predictions)

    output_hashes = {
        "alternative_completed.parquet": atomic_parquet_exact(
            alternative_completed, run_dir / "alternative_completed.parquet", PRIMARY_PREDICTORS
        ),
        "preoutcome_predictions.parquet": atomic_parquet_exact(
            predictions,
            run_dir / "preoutcome_predictions.parquet",
            ["reference_probability", "alternative_probability", "alternative_minus_reference"],
        ),
    }
    _write_csv_atomic(tolerance_diagnostics, run_dir / "TOLERANCE_PROPOSAL.csv")
    _write_csv_atomic(_model_frame(models), run_dir / "PREDICTOR_ONLY_OLS_COEFFICIENTS.csv")
    _write_csv_atomic(development_trace, run_dir / "DEVELOPMENT_CONVERGENCE_TRACE.csv")
    _write_csv_atomic(external_trace, run_dir / "EXTERNAL_CONVERGENCE_TRACE.csv")
    _write_csv_atomic(burden_summary, run_dir / "MISSINGNESS_BURDEN_PREDICTION_SUMMARY.csv")

    source_hashes_after = verify_source_hashes(EXPECTED_PREPARE_HASHES)
    if source_hashes_after != source_hashes:
        recorder.fail("PROTECTED_SOURCE_CHANGED", "A governed source or fixed artifact changed during preparation")
    manifest = {
        "analysis_phase": "preoutcome",
        "post_hoc": True,
        "outcomes_accessed": False,
        "model_refit": False,
        "mortality_model_changed": False,
        "recalibration": False,
        "performance_not_computed": True,
        "constant_prevalence_comparator_used": False,
        "predictor_only_conditional_models": True,
        "tolerance_policy_id": TOLERANCE_POLICY_ID,
        "tolerance_formula": TOLERANCE_FORMULA,
        "tolerance_approval_status": "APPROVED_BY_MANIFEST_FOR_EXECUTION",
        "risk_tolerance": RISK_TOLERANCE,
        "max_cycles": MAX_CYCLES,
        "canonical_bounds_hash": canonical_json_hash(bounds),
        "reference_completed_predictor_reproduction": reproduction_receipt,
        "source_hashes": source_hashes,
        "source_hashes_postflight": source_hashes_after,
        "approval_document_sha256": sha256(Path(approval["approval_document_path"])),
        "approval_summary": approval,
        "secure_environment_attestation": environment_attestation,
        "output_hashes": output_hashes,
        "created_utc": utcnow(),
    }
    atomic_json(manifest, run_dir / "PREOUTCOME_LOCK_MANIFEST.json")
    return run_dir


def run_evaluate(secure_root: Path, approval: dict[str, Any], _: dict[str, Any]) -> Path:
    """Outcome phase. Impossible without separate outcome and lock approvals."""
    source_hashes = verify_source_hashes({**EXPECTED_PREPARE_HASHES, **EXPECTED_EVALUATION_HASHES})
    lock_manifest_path = Path(approval["preoutcome_lock_manifest_path"])
    if sha256(lock_manifest_path) != approval["preoutcome_lock_manifest_sha256"]:
        raise GuardFailure("Approved pre-outcome lock manifest hash mismatch")
    lock = _load_json(lock_manifest_path)
    if lock.get("outcomes_accessed") is not False or lock.get("performance_not_computed") is not True:
        raise GuardFailure("Approved manifest does not describe an outcome-blind analysis phase")
    run_dir = lock_manifest_path.parent
    try:
        run_dir.resolve().relative_to(secure_root.resolve())
    except ValueError as exc:
        raise GuardFailure("Approved pre-outcome lock is outside the approved restricted root") from exc
    predictions_path = run_dir / "preoutcome_predictions.parquet"
    if sha256(predictions_path) != lock.get("output_hashes", {}).get("preoutcome_predictions.parquet"):
        raise GuardFailure("Fixed pre-outcome prediction hash mismatch")
    predictions = pd.read_parquet(predictions_path)
    outcomes = pd.read_parquet(OUTCOME_EVALUABLE)
    required_outcome_columns = EXTERNAL_KEYS + ["hospital_expire_flag"]
    if not set(required_outcome_columns).issubset(outcomes.columns):
        raise GuardFailure("Outcome attachment required columns/order are absent")
    merged = predictions.merge(outcomes[required_outcome_columns], on=EXTERNAL_KEYS, how="inner", validate="1:1", sort=False)
    if len(merged) != 1897 or int(merged["hospital_expire_flag"].sum()) != 433 or merged["hospital_id"].nunique() != 123:
        raise GuardFailure("Locked evaluable N/events/hospital identity failed")
    y = merged["hospital_expire_flag"].to_numpy(np.float64)
    reference = merged["reference_probability"].to_numpy(np.float64)
    alternative = merged["alternative_probability"].to_numpy(np.float64)
    checkpoint = run_dir / "PAIRED_HOSPITAL_BOOTSTRAP_CHECKPOINT.csv"
    replicates = run_paired_hospital_bootstrap(
        y, reference, alternative, merged["hospital_id"].to_numpy(), checkpoint,
    )
    point_reference = scalar_metrics(y, reference)
    point_alternative = scalar_metrics(y, alternative)
    locked_result = _load_json(REFERENCE_PERFORMANCE)
    locked_expected = {name: float(value["point"]) for name, value in locked_result["metrics"].items()}
    reproduction_difference = {name: abs(point_reference[name] - locked_expected[name]) for name in locked_expected}
    if any(value > 5e-12 for value in reproduction_difference.values()):
        raise GuardFailure(f"Reference scalar reproduction failed at 5e-12: {reproduction_difference}")
    if abs(float(reference.mean()) - float(locked_result["mean_locked_predicted_risk"])) > 5e-12:
        raise GuardFailure("Reference mean-risk reproduction failed at 5e-12")
    point = {
        name: {
            "reference": point_reference[name],
            "alternative": point_alternative[name],
            "alternative_minus_reference": point_alternative[name] - point_reference[name],
        }
        for name in point_reference
    }
    interval_summary = bootstrap_interval_summary(replicates)
    _write_csv_atomic(interval_summary, run_dir / "PAIRED_BOOTSTRAP_INTERVAL_SUMMARY.csv")
    source_hashes_after = verify_source_hashes({**EXPECTED_PREPARE_HASHES, **EXPECTED_EVALUATION_HASHES})
    if source_hashes_after != source_hashes:
        raise GuardFailure("A governed source or fixed artifact changed during evaluation")
    atomic_json({
        "status": "COMPLETE",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "point_metrics": point,
        "reference_reproduction_absolute_differences": reproduction_difference,
        "reference_reproduction_tolerance": 5e-12,
        "bootstrap_sha256": sha256(checkpoint),
        "source_hashes": source_hashes,
        "source_hashes_postflight": source_hashes_after,
        "created_utc": utcnow(),
    }, run_dir / "EVALUATION_MANIFEST.json")
    if len(replicates) != BOOTSTRAP_REPLICATES:
        raise GuardFailure("Paired hospital-cluster bootstrap incomplete")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "evaluate"], default="prepare")
    parser.add_argument("--execute", action="store_true", help="Still requires all approval/environment gates")
    parser.add_argument("--approval-manifest", type=Path)
    parser.add_argument("--secure-root", type=Path, default=Path(os.environ["MIMIC_EXTERNAL_IMPUTATION_RESTRICTED_ROOT"]) if "MIMIC_EXTERNAL_IMPUTATION_RESTRICTED_ROOT" in os.environ else None)
    parser.add_argument("--environment-attestation", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        secure_root, approval, environment_attestation = execution_gate(
            args.execute, args.phase, args.approval_manifest, args.secure_root, args.environment_attestation
        )
        result = run_prepare(secure_root, approval, environment_attestation) if args.phase == "prepare" else run_evaluate(secure_root, approval, environment_attestation)
    except GuardFailure as exc:
        print(f"Analysis stopped: {exc}", file=sys.stderr)
        return 2
    print(f"Analysis records written to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
