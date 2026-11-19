#!/usr/bin/env python3
"""Independent, privacy-preserving verifier for the fixed eICU DCA.

This file does not import or call decision_curve_analysis.py. It independently
reconstructs the locked probabilities, parses the governed threshold policy,
computes decision-curve estimands, reproduces the shared hospital bootstrap, and
compares results with the restricted internal aggregates entirely in memory.

Sensitive data are read only when --run-sensitive-reproduction is explicitly supplied.
No files are written. Console output is limited to aggregate PASS / FAIL /
INPUTS_REQUIRED statuses and never contains threshold-specific results, confidence
limits, counts, patient or hospital identifiers, or bootstrap samples.

Exit codes: 0=PASS, 1=INPUTS_REQUIRED, 2=FAIL.
A PASS verifies the fixed restricted calculations; it does not authorize release.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import brentq
from scipy.special import expit, logit
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from helpers._study_records import (
    OUTCOME_MANIFEST,
    OUTCOME_MANIFEST_SHA256,
    PERFORMANCE_MANIFEST,
    PERFORMANCE_MANIFEST_SHA256,
    add_manifest_arguments,
    manifest_paths,
    verify_study_file,
)



EXPECTED_N = 1897
EXPECTED_EVENTS = 433
EXPECTED_HOSPITALS = 123
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261011
MINIMUM_VALID = 1980

SCALES = ("raw", "standardized")
ESTIMANDS = (
    "model",
    "treat_all",
    "treat_none",
    "model_minus_treat_none",
    "model_minus_treat_all",
)

PATHS = {
    "evaluable": Path("outputs/08_external_transport/eicu_outcome_attachment/eicu_external_validation_evaluable.parquet"),
    "coefficients": Path("outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_locked_coefficient_vector.csv"),
    "primary_performance": Path("outputs/08_external_transport/eicu_locked_external_validation/EICU_LOCKED_PRIMARY_PERFORMANCE.json"),
    "outcome_lock": OUTCOME_MANIFEST,
    "performance_lock": PERFORMANCE_MANIFEST,
    "governance": Path("DCA_SOFA_FEASIBILITY_AND_ANALYSIS_PLAN.md"),
    "original_source": Path("scripts/evaluation/decision_curve/decision_curve_analysis.py"),
    "synthetic_tests": Path("evaluation/decision_curve/EICU_DCA_SYNTHETIC_TESTS.py"),
    "disclosure_review": Path("evaluation/decision_curve/EICU_DCA_MANUSCRIPT_REPORTING_DISCLOSURE_REVIEW.md"),
    "result_manifest": Path("work/authorized_results/eicu_dca/restricted_internal/DCA_ANALYSIS_MANIFEST.json"),
    "point_estimates": Path("work/authorized_results/eicu_dca/restricted_internal/DCA_POINT_ESTIMATES.csv"),
    "pointwise_intervals": Path("work/authorized_results/eicu_dca/restricted_internal/DCA_POINTWISE_INTERVALS.csv"),
    "bootstrap_accounting": Path("work/authorized_results/eicu_dca/restricted_internal/DCA_BOOTSTRAP_ACCOUNTING.json"),
    "external_status": Path("work/authorized_results/eicu_dca_external_status/DCA_EXTERNAL_RELEASE_STATUS.json"),
}




PINNED_HASHES = {
    "evaluable": "3971f51cd11e83ad85b38fcd7a4cf7e3e5943aca52e668911f61783b08052e8a",
    "coefficients": "3773d8ce03655e014e0e0ea578ba85888470ae248070552b089fd94c27afb513",
    "primary_performance": "96a0d04c7cf2ccdf3747e94837e8206fa59928d5b90198c4b5561fdfe7d600a4",
    "outcome_lock": OUTCOME_MANIFEST_SHA256,
    "performance_lock": PERFORMANCE_MANIFEST_SHA256,
    "governance": "26e5874928e66825f4302ff2f1087518eac9136680dffe8bca10b84c9b746d1d",
    "original_source": "d8751a94fc102871b40a1494b12070e1a0353667323e9dca8d1e8f18dd7538f1",
    "synthetic_tests": "c5e152f029a1bbd0776705083e3fb4efa2e9782063022abcbef9cac8ba7c3157",
    "disclosure_review": "0d48ab77c763af9932d50932e3662786b1652fa2093c0aefcf9a77d9e973b28a",
    "result_manifest": "b2071fafb5d6db22672c2efbbe9e4dbfad2e902c1d784a2805be1e593694cb79",
    "point_estimates": "d362f046ac17bc9d316024bc2bb20b20b6925e52182d4f2d35c225dc203543fa",
    "pointwise_intervals": "038e39cf73e29ab1abf2fbb2e467d316db20687c3b2aa07ce9180816f4c8e274",
    "bootstrap_accounting": "0362697d1d9715b47a8dce50c44dc85a92b4b40191c54f0727258469323d9b5b",
    "external_status": "b3d8dd5b85c9677b8f3da7571e68d7111efd30ea9f7466062160006d7ea33e7c",
}

TOL_EXACT = 1e-12
TOL_CALIBRATION = 5e-8
POLICY_PATTERN = re.compile(
    r"<!--\s*EICU_DCA_POLICY\s*(\{.*?\})\s*EICU_DCA_POLICY\s*-->", re.DOTALL
)


class VerificationFailure(RuntimeError):
    pass


class VerificationUnavailable(RuntimeError):
    pass


@dataclass
class Check:
    name: str
    status: str
    detail: str


class Recorder:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, name: str, status: str, detail: str) -> None:
        if status not in {"PASS", "FAIL", "INPUTS_REQUIRED"}:
            raise ValueError(status)
        self.checks.append(Check(name, status, detail))

    def require(self, condition: bool, name: str, pass_detail: str) -> None:
        if not condition:
            self.add(name, "FAIL", "restricted verification criterion not met")
            raise VerificationFailure(name)
        self.add(name, "PASS", pass_detail)

    def final_status(self) -> str:
        if any(item.status == "FAIL" for item in self.checks):
            return "FAIL"
        if any(item.status == "INPUTS_REQUIRED" for item in self.checks):
            return "INPUTS_REQUIRED"
        return "PASS"

    def print_summary(self, elapsed_seconds: float) -> None:
        print("\nINDEPENDENT eICU DCA NUMERICAL VERIFICATION")
        print("=" * 50)
        for item in self.checks:
            print(f"[{item.status}] {item.name}: {item.detail}")
        print("-" * 50)
        print(f"OVERALL: {self.final_status()}")
        print(f"Elapsed seconds: {elapsed_seconds:.1f}")
        print("Access: threshold-level results remain restricted.")
        print("Disclosure: status-only output; no restricted numerical values or row-level material were written.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare_scalar(actual: float, recorded: float, tolerance: float, label: str) -> None:
    if not np.isfinite(actual) or not np.isfinite(recorded):
        raise VerificationFailure(f"{label} nonfinite")
    if abs(float(actual) - float(recorded)) > tolerance:
        raise VerificationFailure(f"{label} disagreement")


def compare_array(actual, recorded, tolerance: float, label: str) -> None:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(recorded, dtype=np.float64)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise VerificationFailure(f"{label} structure")
    if left.size and float(np.max(np.abs(left - right))) > tolerance:
        raise VerificationFailure(f"{label} disagreement")


def sorted_newline_digest(values, *, numeric: bool) -> str:
    if numeric:
        canonical = sorted(int(value) for value in values)
    else:
        canonical = sorted(str(value).strip() for value in values)
    if len(canonical) != len(set(canonical)):
        raise VerificationFailure("identity uniqueness")
    payload = "".join(f"{value}\n" for value in canonical).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_coefficients(path: Path) -> tuple[list[str], float, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or set(rows[0]) != {"term", "locked_coefficient"}:
        raise VerificationFailure("coefficient schema")
    names = [row["term"] for row in rows]
    values = np.asarray([float(row["locked_coefficient"]) for row in rows], dtype=np.float64)
    if names[0] != "const" or len(set(names)) != 20 or not np.isfinite(values).all():
        raise VerificationFailure("coefficient integrity")
    return names[1:], float(values[0]), values[1:]


def auc_rank(outcome: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(outcome, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    positive = y == 1
    n1 = int(positive.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        raise VerificationUnavailable("AUROC requires two outcome classes")
    ranks = rankdata(p, method="average")
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def calibration_intercept(outcome: np.ndarray, eta: np.ndarray) -> float:
    y = np.asarray(outcome, dtype=np.float64)
    z = np.asarray(eta, dtype=np.float64)
    score = lambda alpha: float(np.sum(expit(z + alpha) - y))
    if score(-100.0) * score(100.0) > 0:
        raise VerificationUnavailable("calibration intercept root unavailable")
    return float(brentq(score, -100.0, 100.0, xtol=1e-13, rtol=1e-14, maxiter=200))


def calibration_slope(outcome: np.ndarray, eta: np.ndarray) -> float:
    y = np.asarray(outcome, dtype=np.float64)
    x = np.column_stack([np.ones(len(y)), np.asarray(eta, dtype=np.float64)])
    theta = np.array([0.0, 1.0], dtype=np.float64)
    for _ in range(120):
        linear = x @ theta
        fitted = expit(linear)
        gradient = x.T @ (fitted - y)
        weights = fitted * (1.0 - fitted)
        try:
            step = np.linalg.solve(x.T @ (weights[:, None] * x), gradient)
        except np.linalg.LinAlgError as exc:
            raise VerificationUnavailable("calibration slope Hessian unavailable") from exc
        base = float(np.sum(np.logaddexp(0.0, linear) - y * linear))
        scale = 1.0
        accepted = False
        for _ in range(45):
            candidate = theta - scale * step
            candidate_linear = x @ candidate
            loss = float(np.sum(np.logaddexp(0.0, candidate_linear) - y * candidate_linear))
            if np.isfinite(loss) and loss <= base + 1e-12:
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            raise VerificationUnavailable("calibration slope line search unavailable")
        theta = candidate
        if np.max(np.abs(scale * step)) < 1e-11 or np.max(np.abs(gradient)) < 1e-9:
            return float(theta[1])
    raise VerificationUnavailable("calibration slope did not converge")


def primary_integrity_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(outcome, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise VerificationFailure("outcome/prediction alignment")
    if not np.isfinite(p).all() or np.any((p <= 0.0) | (p >= 1.0)):
        raise VerificationFailure("locked probability validity")
    eta = logit(p)
    expected = float(np.sum(p))
    return {
        "auroc": auc_rank(y, p),
        "brier": float(np.mean(np.square(y - p))),
        "calibration_intercept": calibration_intercept(y, eta),
        "calibration_slope": calibration_slope(y, eta),
        "observed_over_expected": float(np.sum(y) / expected),
        "expected_event_count": expected,
        "mean_locked_predicted_risk": float(np.mean(p)),
    }


def canonical_thresholds(values, label: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise VerificationFailure(f"{label} policy structure")
    output: list[float] = []
    previous: Decimal | None = None
    for value in values:
        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise VerificationFailure(f"{label} policy value") from exc
        if not decimal_value.is_finite() or decimal_value <= 0 or decimal_value >= 1:
            raise VerificationFailure(f"{label} policy range")
        hundred = decimal_value * Decimal(100)
        if hundred != hundred.to_integral_value():
            raise VerificationFailure(f"{label} policy increment")
        canonical = hundred / Decimal(100)
        if previous is not None and canonical <= previous:
            raise VerificationFailure(f"{label} policy ordering")
        previous = canonical
        output.append(float(canonical))
    if len(output) != len(set(output)):
        raise VerificationFailure(f"{label} policy duplicates")
    return tuple(output)


def load_policy(path: Path) -> tuple[dict, tuple[float, ...], tuple[float, ...], np.ndarray]:
    matches = POLICY_PATTERN.findall(path.read_text(encoding="utf-8-sig"))
    if len(matches) != 1:
        raise VerificationFailure("governance policy block")
    policy = json.loads(matches[0])
    required = {
        "status",
        "action",
        "primary_thresholds",
        "sensitivity_thresholds",
        "threshold_rationale",
        "true_positive_tradeoff",
        "false_positive_tradeoff",
    }
    if set(policy) != required or policy["status"] != "FINALIZED_BEFORE_DCA":
        raise VerificationFailure("governance policy status")
    for field in ("action", "threshold_rationale", "true_positive_tradeoff", "false_positive_tradeoff"):
        if not isinstance(policy[field], str) or not policy[field].strip():
            raise VerificationFailure("governance policy rationale")
    primary = canonical_thresholds(policy["primary_thresholds"], "primary")
    sensitivity = canonical_thresholds(policy["sensitivity_thresholds"], "sensitivity")
    union = np.asarray(sorted(set(primary) | set(sensitivity)), dtype=np.float64)
    return policy, primary, sensitivity, union


def threshold_labels(grid: np.ndarray, primary: tuple[float, ...], sensitivity: tuple[float, ...]) -> list[str]:
    primary_set = set(primary)
    sensitivity_set = set(sensitivity)
    labels = []
    for threshold in grid:
        value = float(threshold)
        in_primary = value in primary_set
        in_sensitivity = value in sensitivity_set
        if in_primary and in_sensitivity:
            labels.append("PRIMARY_AND_SENSITIVITY")
        elif in_primary:
            labels.append("PRIMARY")
        elif in_sensitivity:
            labels.append("SENSITIVITY_ONLY")
        else:
            raise VerificationFailure("threshold-set membership")
    return labels


def dca_arrays(outcome: np.ndarray, probability: np.ndarray, grid: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    y = np.asarray(outcome, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    n = len(y)
    if n == 0 or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise VerificationFailure("DCA vector alignment")
    action = p[:, None] >= grid[None, :]
    tp = np.sum(action & (y[:, None] == 1), axis=0, dtype=np.int64)
    fp = np.sum(action & (y[:, None] == 0), axis=0, dtype=np.int64)
    prevalence = float(np.mean(y))
    weights = grid / (1.0 - grid)
    model = tp / n - fp / n * weights
    treat_all = prevalence - (1.0 - prevalence) * weights
    treat_none = np.zeros_like(grid)
    raw = {
        "model": model,
        "treat_all": treat_all,
        "treat_none": treat_none,
        "model_minus_treat_none": model - treat_none,
        "model_minus_treat_all": model - treat_all,
    }
    standardized = {
        name: values / prevalence if prevalence > 0 else np.full_like(values, np.nan)
        for name, values in raw.items()
    }
    counts = {
        "action_positive": np.sum(action, axis=0, dtype=np.int64),
        "true_positive": tp,
        "false_positive": fp,
    }
    return {"raw": raw, "standardized": standardized}, counts


def bootstrap_dca(outcome: np.ndarray, probability: np.ndarray, sites: np.ndarray, grid: np.ndarray) -> tuple[dict, dict]:
    hospitals = np.unique(sites)
    if len(hospitals) != EXPECTED_HOSPITALS:
        raise VerificationFailure("bootstrap hospital count")
    groups = [np.flatnonzero(sites == hospital) for hospital in hospitals]
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    samples = {
        (scale, estimand): np.full((BOOTSTRAP_REPETITIONS, len(grid)), np.nan, dtype=np.float64)
        for scale in SCALES
        for estimand in ESTIMANDS
    }
    sample_sizes = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.int64)
    event_counts = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.int64)
    distinct_hospitals = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.int64)
    zero_event_repetitions = 0
    for repetition in range(BOOTSTRAP_REPETITIONS):
        chosen = rng.integers(0, len(hospitals), size=len(hospitals), dtype=np.int64)
        index = np.concatenate([groups[int(position)] for position in chosen])
        yy = outcome[index]
        pp = probability[index]
        values, _ = dca_arrays(yy, pp, grid)
        sample_sizes[repetition] = len(index)
        event_counts[repetition] = int(np.sum(yy))
        distinct_hospitals[repetition] = int(np.unique(chosen).size)
        if event_counts[repetition] == 0:
            zero_event_repetitions += 1
        for scale in SCALES:
            for estimand in ESTIMANDS:
                samples[(scale, estimand)][repetition] = values[scale][estimand]
    accounting = {
        "sample_size_min": int(sample_sizes.min()),
        "sample_size_max": int(sample_sizes.max()),
        "sample_size_mean": float(sample_sizes.mean()),
        "events_min": int(event_counts.min()),
        "events_max": int(event_counts.max()),
        "distinct_selected_hospitals_min": int(distinct_hospitals.min()),
        "distinct_selected_hospitals_max": int(distinct_hospitals.max()),
        "zero_event_repetitions": int(zero_event_repetitions),
    }
    return samples, accounting



def authenticate_inputs(
    root: Path, recorder: Recorder, manifests: dict[Path, Path] | None = None,
) -> dict[str, Path]:
    """Verify the specified study inputs against their recorded digests."""
    resolved: dict[str, Path] = {}
    for key, relative in PATHS.items():
        try:
            path = root / (manifests or {}).get(relative, relative)
            resolved[key] = verify_study_file(path, PINNED_HASHES[key])
        except FileNotFoundError as exc:
            recorder.require(False, f"input::{key}", str(exc))
        except ValueError as exc:
            recorder.require(False, f"input integrity::{key}", str(exc))
    return resolved


def verify_hashes(
    root: Path, recorder: Recorder, manifests: dict[Path, Path] | None = None,
) -> dict[str, Path]:
    return authenticate_inputs(root, recorder, manifests)


def verify_restricted_results(
    root: Path,
    recorder: Recorder,
    outcome: np.ndarray,
    probability: np.ndarray,
    grid: np.ndarray,
    primary: tuple[float, ...],
    sensitivity: tuple[float, ...],
    samples: dict,
    derived_accounting: dict,
) -> None:
    point = pd.read_csv(root / PATHS["point_estimates"])
    intervals = pd.read_csv(root / PATHS["pointwise_intervals"])
    accounting = read_json(root / PATHS["bootstrap_accounting"])
    labels = threshold_labels(grid, primary, sensitivity)
    point_values, counts = dca_arrays(outcome, probability, grid)

    recorder.require(len(point) == len(grid), "restricted point-table shape", "one row per governed unique threshold")
    compare_array(point["threshold"], grid, TOL_EXACT, "restricted threshold grid")
    recorder.require(point["threshold_set"].astype(str).tolist() == labels, "restricted threshold provenance", "primary/sensitivity labels agree")
    recorder.require(point["n"].eq(EXPECTED_N).all(), "restricted point N", "fixed N retained at every threshold")
    recorder.require(point["events"].eq(EXPECTED_EVENTS).all(), "restricted point events", "fixed event count retained")
    recorder.require(point["non_events"].eq(EXPECTED_N - EXPECTED_EVENTS).all(), "restricted point non-events", "fixed non-event count retained")
    recorder.require(np.array_equal(point["model_action_positive"].to_numpy(dtype=np.int64), counts["action_positive"]), "restricted action counts", "all threshold cells agree internally")
    recorder.require(np.array_equal(point["model_true_positive"].to_numpy(dtype=np.int64), counts["true_positive"]), "restricted true-positive counts", "all threshold cells agree internally")
    recorder.require(np.array_equal(point["model_false_positive"].to_numpy(dtype=np.int64), counts["false_positive"]), "restricted false-positive counts", "all threshold cells agree internally")
    recorder.require(point["standardized_status"].eq("ESTIMABLE").all(), "restricted standardized status", "all point estimates estimable")
    for scale in SCALES:
        for estimand in ESTIMANDS:
            compare_array(
                point_values[scale][estimand],
                point[f"nb_{scale}_{estimand}"],
                TOL_EXACT,
                "restricted point estimates",
            )
    recorder.add("restricted pointwise calculations", "PASS", "all formulas, strategies, contrasts, counts, and governed thresholds agree")

    expected_rows = len(grid) * len(SCALES) * len(ESTIMANDS)
    recorder.require(len(intervals) == expected_rows, "restricted interval-table shape", "complete scale/estimand/threshold Cartesian product")
    interval_lookup = intervals.set_index(["scale", "estimand", "threshold"], verify_integrity=True)
    for scale in SCALES:
        for estimand in ESTIMANDS:
            matrix = samples[(scale, estimand)]
            for column_index, threshold in enumerate(grid):
                row = interval_lookup.loc[(scale, estimand, float(threshold))]
                valid = matrix[:, column_index]
                valid = valid[np.isfinite(valid)]
                if len(valid) < MINIMUM_VALID:
                    raise VerificationUnavailable("bootstrap validity threshold not met")
                low, high = np.percentile(valid, [2.5, 97.5], method="linear")
                compare_scalar(point_values[scale][estimand][column_index], float(row["point_estimate"]), TOL_EXACT, "restricted interval point")
                compare_scalar(float(low), float(row["ci95_low"]), TOL_EXACT, "restricted interval lower bound")
                compare_scalar(float(high), float(row["ci95_high"]), TOL_EXACT, "restricted interval upper bound")
                if int(row["valid_repetitions"]) != len(valid) or int(row["invalid_repetitions"]) != BOOTSTRAP_REPETITIONS - len(valid):
                    raise VerificationFailure("restricted bootstrap validity counts")
                if str(row["release_status"]) != "PASS":
                    raise VerificationFailure("restricted interval release status")
                expected_label = labels[int(np.flatnonzero(np.isclose(grid, threshold, atol=0.0, rtol=0.0))[0])]
                if str(row["threshold_set"]) != expected_label:
                    raise VerificationFailure("restricted interval threshold provenance")
    recorder.add("restricted bootstrap intervals", "PASS", "all pointwise linear-percentile intervals and validity counts agree")

    fixed_accounting = {
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_rng": "numpy.random.Generator(PCG64)",
        "bootstrap_unit": "hospital",
        "observed_hospitals": EXPECTED_HOSPITALS,
        "hospital_selections_per_repetition": EXPECTED_HOSPITALS,
        "same_draw_used_for_all_estimands": True,
        "minimum_valid_required": MINIMUM_VALID,
        "failed_draws_redrawn": 0,
        "overlapping_thresholds_calculated_once": True,
        "single_shared_bootstrap_across_threshold_sets": True,
    }
    for key, expected in fixed_accounting.items():
        if accounting.get(key) != expected:
            raise VerificationFailure("restricted bootstrap accounting contract")
    for key, expected in derived_accounting.items():
        if isinstance(expected, float):
            compare_scalar(float(expected), float(accounting.get(key)), TOL_EXACT, "restricted bootstrap accounting")
        elif accounting.get(key) != expected:
            raise VerificationFailure("restricted bootstrap accounting")
    recorder.require(accounting.get("valid_repetitions_min_across_estimands_thresholds") == 2000, "minimum bootstrap validity", "2,000 valid values in every restricted cell")
    recorder.require(accounting.get("valid_repetitions_max_across_estimands_thresholds") == 2000, "maximum bootstrap validity", "2,000 valid values in every restricted cell")
    recorder.require(accounting.get("insufficient_estimand_threshold_cells") == [], "bootstrap insufficiency", "no insufficient restricted cells")
    recorder.require(accounting.get("failure_counts", {}).get("HARD_FAILURES") == 0, "bootstrap hard failures", "no hard failures or redraws")



def verify_recorded_statuses(manifest: dict, external_status: dict, recorder: Recorder) -> None:
    """Verify that the authenticated outputs respect the disclosure boundary."""
    recorder.require(manifest.get("status") in {"COMPLETE_RESTRICTED_INTERNAL_AGGREGATES", "RESTRICTED_INTERNAL_AGGREGATES_PUBLISHED"}, "restricted manifest status", "restricted results publication status confirmed")
    recorder.require(manifest.get("restricted_internal_results_only") is True, "restricted disclosure class", "threshold results remain restricted")
    recorder.require(external_status.get("threshold_level_results_included") is False, "external threshold exclusion", "status-only external package")



def run(root: Path, recorder: Recorder, manifests: dict[Path, Path] | None = None) -> None:
    resolved = verify_hashes(root, recorder, manifests)
    manifest = read_json(root / PATHS["result_manifest"])
    external_status = read_json(root / PATHS["external_status"])
    verify_recorded_statuses(manifest, external_status, recorder)

    expected_software = manifest.get("software", {})
    current_software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    software_match = all(str(current_software[key]) == str(expected_software.get(key)) for key in current_software)
    recorder.add(
        "software assumptions recorded",
        "PASS",
        "recorded execution versions match"
        if software_match
        else "version drift recorded; PASS still requires complete numerical agreement",
    )

    outcome_lock = read_json(resolved["outcome_lock"])
    policy, primary, sensitivity, grid = load_policy(root / PATHS["governance"])
    recorder.require(tuple(manifest.get("primary_thresholds", [])) == primary, "primary grid provenance", "restricted manifest matches governed primary grid")
    recorder.require(tuple(manifest.get("sensitivity_thresholds", [])) == sensitivity, "sensitivity grid provenance", "restricted manifest matches governed sensitivity grid")
    recorder.require(manifest.get("thresholds_selected_from_observed_dca") is False, "outcome-independent grid record", "grid was not selected from DCA results")
    recorder.require(manifest.get("clinical_policy_status") == policy["status"], "clinical policy provenance", "finalized governance status agrees")

    predictors, intercept, beta = load_coefficients(root / PATHS["coefficients"])
    columns = [
        "unique_pid",
        "hstay_id",
        "stay_id",
        "hospital_id",
        "outcome_evaluable",
        "hospital_expire_flag",
        *predictors,
    ]
    frame = pd.read_parquet(root / PATHS["evaluable"], columns=columns)
    recorder.require(len(frame) == EXPECTED_N, "fixed cohort size", "N=1,897")
    recorder.require(frame["stay_id"].is_unique and frame["unique_pid"].is_unique, "fixed cohort uniqueness", "one selected row per patient/stay")
    recorder.require(frame["hospital_id"].nunique() == EXPECTED_HOSPITALS, "fixed hospital count", "123 hospitals")
    recorder.require(frame["outcome_evaluable"].eq(1).all(), "fixed outcome availability", "all reproduction rows outcome-evaluable")
    outcome = pd.to_numeric(frame["hospital_expire_flag"], errors="raise").to_numpy(dtype=np.int8)
    recorder.require(np.isin(outcome, [0, 1]).all() and int(np.sum(outcome)) == EXPECTED_EVENTS, "fixed outcome identity", "binary vector with 433 events")
    identity_lock = outcome_lock.get("identity_lock", {})
    recorder.require(sorted_newline_digest(frame["unique_pid"], numeric=False) == identity_lock.get("evaluable_uniquePid_sha256"), "unique-patient identity digest", "matches fixed evaluable identity")
    recorder.require(sorted_newline_digest(frame["hstay_id"], numeric=True) == identity_lock.get("evaluable_hstay_id_sha256"), "hospital-stay identity digest", "matches fixed evaluable identity")
    recorder.require(sorted_newline_digest(frame["stay_id"], numeric=True) == identity_lock.get("evaluable_stay_id_sha256"), "unit-stay identity digest", "matches fixed evaluable identity")

    matrix = frame[predictors].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise VerificationFailure("fixed predictor validity")
    probability = expit(intercept + matrix @ beta)
    metrics = primary_integrity_metrics(outcome, probability)
    fixed_performance = read_json(root / PATHS["primary_performance"])
    for metric in ("auroc", "brier", "calibration_intercept", "calibration_slope", "observed_over_expected"):
        tolerance = TOL_CALIBRATION if metric.startswith("calibration") else TOL_EXACT
        compare_scalar(metrics[metric], fixed_performance["metrics"][metric]["point"], tolerance, "primary evaluation integrity")
    compare_scalar(metrics["expected_event_count"], fixed_performance["expected_event_count"], TOL_EXACT, "expected-event integrity")
    compare_scalar(metrics["mean_locked_predicted_risk"], fixed_performance["mean_locked_predicted_risk"], TOL_EXACT, "mean-risk integrity")
    for key in ("auroc", "brier", "expected_event_count", "mean_locked_predicted_risk", "observed_over_expected"):
        compare_scalar(metrics[key], manifest["prediction_reproduction"][key], TOL_EXACT, "restricted manifest prediction reproduction")
    recorder.add("locked-model reconstruction", "PASS", "prediction totals, AUROC, Brier, calibration diagnostics, and O/E agree")

    samples, derived_accounting = bootstrap_dca(outcome, probability, frame["hospital_id"].to_numpy(), grid)
    recorder.add("independent hospital bootstrap", "PASS", "2,000 shared PCG64 hospital-cluster draws recomputed")
    verify_restricted_results(root, recorder, outcome, probability, grid, primary, sensitivity, samples, derived_accounting)
    recorder.add("privacy and release guard", "PASS", "verification produced no restricted outputs and did not alter release authorization")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    add_manifest_arguments(parser)
    parser.add_argument("--project-root", type=Path, default=default_root, help="Repository root; defaults to the root inferred from this script.")
    parser.add_argument(
        "--run-sensitive-reproduction",
        action="store_true",
        help="Required owner-controlled guard. Reads authorized fixed patient-level inputs locally and writes nothing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_sensitive_reproduction:
        print("INPUTS_REQUIRED: no sensitive input was opened. Owner authorization requires --run-sensitive-reproduction.")
        return 1
    recorder = Recorder()
    started = time.perf_counter()
    try:
        run(args.project_root.resolve(), recorder, manifest_paths(args))
    except VerificationUnavailable as exc:
        recorder.add("required component", "INPUTS_REQUIRED", str(exc))
    except VerificationFailure as exc:
        if not any(item.status == "FAIL" for item in recorder.checks):
            recorder.add("verification", "FAIL", str(exc))
    except Exception as exc:  # fail closed without exposing restricted context
        recorder.add("unexpected verifier error", "FAIL", type(exc).__name__)
    recorder.print_summary(time.perf_counter() - started)
    status = recorder.final_status()
    return 0 if status == "PASS" else 1 if status == "INPUTS_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
