#!/usr/bin/env python3
"""Independent local numerical verifier for the completed eICU APACHE benchmark.

This program is intentionally separate from apache_benchmark.py.
It does not import or call that program.  It independently rebuilds the common
cohort, locked-model probabilities, scalar metrics, paired contrasts, hospital
bootstrap intervals, and calibration-curve coordinates from fixed inputs.

Sensitive inputs are read only after --run-sensitive-reproduction is supplied.  The
program writes no files and prints aggregate verification statuses only.

Exit codes:
    0  PASS          all required components independently agreed
    1  INPUTS_REQUIRED a required component could not be independently evaluated
    2  FAIL          an integrity, cohort, or numerical agreement check failed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
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



EXPECTED = {
    "evaluable_n": 1897,
    "evaluable_events": 433,
    "evaluable_hospitals": 123,
    "record_available_n": 1699,
    "common_n": 1582,
    "common_events": 368,
    "common_hospitals": 115,
    "bootstrap_repetitions": 2000,
    "bootstrap_seed": 20261010,
    "minimum_valid": 1980,
    "curve_grid_n": 120,
}

MODEL_LOCKED = "Locked harmonized MIMIC Cr/RRT"
MODEL_IV = "APACHE IV"
MODEL_IVA = "APACHE IVa"
MODELS = (MODEL_LOCKED, MODEL_IV, MODEL_IVA)

PATHS = {
    "evaluable": Path("outputs/08_external_transport/eicu_outcome_attachment/eicu_external_validation_evaluable.parquet"),
    "coefficients": Path("outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/harmonized_cr_rrt_locked_coefficient_vector.csv"),
    "apache": Path("data/external/eicu-collaborative-research-database-2.0/apachePatientResult.csv.gz"),
    "locked_performance": Path("outputs/08_external_transport/eicu_locked_external_validation/EICU_LOCKED_PRIMARY_PERFORMANCE.json"),
    "outcome_lock": OUTCOME_MANIFEST,
    "performance_lock": PERFORMANCE_MANIFEST,
    "membership_lock": Path("evaluation/apache_iv/EICU_APACHE_COMMON_COHORT_MEMBERSHIP_LOCK.json"),
    "original_source": Path("scripts/evaluation/apache_iv/apache_benchmark.py"),
    "protocol": Path("evaluation/apache_iv/EICU_APACHE_BENCHMARK_PROTOCOL.md"),
    "result_manifest": Path("work/authorized_results/eicu_apache_benchmark/ANALYSIS_MANIFEST.json"),
    "main": Path("work/authorized_results/eicu_apache_benchmark/MAIN_COMPARISON_TABLE.csv"),
    "paired": Path("work/authorized_results/eicu_apache_benchmark/PAIRED_COMPARISON_TABLE.csv"),
    "secondary": Path("work/authorized_results/eicu_apache_benchmark/SECONDARY_CALIBRATION_UNCERTAINTY.csv"),
    "curves": Path("work/authorized_results/eicu_apache_benchmark/CALIBRATION_CURVE_DATA.csv"),
    "curve_figure": Path("work/authorized_results/eicu_apache_benchmark/CALIBRATION_CURVES.png"),
    "flow": Path("work/authorized_results/eicu_apache_benchmark/PATIENT_FLOW.csv"),
    "selection": Path("work/authorized_results/eicu_apache_benchmark/APACHE_AVAILABILITY_SELECTION_SUMMARY.csv"),
    "failure_summary": Path("work/authorized_results/eicu_apache_benchmark/BOOTSTRAP_FAILURE_SUMMARY.json"),
}




PINNED_HASHES = {
    "evaluable": "3971f51cd11e83ad85b38fcd7a4cf7e3e5943aca52e668911f61783b08052e8a",
    "coefficients": "3773d8ce03655e014e0e0ea578ba85888470ae248070552b089fd94c27afb513",
    "apache": "f9870a5ad58b1519cf07f51cbe2717a8bf390f74159e2310e2bac3b6bb7a070a",
    "locked_performance": "96a0d04c7cf2ccdf3747e94837e8206fa59928d5b90198c4b5561fdfe7d600a4",
    "outcome_lock": OUTCOME_MANIFEST_SHA256,
    "performance_lock": PERFORMANCE_MANIFEST_SHA256,
    "membership_lock": "adcf6d42a83364cc481b3396859bfaab46a96a9a244305816baa3156d438842a",
    "original_source": "d4308e58236dcc5499927ec321c0fe9a40332c6fa89ea448a4569356ab3a5b43",  # analysis source digest
    "protocol": "104ee4b65d9b63e75c343251b4056fa58f8a789f6a2c04f67f029fd6aae6894b",
    "result_manifest": "35331b9bc314f4529ac63e3067ddc76030d3cf3c979d74b22b12ed3302b4874b",
    "main": "64bc866f371d4fda350047fe64395939bdc6d738b14381defbddc52536e0f2ed",
    "paired": "f9687c7df7e4ebf5e15fe3672d69da4c11da79436d0a1c390e44dbbdd3fc6654",
    "secondary": "bdc72c53071da5802221b37adee15f74c1cae34f56f677b3cbf88f50f0e0a7dd",
    "curves": "6d8aa5358ccf104449b564f45fec576914b6790d1a01bd743c8ddc55dff7ecde",
    "curve_figure": "4cf16328c7c6c8587832ff5bfdadae048760796d6f9a8b990ac1a9577c553381",
    "flow": "af7fa5d7ab06810121b2749ec441116480fd81531094f9e7e53220ca321da87f",
    "selection": "82e453244eafafaa74c35ed7df95401f603d1b98573773297f2737239ae52b28",
    "failure_summary": "0684158bd4f37af1647718dd2b81bf0950980b6b2bab6c014a088cc0b7b33428",
}

EXPECTED_MEMBERSHIP_SHA256 = "a7cc19756c7831c7b1b3ceeaa35c78575562607928cc2a94df22064484ba505a"

# Exact arithmetic quantities should reproduce very tightly.  Calibration fits
# deliberately use a different solver, so their tolerances allow only negligible
# optimizer/BLAS differences, not scientifically meaningful disagreement.
TOL_EXACT = 1e-12
TOL_CALIBRATION = 5e-8
TOL_CURVE = 2e-7


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

    def require(self, condition: bool, name: str, detail: str) -> None:
        if not condition:
            self.add(name, "FAIL", detail)
            raise VerificationFailure(f"{name}: {detail}")
        self.add(name, "PASS", detail)

    def final_status(self) -> str:
        if any(c.status == "FAIL" for c in self.checks):
            return "FAIL"
        if any(c.status == "INPUTS_REQUIRED" for c in self.checks):
            return "INPUTS_REQUIRED"
        return "PASS"

    def print_summary(self, elapsed: float) -> None:
        print("\nINDEPENDENT APACHE NUMERICAL VERIFICATION")
        print("=" * 48)
        for check in self.checks:
            print(f"[{check.status}] {check.name}: {check.detail}")
        print("-" * 48)
        print(f"OVERALL: {self.final_status()}")
        print(f"Elapsed seconds: {elapsed:.1f}")
        print("Disclosure: aggregate statuses only; no identifiers, patient predictions, hospital cells, or replicate values were written.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def membership_digest(stay_ids: np.ndarray) -> str:
    values = np.asarray(stay_ids, dtype=np.int64)
    if values.ndim != 1 or len(values) == 0 or np.any(values <= 0):
        raise VerificationFailure("common membership identifiers are malformed")
    values = np.sort(values)
    if np.unique(values).size != values.size:
        raise VerificationFailure("common membership contains duplicate identifiers")
    payload = "".join(f"{int(value)}\n" for value in values).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def compare_scalar(actual: float, expected: float, tolerance: float, label: str) -> float:
    if not np.isfinite(actual) or not np.isfinite(expected):
        raise VerificationFailure(f"{label} is nonfinite")
    delta = abs(float(actual) - float(expected))
    if delta > tolerance:
        raise VerificationFailure(
            f"{label} disagreement exceeds tolerance {tolerance:.3g}; max aggregate delta={delta:.3g}"
        )
    return delta


def compare_array(actual, expected, tolerance: float, label: str) -> float:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        raise VerificationFailure(f"{label} shape differs")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise VerificationFailure(f"{label} contains nonfinite values")
    delta = float(np.max(np.abs(left - right))) if left.size else 0.0
    if delta > tolerance:
        raise VerificationFailure(
            f"{label} disagreement exceeds tolerance {tolerance:.3g}; max aggregate delta={delta:.3g}"
        )
    return delta


def auc_rank(outcome: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(outcome, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    positive = y == 1
    n1 = int(positive.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        raise VerificationUnavailable("AUROC needs both outcome classes")
    ranks = rankdata(p, method="average")
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def calibration_intercept(outcome: np.ndarray, eta: np.ndarray) -> float:
    y = np.asarray(outcome, dtype=np.float64)
    z = np.asarray(eta, dtype=np.float64)
    score = lambda alpha: float(np.sum(expit(z + alpha) - y))
    if score(-100.0) * score(100.0) > 0:
        raise VerificationUnavailable("calibration-intercept score root is not bracketed")
    return float(brentq(score, -100.0, 100.0, xtol=1e-13, rtol=1e-14, maxiter=200))


def logistic_mle(outcome: np.ndarray, design: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    """Unpenalized binomial MLE via an independent Newton/line-search solver."""
    y = np.asarray(outcome, dtype=np.float64)
    x = np.asarray(design, dtype=np.float64)
    if x.ndim != 2 or len(y) != x.shape[0] or not np.isfinite(x).all():
        raise VerificationUnavailable("calibration design is malformed")
    singular = np.linalg.svd(x, compute_uv=False)
    if singular.size == 0 or singular[-1] <= 1e-12 * singular[0]:
        raise VerificationUnavailable("calibration design is rank deficient")
    theta = np.zeros(x.shape[1], dtype=np.float64) if start is None else np.asarray(start, dtype=np.float64).copy()
    for _ in range(120):
        linear = x @ theta
        q = expit(linear)
        gradient = x.T @ (q - y)
        weights = q * (1.0 - q)
        hessian = x.T @ (weights[:, None] * x)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as exc:
            raise VerificationUnavailable("calibration Hessian is singular") from exc
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
            raise VerificationUnavailable("calibration line search did not converge")
        theta = candidate
        if np.max(np.abs(scale * step)) < 1e-11 or np.max(np.abs(gradient)) < 1e-9:
            if not np.isfinite(theta).all():
                raise VerificationUnavailable("calibration estimate is nonfinite")
            return theta
    raise VerificationUnavailable("calibration Newton solver did not converge")


def scalar_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(outcome, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise VerificationFailure("metric vectors are malformed")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise VerificationFailure("prediction vector is invalid")
    if np.any((p == 0.0) | (p == 1.0)):
        raise VerificationUnavailable("logit calibration is not estimable for boundary probabilities")
    eta = logit(p)
    slope = logistic_mle(y, np.column_stack([np.ones(len(y)), eta]), start=np.array([0.0, 1.0]))
    expected = float(np.sum(p))
    if expected <= 0:
        raise VerificationUnavailable("O/E is not estimable with zero expected events")
    return {
        "auroc": auc_rank(y, p),
        "brier": float(np.mean(np.square(y - p))),
        "calibration_intercept": calibration_intercept(y, eta),
        "calibration_slope": float(slope[1]),
        "observed_over_expected": float(np.sum(y) / expected),
        "observed_mortality": float(np.mean(y)),
        "mean_predicted_mortality": float(np.mean(p)),
        "expected_events": expected,
    }


def rcs_design(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    k = np.asarray(knots, dtype=np.float64)
    if k.shape != (4,) or not np.all(np.diff(k) > 1e-10):
        raise VerificationUnavailable("calibration spline knots are not distinct")
    span = k[-1] - k[0]
    columns = [np.ones_like(x), x]
    for j in range(2):
        column = (
            np.maximum(x - k[j], 0.0) ** 3
            - np.maximum(x - k[-2], 0.0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
            + np.maximum(x - k[-1], 0.0) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])
        ) / (span * span)
        columns.append(column)
    return np.column_stack(columns)


def curve_spec(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eta = logit(np.asarray(probability, dtype=np.float64))
    knots = np.quantile(eta, [0.05, 0.35, 0.65, 0.95])
    grid = np.linspace(float(np.quantile(eta, 0.01)), float(np.quantile(eta, 0.99)), EXPECTED["curve_grid_n"])
    return knots, grid


def calibration_curve(outcome: np.ndarray, probability: np.ndarray, knots: np.ndarray, grid: np.ndarray) -> np.ndarray:
    eta = logit(np.asarray(probability, dtype=np.float64))
    coefficients = logistic_mle(outcome, rcs_design(eta, knots))
    curve = expit(rcs_design(grid, knots) @ coefficients)
    if not np.isfinite(curve).all() or np.any((curve <= 0.0) | (curve >= 1.0)):
        raise VerificationUnavailable("calibration curve is nonfinite or saturated")
    return curve


def load_coefficients(path: Path) -> tuple[list[str], float, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or set(rows[0]) != {"term", "locked_coefficient"}:
        raise VerificationFailure("locked coefficient layout differs")
    names = [row["term"] for row in rows]
    values = np.asarray([float(row["locked_coefficient"]) for row in rows], dtype=np.float64)
    if names[0] != "const" or len(set(names)) != 20 or not np.isfinite(values).all():
        raise VerificationFailure("locked coefficients are malformed")
    return names[1:], float(values[0]), values[1:]


def load_apache_common(path: Path, fixed_ids: set[int]) -> tuple[pd.DataFrame, int]:
    columns = [
        "apachepatientresultsid",
        "patientunitstayid",
        "apacheversion",
        "predictedhospitalmortality",
    ]
    frame = pd.read_csv(path, usecols=columns, dtype="string", low_memory=False)
    if set(frame["apacheversion"].dropna().unique()) != {"IV", "IVa"}:
        raise VerificationFailure("APACHE version labels differ")
    counts = frame["apacheversion"].value_counts()
    if any(int(counts.get(version, -1)) != 148532 for version in ("IV", "IVa")):
        raise VerificationFailure("APACHE source version row counts differ")

    result_id = pd.to_numeric(frame["apachepatientresultsid"], errors="coerce")
    stay_id = pd.to_numeric(frame["patientunitstayid"], errors="coerce")
    if result_id.isna().any() or stay_id.isna().any():
        raise VerificationFailure("APACHE linkage identifiers are missing or nonnumeric")
    if not np.equal(result_id, np.floor(result_id)).all() or not np.equal(stay_id, np.floor(stay_id)).all():
        raise VerificationFailure("APACHE linkage identifiers are noninteger")
    if (result_id <= 0).any() or (stay_id <= 0).any() or result_id.duplicated().any():
        raise VerificationFailure("APACHE identifiers are nonpositive or duplicated")
    frame["stay_id"] = stay_id.astype("int64")
    if frame.duplicated(["stay_id", "apacheversion"]).any():
        raise VerificationFailure("APACHE stay/version rows are duplicated")

    matched = frame.loc[frame["stay_id"].isin(fixed_ids)].copy()
    text = matched["predictedhospitalmortality"].str.strip()
    prediction = pd.to_numeric(text, errors="coerce")
    if prediction.isna().any() or not np.isfinite(prediction.to_numpy(dtype=float)).all():
        raise VerificationFailure("study-cohort APACHE prediction is blank or nonnumeric")
    allowed = prediction.eq(-1.0) | prediction.between(0.0, 1.0, inclusive="both")
    if not allowed.all():
        raise VerificationFailure("study-cohort APACHE prediction has an unexpected value")
    matched["prediction"] = prediction.astype(float)

    record_sets = {
        version: set(matched.loc[matched["apacheversion"].eq(version), "stay_id"].astype(int))
        for version in ("IV", "IVa")
    }
    if record_sets["IV"] != record_sets["IVa"] or len(record_sets["IV"]) != EXPECTED["record_available_n"]:
        raise VerificationFailure("APACHE record-available sets or count differ")
    valid_sets = {
        version: set(
            matched.loc[
                matched["apacheversion"].eq(version)
                & matched["prediction"].between(0.0, 1.0, inclusive="both"),
                "stay_id",
            ].astype(int)
        )
        for version in ("IV", "IVa")
    }
    if valid_sets["IV"] != valid_sets["IVa"] or len(valid_sets["IV"]) != EXPECTED["common_n"]:
        raise VerificationFailure("APACHE valid common sets or count differ")
    wide = matched.pivot(index="stay_id", columns="apacheversion", values="prediction")
    common_ids = np.asarray(sorted(valid_sets["IV"]), dtype=np.int64)
    wide = wide.loc[common_ids, ["IV", "IVa"]]
    if wide.isna().any().any() or not np.isfinite(wide.to_numpy(dtype=float)).all():
        raise VerificationFailure("APACHE common prediction matrix is incomplete")
    return wide, len(record_sets["IV"])



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
) -> dict:
    resolved = authenticate_inputs(root, recorder, manifests)
    manifest = read_json(resolved["result_manifest"])
    for filename, expected_hash in manifest.get("output_hashes", {}).items():
        output_path = root / "work/authorized_results/eicu_apache_benchmark" / filename
        recorder.require(output_path.is_file(), f"result file::{filename}", "aggregate result present")
        recorder.require(sha256_file(output_path) == expected_hash, f"result hash::{filename}", "aggregate SHA-256 matches manifest")

    return manifest


def bootstrap_reproduction(
    outcome: np.ndarray,
    sites: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]], int]:
    hospitals = np.unique(sites)
    if len(hospitals) != EXPECTED["common_hospitals"]:
        raise VerificationFailure("common hospital count differs before bootstrap")
    groups = [np.flatnonzero(sites == hospital) for hospital in hospitals]
    rng = np.random.Generator(np.random.PCG64(EXPECTED["bootstrap_seed"]))

    scalar_keys = ("auroc", "brier", "calibration_intercept", "calibration_slope", "observed_over_expected")
    samples = {
        f"{model}::{metric}": np.full(EXPECTED["bootstrap_repetitions"], np.nan, dtype=np.float64)
        for model in MODELS
        for metric in scalar_keys
    }
    for comparator in (MODEL_IV, MODEL_IVA):
        samples[f"{comparator}::auroc_difference"] = np.full(EXPECTED["bootstrap_repetitions"], np.nan)
        samples[f"{comparator}::brier_difference"] = np.full(EXPECTED["bootstrap_repetitions"], np.nan)

    specifications = {model: curve_spec(predictions[model]) for model in MODELS}
    curve_samples = {
        model: np.full((EXPECTED["bootstrap_repetitions"], EXPECTED["curve_grid_n"]), np.nan)
        for model in MODELS
    }
    failed = 0
    for repetition in range(EXPECTED["bootstrap_repetitions"]):
        chosen = rng.integers(0, len(hospitals), size=len(hospitals))
        index = np.concatenate([groups[int(position)] for position in chosen])
        yy = outcome[index]
        if np.unique(yy).size != 2:
            failed += 1
            continue
        replicate: dict[str, dict[str, float]] = {}
        for model in MODELS:
            pp = predictions[model][index]
            try:
                metrics = scalar_metrics(yy, pp)
                replicate[model] = metrics
                for metric in scalar_keys:
                    samples[f"{model}::{metric}"][repetition] = metrics[metric]
                knots, grid = specifications[model]
                curve_samples[model][repetition] = calibration_curve(yy, pp, knots, grid)
            except VerificationUnavailable:
                failed += 1
        if all(model in replicate for model in MODELS):
            for comparator in (MODEL_IV, MODEL_IVA):
                samples[f"{comparator}::auroc_difference"][repetition] = (
                    replicate[MODEL_LOCKED]["auroc"] - replicate[comparator]["auroc"]
                )
                samples[f"{comparator}::brier_difference"][repetition] = (
                    replicate[comparator]["brier"] - replicate[MODEL_LOCKED]["brier"]
                )
    return samples, curve_samples, specifications, failed


def percentile(values: np.ndarray) -> tuple[float, float, int]:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if len(valid) < EXPECTED["minimum_valid"]:
        raise VerificationUnavailable("fewer than 1,980 valid bootstrap values")
    low, high = np.percentile(valid, [2.5, 97.5], method="linear")
    return float(low), float(high), int(len(valid))


def verify_aggregates(
    root: Path,
    recorder: Recorder,
    outcome: np.ndarray,
    sites: np.ndarray,
    predictions: dict[str, np.ndarray],
    point_metrics: dict[str, dict[str, float]],
    samples: dict[str, np.ndarray],
    curve_samples: dict[str, np.ndarray],
    specifications: dict[str, tuple[np.ndarray, np.ndarray]],
    bootstrap_failures: int,
) -> None:
    main = pd.read_csv(root / PATHS["main"]).set_index("model")
    secondary = pd.read_csv(root / PATHS["secondary"]).set_index(["model", "metric"])
    paired = pd.read_csv(root / PATHS["paired"]).set_index("comparator")
    curves = pd.read_csv(root / PATHS["curves"])
    failure_summary = read_json(root / PATHS["failure_summary"])

    max_exact = 0.0
    max_calibration = 0.0
    for model in MODELS:
        row = main.loc[model]
        recorder.require(int(row["n"]) == EXPECTED["common_n"], f"{model} N", "N=1,582")
        recorder.require(int(row["deaths"]) == EXPECTED["common_events"], f"{model} deaths", "deaths=368")
        recorder.require(int(row["hospitals"]) == EXPECTED["common_hospitals"], f"{model} hospitals", "hospitals=115")
        metrics = point_metrics[model]
        for metric in ("auroc", "brier", "observed_over_expected", "observed_mortality", "mean_predicted_mortality"):
            max_exact = max(max_exact, compare_scalar(metrics[metric], float(row[metric]), TOL_EXACT, f"{model} {metric}"))
        for metric in ("calibration_intercept", "calibration_slope"):
            max_calibration = max(
                max_calibration,
                compare_scalar(metrics[metric], float(row[metric]), TOL_CALIBRATION, f"{model} {metric}"),
            )
        for metric, low_col, high_col in (
            ("auroc", "auroc_ci95_low", "auroc_ci95_high"),
            ("brier", "brier_ci95_low", "brier_ci95_high"),
        ):
            low, high, valid_n = percentile(samples[f"{model}::{metric}"])
            max_exact = max(max_exact, compare_scalar(low, float(row[low_col]), TOL_EXACT, f"{model} {metric} CI low"))
            max_exact = max(max_exact, compare_scalar(high, float(row[high_col]), TOL_EXACT, f"{model} {metric} CI high"))
            recorded_n = int(failure_summary["valid_scalar_and_difference_counts"][f"{model}::{metric}"])
            recorder.require(valid_n == recorded_n == 2000, f"{model} {metric} validity", "2,000 shared-draw estimates")

        for metric in ("calibration_intercept", "calibration_slope", "observed_over_expected"):
            sec = secondary.loc[(model, metric)]
            low, high, valid_n = percentile(samples[f"{model}::{metric}"])
            tolerance = TOL_CALIBRATION if metric != "observed_over_expected" else TOL_EXACT
            max_calibration = max(max_calibration, compare_scalar(low, float(sec["ci95_low"]), tolerance, f"{model} {metric} CI low"))
            max_calibration = max(max_calibration, compare_scalar(high, float(sec["ci95_high"]), tolerance, f"{model} {metric} CI high"))
            recorder.require(valid_n == int(sec["valid_bootstrap_repetitions"]) == 2000, f"{model} {metric} validity", "2,000 shared-draw estimates")

    for comparator in (MODEL_IV, MODEL_IVA):
        row = paired.loc[comparator]
        auc_difference = point_metrics[MODEL_LOCKED]["auroc"] - point_metrics[comparator]["auroc"]
        max_exact = max(max_exact, compare_scalar(auc_difference, float(row["paired_auroc_difference_harmonized_minus_apache"]), TOL_EXACT, f"{comparator} paired AUROC difference"))
        low, high, valid_n = percentile(samples[f"{comparator}::auroc_difference"])
        max_exact = max(max_exact, compare_scalar(low, float(row["paired_auroc_difference_ci95_low"]), TOL_EXACT, f"{comparator} paired AUROC CI low"))
        max_exact = max(max_exact, compare_scalar(high, float(row["paired_auroc_difference_ci95_high"]), TOL_EXACT, f"{comparator} paired AUROC CI high"))
        brier_difference = point_metrics[comparator]["brier"] - point_metrics[MODEL_LOCKED]["brier"]
        max_exact = max(max_exact, compare_scalar(brier_difference, float(row["paired_brier_difference_apache_minus_harmonized"]), TOL_EXACT, f"{comparator} paired Brier difference"))
        b_low, b_high, b_valid_n = percentile(samples[f"{comparator}::brier_difference"])
        max_exact = max(max_exact, compare_scalar(b_low, float(row["paired_brier_difference_ci95_low"]), TOL_EXACT, f"{comparator} paired Brier CI low"))
        max_exact = max(max_exact, compare_scalar(b_high, float(row["paired_brier_difference_ci95_high"]), TOL_EXACT, f"{comparator} paired Brier CI high"))
        recorder.require(valid_n == b_valid_n == 2000, f"{comparator} paired validity", "2,000 identical-patient shared-draw contrasts")

    max_curve = 0.0
    for model in MODELS:
        rows = curves.loc[curves["model"].eq(model)].reset_index(drop=True)
        recorder.require(len(rows) == EXPECTED["curve_grid_n"], f"{model} curve rows", "120 aggregate grid points")
        knots, grid = specifications[model]
        point_curve = calibration_curve(outcome, predictions[model], knots, grid)
        valid_curves = curve_samples[model][np.isfinite(curve_samples[model]).all(axis=1)]
        if len(valid_curves) < EXPECTED["minimum_valid"]:
            raise VerificationUnavailable(f"{model} calibration curve has insufficient valid repetitions")
        low, high = np.percentile(valid_curves, [2.5, 97.5], axis=0, method="linear")
        max_curve = max(max_curve, compare_array(expit(grid), rows["predicted_probability"], TOL_EXACT, f"{model} curve grid"))
        max_curve = max(max_curve, compare_array(point_curve, rows["fitted_observed_probability"], TOL_CURVE, f"{model} point curve"))
        max_curve = max(max_curve, compare_array(low, rows["pointwise_ci95_low"], TOL_CURVE, f"{model} curve CI low"))
        max_curve = max(max_curve, compare_array(high, rows["pointwise_ci95_high"], TOL_CURVE, f"{model} curve CI high"))
        recorded_curve_n = int(failure_summary["valid_complete_curve_counts"][model])
        recorder.require(len(valid_curves) == recorded_curve_n == 2000, f"{model} curve validity", "2,000 complete curves")

    recorder.require(bootstrap_failures == 0, "independent bootstrap failures", "no independent scalar/curve failures")
    recorder.require(failure_summary.get("failure_events") == [], "recorded bootstrap failures", "original record reports no failures or redraws")
    recorder.add("scalar numerical agreement", "PASS", f"maximum exact-metric/interval delta <= {max_exact:.3g}")
    recorder.add("calibration numerical agreement", "PASS", f"maximum independent-solver delta <= {max_calibration:.3g}")
    recorder.add("calibration-curve numerical agreement", "PASS", f"maximum coordinate/band delta <= {max_curve:.3g}")


def run(root: Path, recorder: Recorder, manifests: dict[Path, Path] | None = None) -> None:
    manifest = verify_hashes(root, recorder, manifests)

    expected_versions = manifest.get("software", {})
    current_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    version_match = all(str(current_versions[key]) == str(expected_versions.get(key)) for key in current_versions)
    recorder.add(
        "software assumptions recorded",
        "PASS",
        "matches recorded Python/NumPy/pandas/SciPy versions"
        if version_match
        else "version drift detected; the reproduction must earn PASS through numerical agreement rather than environment identity",
    )

    predictors, intercept, beta = load_coefficients(root / PATHS["coefficients"])
    required_columns = ["stay_id", "hospital_id", "hospital_expire_flag", *predictors]
    evaluable = pd.read_parquet(root / PATHS["evaluable"], columns=required_columns)
    recorder.require(len(evaluable) == EXPECTED["evaluable_n"], "fixed evaluable N", "N=1,897")
    recorder.require(evaluable["stay_id"].is_unique, "fixed identity uniqueness", "one row per stay")
    y_full = pd.to_numeric(evaluable["hospital_expire_flag"], errors="raise").to_numpy(dtype=np.int8)
    recorder.require(np.isin(y_full, [0, 1]).all() and int(y_full.sum()) == EXPECTED["evaluable_events"], "fixed outcome vector", "binary with 433 events")
    recorder.require(evaluable["hospital_id"].nunique() == EXPECTED["evaluable_hospitals"], "fixed evaluable hospitals", "123 hospitals")

    matrix = evaluable[predictors].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise VerificationFailure("locked predictor matrix contains nonfinite values")
    locked_full = expit(intercept + matrix @ beta)
    locked_full_metrics = scalar_metrics(y_full, locked_full)
    original_performance = read_json(root / PATHS["locked_performance"])
    for metric in ("auroc", "brier", "calibration_intercept", "calibration_slope", "observed_over_expected"):
        tolerance = TOL_CALIBRATION if metric.startswith("calibration") else TOL_EXACT
        compare_scalar(locked_full_metrics[metric], original_performance["metrics"][metric]["point"], tolerance, f"full locked reproduction {metric}")
    compare_scalar(float(np.sum(locked_full)), float(original_performance["expected_event_count"]), TOL_EXACT, "full locked expected events")
    compare_scalar(float(np.mean(locked_full)), float(original_performance["mean_locked_predicted_risk"]), TOL_EXACT, "full locked mean risk")
    recorder.add("full locked-model integrity reproduction", "PASS", "all five metrics and prediction totals agree")

    apache_wide, record_available_n = load_apache_common(root / PATHS["apache"], set(evaluable["stay_id"].astype(int)))
    recorder.require(record_available_n == EXPECTED["record_available_n"], "APACHE record availability", "N=1,699")
    common_ids = apache_wide.index.to_numpy(dtype=np.int64)
    recorder.require(membership_digest(common_ids) == EXPECTED_MEMBERSHIP_SHA256, "common membership digest", "matches independently locked SHA-256")
    lock = read_json(root / PATHS["membership_lock"])
    recorder.require(lock.get("common_cohort_membership_sha256") == EXPECTED_MEMBERSHIP_SHA256, "membership lock contents", "pinned digest agrees")

    common = evaluable.set_index("stay_id").loc[common_ids].reset_index()
    recorder.require(np.array_equal(common["stay_id"].to_numpy(dtype=np.int64), common_ids), "three-model row alignment", "identical sorted stay order")
    outcome = common["hospital_expire_flag"].to_numpy(dtype=np.int8)
    sites = common["hospital_id"].to_numpy()
    recorder.require(len(common) == EXPECTED["common_n"], "common cohort N", "N=1,582")
    recorder.require(int(outcome.sum()) == EXPECTED["common_events"], "common cohort deaths", "368 deaths")
    recorder.require(np.unique(sites).size == EXPECTED["common_hospitals"], "common cohort hospitals", "115 hospitals")

    locked_common = expit(intercept + common[predictors].to_numpy(dtype=np.float64) @ beta)
    predictions = {
        MODEL_LOCKED: locked_common,
        MODEL_IV: apache_wide["IV"].to_numpy(dtype=np.float64),
        MODEL_IVA: apache_wide["IVa"].to_numpy(dtype=np.float64),
    }
    for model, probability in predictions.items():
        recorder.require(
            probability.shape == outcome.shape and np.isfinite(probability).all() and np.all((probability >= 0.0) & (probability <= 1.0)),
            f"{model} predictions",
            "one finite unchanged probability per aligned patient",
        )
    recorder.add("outcome identity across models", "PASS", "one fixed outcome vector used for all three aligned prediction vectors")

    point_metrics = {model: scalar_metrics(outcome, probability) for model, probability in predictions.items()}
    recorder.add("independent point estimators", "PASS", "AUROC, Brier, intercept, slope, and O/E recomputed without original metric functions")

    samples, curve_samples, specifications, failed = bootstrap_reproduction(outcome, sites, predictions)
    recorder.add("shared hospital bootstrap", "PASS", "2,000 PCG64 draws; one 115-hospital draw reused for all models, contrasts, and curves")
    verify_aggregates(root, recorder, outcome, sites, predictions, point_metrics, samples, curve_samples, specifications, failed)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    add_manifest_arguments(parser)
    parser.add_argument("--project-root", type=Path, default=default_root, help="Repository root (default inferred from script location).")
    parser.add_argument(
        "--run-sensitive-reproduction",
        action="store_true",
        help="Required explicit guard. Reads authorized local patient-level inputs and performs the reproduction without writing outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_sensitive_reproduction:
        print("INPUTS_REQUIRED: no sensitive input was opened. Supply --run-sensitive-reproduction for the owner-authorized local reproduction.")
        return 1
    root = args.project_root.resolve()
    recorder = Recorder()
    started = time.perf_counter()
    try:
        run(root, recorder, manifest_paths(args))
    except VerificationUnavailable as exc:
        recorder.add("required component", "INPUTS_REQUIRED", str(exc))
    except VerificationFailure as exc:
        if not any(check.status == "FAIL" for check in recorder.checks):
            recorder.add("verification", "FAIL", str(exc))
    except Exception as exc:  # fail closed without exposing row-level context
        recorder.add("unexpected verifier error", "FAIL", type(exc).__name__)
    recorder.print_summary(time.perf_counter() - started)
    status = recorder.final_status()
    return 0 if status == "PASS" else 1 if status == "INPUTS_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
