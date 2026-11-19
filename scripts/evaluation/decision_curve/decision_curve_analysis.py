#!/usr/bin/env python3
"""Fail-closed, aggregate-only DCA for the fixed eICU validation cohort.

Study-specific source implementation; requires authorized inputs. Importing this module never reads patient data.
Real-data execution requires ``--execute-real-data`` and a finalized structured
policy in the governing methodology document. A command-line assertion cannot
substitute for that scientific record. Restricted internal aggregate results are
kept separate from an externally shareable status-only package. No output contains
identifiers, patient predictions/outcomes, cluster draws, or bootstrap-replicate
values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from numbers import Real
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from helpers._study_records import (
    OUTCOME_MANIFEST as OUTCOME_LOCK_PATH,
    OUTCOME_MANIFEST_SHA256,
    PERFORMANCE_MANIFEST as PERFORMANCE_LOCK_PATH,
    PERFORMANCE_MANIFEST_SHA256,
    add_manifest_arguments,
    manifest_paths,
    verify_study_file,
)


REAL_BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261011
MINIMUM_VALID_REPETITIONS = 1980
EXPECTED_N = 1897
EXPECTED_HOSPITALS = 123
EXPECTED_EVENTS = 433
IDENTITY_COLUMNS = ("unique_pid", "hstay_id", "stay_id", "hospital_id")
OUTCOME_COLUMN = "hospital_expire_flag"
CANDIDATE_PRIMARY_THRESHOLDS = tuple(
    float(Decimal("0.05") + Decimal(i) * Decimal("0.01")) for i in range(26)
)
THRESHOLD_INCREMENT = Decimal("0.01")
THRESHOLD_ULP_TOLERANCE = 8

PLAN_PATH = Path("DCA_SOFA_FEASIBILITY_AND_ANALYSIS_PLAN.md")
ANALYSIS_SOURCE_PATH = Path(__file__).resolve()
SYNTHETIC_TEST_PATH = Path("evaluation/decision_curve/EICU_DCA_SYNTHETIC_TESTS.py")
POLICY_START = "<!-- EICU_DCA_POLICY"
POLICY_END = "EICU_DCA_POLICY -->"
EVALUABLE_PATH = Path(
    "outputs/08_external_transport/eicu_outcome_attachment/"
    "eicu_external_validation_evaluable.parquet"
)
COEFFICIENT_PATH = Path(
    "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/"
    "harmonized_cr_rrt_locked_coefficient_vector.csv"
)
PERFORMANCE_PATH = Path(
    "outputs/08_external_transport/eicu_locked_external_validation/"
    "EICU_LOCKED_PRIMARY_PERFORMANCE.json"
)
EXPECTED_HASHES = {
    EVALUABLE_PATH: "3971f51cd11e83ad85b38fcd7a4cf7e3e5943aca52e668911f61783b08052e8a",
    COEFFICIENT_PATH: "3773d8ce03655e014e0e0ea578ba85888470ae248070552b089fd94c27afb513",
    OUTCOME_LOCK_PATH: OUTCOME_MANIFEST_SHA256,
    PERFORMANCE_PATH: "96a0d04c7cf2ccdf3747e94837e8206fa59928d5b90198c4b5561fdfe7d600a4",
    PERFORMANCE_LOCK_PATH: PERFORMANCE_MANIFEST_SHA256,
}

RAW_ESTIMANDS = (
    "model",
    "treat_all",
    "treat_none",
    "model_minus_treat_none",
    "model_minus_treat_all",
)
SCALES = ("raw", "standardized")
CORE_POINT_COLUMNS = {
    "threshold",
    "n",
    "events",
    "non_events",
    "prevalence",
    "model_action_positive",
    "model_true_positive",
    "model_false_positive",
    *{f"nb_{scale}_{name}" for scale in SCALES for name in RAW_ESTIMANDS},
    "standardized_status",
}
CORE_INTERVAL_COLUMNS = {
    "threshold",
    "scale",
    "estimand",
    "point_estimate",
    "ci95_low",
    "ci95_high",
    "valid_repetitions",
    "invalid_repetitions",
    "release_status",
}
POINT_COLUMNS = CORE_POINT_COLUMNS | {"threshold_set"}
INTERVAL_COLUMNS = CORE_INTERVAL_COLUMNS | {"threshold_set"}
PROHIBITED_OUTPUT_NAMES = {
    "unique_pid",
    "hstay_id",
    "stay_id",
    "hospital_id",
    "patient_id",
    "patientunitstayid",
    "prediction",
    "probability",
    "outcome",
    "prediction_vector",
    "outcome_vector",
    "bootstrap_repetition",
    "cluster_draw",
    "draw_index",
    "replicate",
}
EXTERNAL_FORBIDDEN_KEYS = {
    "point_estimates",
    "point_estimate",
    "intervals",
    "net_benefit",
    "threshold",
    "thresholds",
    "ci95_low",
    "ci95_high",
    "valid_repetitions",
    "invalid_repetitions",
    "events",
    "non_events",
    "prevalence",
    "model_action_positive",
    "model_true_positive",
    "model_false_positive",
    "expected_event_count",
    "mean_locked_predicted_risk",
    "prediction_reproduction",
}


class DCAError(RuntimeError):
    """Fail-closed DCA error."""


class DCAReleaseBlocked(DCAError):
    """Raised when prespecified bootstrap validity is insufficient."""

    def __init__(self, message: str, result: "DCAResult"):
        super().__init__(message)
        self.result = result


class DCAResult(NamedTuple):
    point_estimates: list[dict]
    intervals: list[dict]
    accounting: dict
    release_ready: bool
    samples: dict[tuple[str, str], np.ndarray] | None = None


def fail(message: str) -> None:
    raise DCAError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hashes(root: Path, manifests: dict[Path, Path] | None = None) -> dict[str, str]:
    verified: dict[str, str] = {}
    for relative, expected in EXPECTED_HASHES.items():
        try:
            verify_study_file(root / (manifests or {}).get(relative, relative), expected)
        except (FileNotFoundError, ValueError) as exc:
            raise DCAError(str(exc)) from exc
        verified[relative.as_posix()] = expected
    return verified


def execution_source_hashes(root: Path = ROOT) -> dict[str, str]:
    """Hash the exact analysis and synthetic-test sources used at execution."""
    test_path = root / SYNTHETIC_TEST_PATH
    if not test_path.is_file():
        test_path = ANALYSIS_SOURCE_PATH.with_name(SYNTHETIC_TEST_PATH.name)
    paths = {
        "analysis_source": ANALYSIS_SOURCE_PATH,
        "synthetic_test_suite": test_path,
    }
    recorded: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            fail(f"required execution source is missing: {path.name}")
        recorded[label] = sha256_file(path)
    return recorded


def parse_governing_policy_text(text: str) -> dict:
    """Parse and validate the sole structured DCA policy block."""
    if text.count(POLICY_START) != 1 or text.count(POLICY_END) != 1:
        fail("governing methodology must contain exactly one DCA policy block")
    start = text.index(POLICY_START) + len(POLICY_START)
    end = text.index(POLICY_END, start)
    try:
        policy = json.loads(text[start:end].strip())
    except json.JSONDecodeError as exc:
        raise DCAError("governing DCA policy block is not valid JSON") from exc
    required = {
        "status",
        "action",
        "primary_thresholds",
        "sensitivity_thresholds",
        "threshold_rationale",
        "true_positive_tradeoff",
        "false_positive_tradeoff",
    }
    if not isinstance(policy, dict) or set(policy) != required:
        fail("governing DCA policy block has an unexpected schema")
    return policy


def load_finalized_governing_policy(path: Path) -> tuple[dict, tuple[float, ...], str]:
    """Require a prospectively finalized action and fixed primary grid."""
    if not path.is_file():
        fail("governing methodology document is missing")
    policy = parse_governing_policy_text(path.read_text(encoding="utf-8-sig"))
    if policy["status"] != "FINALIZED_BEFORE_DCA":
        fail("governing clinical action and threshold policy remain unresolved")
    for field in (
        "action",
        "threshold_rationale",
        "true_positive_tradeoff",
        "false_positive_tradeoff",
    ):
        if not isinstance(policy[field], str) or not policy[field].strip():
            fail(f"governing DCA policy lacks a substantive {field}")
    if policy["primary_thresholds"] is None:
        fail("governing DCA policy lacks a fixed primary threshold grid")
    grid = tuple(validate_thresholds(policy["primary_thresholds"]).tolist())
    sensitivity = policy["sensitivity_thresholds"]
    if sensitivity is not None:
        validate_thresholds(sensitivity)
    return policy, grid, sha256_file(path)


def sha_set(series: pd.Series, integer: bool) -> str:
    values = [int(v) if integer else str(v).strip() for v in series if pd.notna(v)]
    payload = "\n".join(str(value) for value in sorted(set(values))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_thresholds(thresholds: Iterable[float]) -> np.ndarray:
    raw_values = list(thresholds)
    if not raw_values:
        fail("threshold grid must be nonempty")
    canonical: list[float] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, Decimal)):
            fail("threshold definitions must be numeric probabilities")
        try:
            numeric = float(value)
            decimal_value = Decimal(str(value))
            step = (decimal_value / THRESHOLD_INCREMENT).quantize(
                Decimal("1"), rounding=ROUND_HALF_EVEN
            )
            canonical_decimal = step * THRESHOLD_INCREMENT
            canonical_value = float(canonical_decimal)
        except (TypeError, ValueError, InvalidOperation, OverflowError) as exc:
            raise DCAError("threshold definitions must be finite numeric values") from exc
        if not np.isfinite(numeric):
            fail("threshold definitions must be finite numeric values")
        tolerance = THRESHOLD_ULP_TOLERANCE * abs(np.spacing(canonical_value))
        if abs(numeric - canonical_value) > tolerance:
            fail(
                "threshold is not on the prospectively specified 0.01 probability grid"
            )
        if not 0.0 < canonical_value < 1.0:
            fail("every threshold must be strictly between zero and one")
        canonical.append(canonical_value)
    values = np.asarray(canonical, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        fail("threshold grid must be a nonempty finite one-dimensional sequence")
    if np.any(np.diff(values) <= 0.0):
        fail(
            "threshold grid must be strictly increasing with no duplicates after "
            "decimal canonicalization"
        )
    return values


def parse_thresholds(text: str | None) -> tuple[float, ...]:
    if text is None:
        return CANDIDATE_PRIMARY_THRESHOLDS
    try:
        values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise DCAError("threshold list contains a nonnumeric value") from exc
    return tuple(validate_thresholds(values).tolist())


def validate_binary_outcome(outcome: np.ndarray) -> np.ndarray:
    raw = np.asarray(outcome)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number):
        fail("outcome vector must be one-dimensional and numeric")
    values = raw.astype(np.float64, copy=False)
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        fail("outcomes must be finite and exactly coded 0/1")
    return values.astype(np.int8)


def validate_probability(probability: np.ndarray) -> np.ndarray:
    raw = np.asarray(probability)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number):
        fail("probability vector must be one-dimensional and numeric")
    values = raw.astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        fail("probabilities must be complete and finite")
    if np.any((values < 0.0) | (values > 1.0)):
        fail("probabilities must remain in the closed interval [0,1]")
    return values


def fixed_reproduction_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Calculate reference performance metrics from fixed outcomes and predicted risks."""
    y = validate_binary_outcome(outcome)
    p = validate_probability(probability)
    if y.size == 0 or len(y) != len(p):
        fail("fixed reproduction vectors must be nonempty and equal in length")
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        fail("fixed AUROC reproduction requires both outcome classes")
    expected = float(p.sum())
    if expected <= 0.0:
        fail("fixed O/E reproduction requires positive expected events")
    ranks = rankdata(p, method="average")
    auroc = (
        float(ranks[y == 1].sum()) - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    return {
        "mean_locked_predicted_risk": float(p.mean()),
        "expected_event_count": expected,
        "auroc": float(auroc),
        "brier": float(np.mean((y - p) ** 2)),
        "observed_over_expected": float(positives / expected),
    }


def verify_fixed_reproduction(
    outcome: np.ndarray,
    probability: np.ndarray,
    fixed_performance: dict,
    *,
    tolerance: float = 1e-12,
) -> dict[str, float]:
    if not np.isfinite(tolerance) or tolerance < 0:
        fail("fixed reproduction tolerance must be finite and nonnegative")
    observed = fixed_reproduction_metrics(outcome, probability)
    try:
        expected = {
            "mean_locked_predicted_risk": float(
                fixed_performance["mean_locked_predicted_risk"]
            ),
            "expected_event_count": float(fixed_performance["expected_event_count"]),
            "auroc": float(fixed_performance["metrics"]["auroc"]["point"]),
            "brier": float(fixed_performance["metrics"]["brier"]["point"]),
            "observed_over_expected": float(
                fixed_performance["metrics"]["observed_over_expected"]["point"]
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise DCAError("fixed performance record lacks required reproduction metrics") from exc
    if not all(np.isfinite(value) for value in expected.values()):
        fail("fixed performance reproduction targets must be finite")
    for key in observed:
        if abs(observed[key] - expected[key]) > tolerance:
            fail(f"locked prediction/outcome reproduction differs from fixed performance: {key}")
    return observed


def validate_sites(sites: np.ndarray, expected_hospitals: int | None = None) -> np.ndarray:
    raw = np.asarray(sites)
    if raw.ndim != 1 or raw.size == 0:
        fail("hospital vector must be nonempty and one-dimensional")
    if pd.isna(raw).any():
        fail("hospital vector contains missing values")
    if expected_hospitals is not None and np.unique(raw).size != expected_hospitals:
        fail("hospital count differs from the fixed contract")
    return raw


def load_coefficients(path: Path) -> tuple[list[str], float, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or not rows or set(rows[0]) != {"term", "locked_coefficient"}:
        fail("locked coefficient format differs from the fixed contract")
    names = [row["term"] for row in rows]
    if names[0] != "const" or len(set(names)) != 20:
        fail("locked coefficient order or names are invalid")
    values = np.asarray([float(row["locked_coefficient"]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        fail("locked coefficients contain nonfinite values")
    return names[1:], float(values[0]), values[1:]


def score_locked_model(
    frame: pd.DataFrame, predictor_names: list[str], intercept: float, beta: np.ndarray
) -> np.ndarray:
    if list(frame[predictor_names].columns) != predictor_names:
        fail("predictor order differs from the locked coefficient order")
    matrix = frame[predictor_names].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        fail("fixed predictor matrix contains nonfinite values")
    eta = intercept + matrix @ np.asarray(beta, dtype=np.float64)
    probability = expit(eta)
    if not np.isfinite(eta).all() or not np.isfinite(probability).all():
        fail("locked prediction reconstruction is nonfinite")
    if np.any((probability < 0.0) | (probability > 1.0)):
        fail("locked predictions fall outside [0,1]")
    return probability


def dca_statistics(
    outcome: np.ndarray, probability: np.ndarray, thresholds: Iterable[float]
) -> list[dict]:
    y = validate_binary_outcome(outcome)
    p = validate_probability(probability)
    grid = validate_thresholds(thresholds)
    if y.size == 0 or len(y) != len(p):
        fail("outcome and probability vectors must be nonempty and equal in length")
    n = int(len(y))
    events = int(y.sum())
    non_events = n - events
    prevalence = events / n
    rows: list[dict] = []
    for threshold in grid:
        action = p >= threshold
        true_positive = int(np.count_nonzero(action & (y == 1)))
        false_positive = int(np.count_nonzero(action & (y == 0)))
        weight = float(threshold / (1.0 - threshold))
        model = true_positive / n - false_positive / n * weight
        treat_all = prevalence - (1.0 - prevalence) * weight
        treat_none = 0.0
        raw = {
            "model": float(model),
            "treat_all": float(treat_all),
            "treat_none": treat_none,
            "model_minus_treat_none": float(model - treat_none),
            "model_minus_treat_all": float(model - treat_all),
        }
        if prevalence > 0.0:
            standardized = {name: float(value / prevalence) for name, value in raw.items()}
            standardized_status = "ESTIMABLE"
        else:
            standardized = {name: None for name in RAW_ESTIMANDS}
            standardized_status = "ZERO_EVENT_PREVALENCE"
        row = {
            "threshold": float(threshold),
            "n": n,
            "events": events,
            "non_events": non_events,
            "prevalence": float(prevalence),
            "model_action_positive": int(action.sum()),
            "model_true_positive": true_positive,
            "model_false_positive": false_positive,
            "standardized_status": standardized_status,
        }
        row.update({f"nb_raw_{name}": value for name, value in raw.items()})
        row.update({f"nb_standardized_{name}": value for name, value in standardized.items()})
        if set(row) != CORE_POINT_COLUMNS:
            fail("internal point-estimate schema mismatch")
        rows.append(row)
    return rows


def cluster_index_from_draw(
    sites: np.ndarray, hospitals: np.ndarray, chosen_positions: np.ndarray
) -> np.ndarray:
    site_values = validate_sites(sites)
    observed = np.asarray(hospitals)
    chosen = np.asarray(chosen_positions)
    if observed.ndim != 1 or observed.size == 0 or np.unique(observed).size != observed.size:
        fail("hospital lookup must contain each observed hospital exactly once")
    if chosen.ndim != 1 or len(chosen) != len(observed):
        fail("cluster draw must select one position per observed hospital")
    if not np.issubdtype(chosen.dtype, np.integer):
        fail("cluster draw positions must be integers")
    if np.any((chosen < 0) | (chosen >= len(observed))):
        fail("cluster draw position lies outside the hospital lookup")
    groups = [np.flatnonzero(site_values == hospital) for hospital in observed]
    if any(group.size == 0 for group in groups):
        fail("hospital lookup contains a hospital absent from the site vector")
    return np.concatenate([groups[int(position)] for position in chosen])


def _point_map(rows: list[dict]) -> dict[tuple[str, str, float], float | None]:
    mapped: dict[tuple[str, str, float], float | None] = {}
    for row in rows:
        threshold = float(row["threshold"])
        for scale in SCALES:
            for estimand in RAW_ESTIMANDS:
                mapped[(scale, estimand, threshold)] = row[f"nb_{scale}_{estimand}"]
    return mapped


def bootstrap_dca(
    outcome: np.ndarray,
    probability: np.ndarray,
    sites: np.ndarray,
    thresholds: Iterable[float],
    *,
    n_bootstrap: int = REAL_BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
    minimum_valid: int = MINIMUM_VALID_REPETITIONS,
    expected_hospitals: int | None = None,
    enforce_minimum: bool = True,
    retain_samples: bool = False,
) -> DCAResult:
    y = validate_binary_outcome(outcome)
    p = validate_probability(probability)
    site_values = validate_sites(sites, expected_hospitals)
    grid = validate_thresholds(thresholds)
    if len(y) != len(p) or len(y) != len(site_values) or len(y) == 0:
        fail("outcome, probability, and hospital vectors must have equal nonzero length")
    if not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        fail("bootstrap repetition count must be a positive integer")
    if not isinstance(minimum_valid, int) or not 0 <= minimum_valid <= n_bootstrap:
        fail("minimum-valid rule must lie between zero and the repetition count")

    point_rows = dca_statistics(y, p, grid)
    point_values = _point_map(point_rows)
    hospitals = np.unique(site_values)
    rng = np.random.Generator(np.random.PCG64(seed))
    sample_shape = (n_bootstrap, len(grid))
    samples = {
        (scale, estimand): np.full(sample_shape, np.nan, dtype=np.float64)
        for scale in SCALES
        for estimand in RAW_ESTIMANDS
    }
    sample_sizes = np.empty(n_bootstrap, dtype=np.int64)
    event_counts = np.empty(n_bootstrap, dtype=np.int64)
    distinct_selected_hospitals = np.empty(n_bootstrap, dtype=np.int64)
    zero_event_repetitions = 0

    for repetition in range(n_bootstrap):
        chosen = rng.integers(0, len(hospitals), size=len(hospitals), dtype=np.int64)
        index = cluster_index_from_draw(site_values, hospitals, chosen)
        if index.size == 0:
            fail("hospital bootstrap produced an empty draw")
        yy = y[index]
        pp = p[index]
        draw_rows = dca_statistics(yy, pp, grid)
        sample_sizes[repetition] = len(index)
        event_counts[repetition] = int(yy.sum())
        distinct_selected_hospitals[repetition] = int(np.unique(chosen).size)
        if event_counts[repetition] == 0:
            zero_event_repetitions += 1
        for threshold_index, row in enumerate(draw_rows):
            for estimand in RAW_ESTIMANDS:
                samples[("raw", estimand)][repetition, threshold_index] = row[
                    f"nb_raw_{estimand}"
                ]
                value = row[f"nb_standardized_{estimand}"]
                if value is not None:
                    samples[("standardized", estimand)][repetition, threshold_index] = value

    interval_rows: list[dict] = []
    insufficient: list[dict] = []
    valid_count_min = n_bootstrap
    valid_count_max = 0
    for scale in SCALES:
        for estimand in RAW_ESTIMANDS:
            matrix = samples[(scale, estimand)]
            for threshold_index, threshold in enumerate(grid):
                column = matrix[:, threshold_index]
                valid = column[np.isfinite(column)]
                valid_count = int(valid.size)
                invalid_count = int(n_bootstrap - valid_count)
                valid_count_min = min(valid_count_min, valid_count)
                valid_count_max = max(valid_count_max, valid_count)
                release_status = "PASS" if valid_count >= minimum_valid else "INSUFFICIENT_VALID"
                if release_status == "PASS" and valid_count:
                    low, high = np.percentile(valid, [2.5, 97.5])
                    ci_low: float | None = float(low)
                    ci_high: float | None = float(high)
                else:
                    ci_low = None
                    ci_high = None
                    insufficient.append(
                        {
                            "scale": scale,
                            "estimand": estimand,
                            "threshold": float(threshold),
                            "valid_repetitions": valid_count,
                        }
                    )
                row = {
                    "threshold": float(threshold),
                    "scale": scale,
                    "estimand": estimand,
                    "point_estimate": point_values[(scale, estimand, float(threshold))],
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "valid_repetitions": valid_count,
                    "invalid_repetitions": invalid_count,
                    "release_status": release_status,
                }
                if set(row) != CORE_INTERVAL_COLUMNS:
                    fail("internal interval schema mismatch")
                interval_rows.append(row)

    accounting = {
        "bootstrap_repetitions": n_bootstrap,
        "bootstrap_seed": seed,
        "bootstrap_rng": "numpy.random.Generator(PCG64)",
        "bootstrap_unit": "hospital",
        "observed_hospitals": int(len(hospitals)),
        "hospital_selections_per_repetition": int(len(hospitals)),
        "same_draw_used_for_all_estimands": True,
        "minimum_valid_required": minimum_valid,
        "valid_repetitions_min_across_estimands_thresholds": valid_count_min,
        "valid_repetitions_max_across_estimands_thresholds": valid_count_max,
        "zero_event_repetitions": zero_event_repetitions,
        "zero_event_policy": (
            "raw net benefit retained; standardized quantities missing with "
            "reason ZERO_EVENT_PREVALENCE"
        ),
        "sample_size_min": int(sample_sizes.min()),
        "sample_size_max": int(sample_sizes.max()),
        "sample_size_mean": float(sample_sizes.mean()),
        "events_min": int(event_counts.min()),
        "events_max": int(event_counts.max()),
        "distinct_selected_hospitals_min": int(distinct_selected_hospitals.min()),
        "distinct_selected_hospitals_max": int(distinct_selected_hospitals.max()),
        "failure_counts": {
            "ZERO_EVENT_PREVALENCE_STANDARDIZED": int(
                zero_event_repetitions * len(grid) * len(RAW_ESTIMANDS)
            ),
            "HARD_FAILURES": 0,
        },
        "insufficient_estimand_threshold_cells": insufficient,
        "failed_draws_redrawn": 0,
    }
    release_ready = not insufficient
    result = DCAResult(
        point_estimates=point_rows,
        intervals=interval_rows,
        accounting=accounting,
        release_ready=release_ready,
        samples=samples if retain_samples else None,
    )
    if enforce_minimum and not release_ready:
        raise DCAReleaseBlocked(
            "fewer than the required valid bootstrap repetitions for at least one estimand/threshold",
            result,
        )
    return result


def bootstrap_policy_thresholds(
    outcome: np.ndarray,
    probability: np.ndarray,
    sites: np.ndarray,
    primary_thresholds: Iterable[float],
    sensitivity_thresholds: Iterable[float] | None,
    **bootstrap_kwargs,
) -> DCAResult:
    """Analyze both policy grids once over their deduplicated union."""
    primary = tuple(validate_thresholds(primary_thresholds).tolist())
    sensitivity = (
        None
        if sensitivity_thresholds is None
        else tuple(validate_thresholds(sensitivity_thresholds).tolist())
    )
    primary_set = set(primary)
    sensitivity_set = set() if sensitivity is None else set(sensitivity)
    union = tuple(sorted(primary_set | sensitivity_set))
    result = bootstrap_dca(
        outcome,
        probability,
        sites,
        union,
        **bootstrap_kwargs,
    )

    def threshold_set_label(threshold: float) -> str:
        in_primary = threshold in primary_set
        in_sensitivity = threshold in sensitivity_set
        if in_primary and in_sensitivity:
            return "PRIMARY_AND_SENSITIVITY"
        if in_primary:
            return "PRIMARY"
        if in_sensitivity:
            return "SENSITIVITY_ONLY"
        fail("internal threshold-set membership mismatch")

    point_rows = [
        {**row, "threshold_set": threshold_set_label(float(row["threshold"]))}
        for row in result.point_estimates
    ]
    interval_rows = [
        {**row, "threshold_set": threshold_set_label(float(row["threshold"]))}
        for row in result.intervals
    ]
    accounting = dict(result.accounting)
    accounting.update(
        {
            "primary_thresholds": list(primary),
            "sensitivity_thresholds": (
                None if sensitivity is None else list(sensitivity)
            ),
            "unique_thresholds_analyzed": list(union),
            "overlapping_thresholds_calculated_once": True,
            "single_shared_bootstrap_across_threshold_sets": True,
        }
    )
    return DCAResult(
        point_estimates=point_rows,
        intervals=interval_rows,
        accounting=accounting,
        release_ready=result.release_ready,
        samples=result.samples,
    )


def validate_aggregate_frame(frame: pd.DataFrame, allowed_columns: set[str], label: str) -> None:
    if set(frame.columns) != allowed_columns:
        fail(f"{label}: aggregate output schema is not allowlisted")
    lowered = [str(column).lower() for column in frame.columns]
    if any(column in PROHIBITED_OUTPUT_NAMES for column in lowered):
        fail(f"{label}: prohibited patient- or replicate-level output column")


def validate_aggregate_json(value, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in PROHIBITED_OUTPUT_NAMES:
                fail(f"aggregate JSON contains prohibited key at {path}")
            validate_aggregate_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_aggregate_json(child, f"{path}[{index}]")
    elif isinstance(value, np.ndarray):
        fail(f"aggregate JSON contains an array at {path}")


def stage_restricted_internal_outputs(
    staging: Path,
    result: DCAResult,
    manifest: dict,
    publication_pair_sha256: str,
) -> None:
    if not result.release_ready:
        fail("aggregate release blocked by the minimum-valid rule")
    if result.samples is not None:
        fail("replicate samples must not be retained by the real-data output path")
    point = pd.DataFrame(result.point_estimates)
    intervals = pd.DataFrame(result.intervals)
    validate_aggregate_frame(point, POINT_COLUMNS, "point estimates")
    validate_aggregate_frame(intervals, INTERVAL_COLUMNS, "interval estimates")
    validate_aggregate_json(result.accounting)
    validate_aggregate_json(manifest)

    point.to_csv(staging / "DCA_POINT_ESTIMATES.csv", index=False)
    intervals.to_csv(staging / "DCA_POINTWISE_INTERVALS.csv", index=False)
    (staging / "DCA_BOOTSTRAP_ACCOUNTING.json").write_text(
        json.dumps(result.accounting, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_hashes = {
        path.name: sha256_file(path) for path in sorted(staging.iterdir()) if path.is_file()
    }
    final_manifest = dict(manifest)
    final_manifest["aggregate_output_sha256"] = output_hashes
    final_manifest["status"] = "STAGED_PAIRED_PUBLICATION"
    final_manifest["disclosure_class"] = "RESTRICTED_INTERNAL"
    final_manifest["publication_pair_sha256"] = publication_pair_sha256
    (staging / "DCA_ANALYSIS_MANIFEST.json").write_text(
        json.dumps(final_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize_restricted_publication(
    restricted_destination: Path, publication_pair_sha256: str
) -> None:
    manifest_path = restricted_destination / "DCA_ANALYSIS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "STAGED_PAIRED_PUBLICATION"
        or manifest.get("publication_pair_sha256") != publication_pair_sha256
    ):
        fail("restricted staged manifest cannot be finalized")
    manifest["status"] = "RESTRICTED_INTERNAL_AGGREGATES_PUBLISHED"
    manifest["paired_external_package_published"] = True
    temporary_manifest = restricted_destination / "DCA_ANALYSIS_MANIFEST.atomic.json"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)


def validate_external_shareable_json(value, path: str = "root") -> None:
    """Reject result-bearing fields from the automatic external package."""
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            result_like = lowered.startswith("nb_") or "net_benefit" in lowered
            if (
                lowered in PROHIBITED_OUTPUT_NAMES
                or lowered in EXTERNAL_FORBIDDEN_KEYS
                or result_like
            ):
                fail(f"external-shareable JSON contains prohibited key at {path}.{key}")
            validate_external_shareable_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_external_shareable_json(child, f"{path}[{index}]")
    elif isinstance(value, np.ndarray):
        fail(f"external-shareable JSON contains an array at {path}")


def build_external_release_status(
    manifest: dict, threshold_count: int, publication_pair_sha256: str
) -> dict:
    """Build a status-only package; threshold-level results remain restricted."""
    payload = {
        "status": "RESULTS_WITHHELD_DISCLOSURE_ASSESSMENT_REQUIRED",
        "disclosure_class": "EXTERNALLY_SHAREABLE_STATUS_ONLY",
        "purpose": manifest["purpose"],
        "endpoint": manifest["endpoint"],
        "model": manifest["model"],
        "bootstrap_repetitions": manifest["bootstrap_repetitions"],
        "minimum_valid_repetitions": manifest["minimum_valid_repetitions"],
        "threshold_count": int(threshold_count),
        "threshold_level_results_included": False,
        "patient_or_replicate_level_material_included": False,
        "reason": (
            "Exact threshold-specific net benefit, confidence limits, and action/event "
            "counts are restricted because small cells and differences across adjacent "
            "thresholds can permit complementary reconstruction. Scientific calculations "
            "are unchanged in the restricted internal package."
        ),
        "future_release_requirement": (
            "A separate disclosure review must jointly assess small-cell suppression, "
            "complementary suppression, threshold thinning/coarsening, rounding, and "
            "whether plotted coordinates permit adjacent-threshold reconstruction."
        ),
        "analysis_source_sha256": manifest["execution_source_sha256"]["analysis_source"],
        "synthetic_test_suite_sha256": manifest["execution_source_sha256"][
            "synthetic_test_suite"
        ],
        "governing_methodology_sha256": manifest["governing_methodology_sha256"],
        "publication_pair_sha256": publication_pair_sha256,
    }
    validate_external_shareable_json(payload)
    return payload


def stage_external_shareable_outputs(
    staging: Path,
    manifest: dict,
    threshold_count: int,
    publication_pair_sha256: str,
) -> None:
    payload = build_external_release_status(
        manifest, threshold_count, publication_pair_sha256
    )
    (staging / "DCA_EXTERNAL_RELEASE_STATUS.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_output_destinations(
    restricted_destination: Path, external_destination: Path
) -> None:
    restricted_resolved = restricted_destination.resolve()
    external_resolved = external_destination.resolve()
    if restricted_resolved == external_resolved:
        fail("restricted and external output directories must be distinct")
    if (
        restricted_resolved in external_resolved.parents
        or external_resolved in restricted_resolved.parents
    ):
        fail("restricted and external output directories must not be nested")
    if restricted_destination.exists():
        fail("restricted output destination already exists; publication is blocked")
    if external_destination.exists():
        fail("external output destination already exists; publication is blocked")


def write_separated_outputs(
    restricted_destination: Path,
    external_destination: Path,
    result: DCAResult,
    manifest: dict,
) -> tuple[Path, Path]:
    validate_output_destinations(restricted_destination, external_destination)
    restricted_destination.parent.mkdir(parents=True, exist_ok=True)
    external_destination.parent.mkdir(parents=True, exist_ok=True)
    restricted_staging = Path(
        tempfile.mkdtemp(
            prefix=f"{restricted_destination.name}.staging_",
            dir=restricted_destination.parent,
        )
    )
    external_staging = Path(
        tempfile.mkdtemp(
            prefix=f"{external_destination.name}.staging_",
            dir=external_destination.parent,
        )
    )
    pair_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    publication_pair_sha256 = hashlib.sha256(pair_payload.encode("utf-8")).hexdigest()
    external_published = False
    restricted_published = False
    try:
        stage_restricted_internal_outputs(
            restricted_staging, result, manifest, publication_pair_sha256
        )
        stage_external_shareable_outputs(
            external_staging,
            manifest,
            len(result.point_estimates),
            publication_pair_sha256,
        )
        # Publish status-only external first. Restricted results appear only after
        # its paired external package is already in place.
        os.replace(external_staging, external_destination)
        external_published = True
        os.replace(restricted_staging, restricted_destination)
        restricted_published = True
        finalize_restricted_publication(
            restricted_destination, publication_pair_sha256
        )
    except Exception:
        if restricted_published and restricted_destination.exists():
            try:
                os.replace(restricted_destination, restricted_staging)
            except OSError as rollback_error:
                raise DCAError(
                    "paired publication failed and restricted rollback also failed; "
                    "the restricted manifest must not be released unless its "
                    "status is RESTRICTED_INTERNAL_AGGREGATES_PUBLISHED"
                ) from rollback_error
        if external_published and external_destination.exists():
            try:
                os.replace(external_destination, external_staging)
            except OSError as rollback_error:
                raise DCAError(
                    "paired publication failed and external rollback also failed; "
                    "do not release either package"
                ) from rollback_error
        raise
    return restricted_destination, external_destination


def _validate_fixed_frame(
    frame: pd.DataFrame, predictor_names: list[str], lock: dict
) -> tuple[np.ndarray, np.ndarray]:
    expected_columns = list(IDENTITY_COLUMNS) + predictor_names + [
        "outcome_evaluable",
        OUTCOME_COLUMN,
    ]
    if list(frame.columns) != expected_columns:
        fail("fixed evaluable schema or predictor order differs")
    if len(frame) != EXPECTED_N or frame["hospital_id"].nunique() != EXPECTED_HOSPITALS:
        fail("fixed evaluable cohort count or hospital count differs")
    for column in IDENTITY_COLUMNS:
        if frame[column].isna().any():
            fail("fixed identity contains missing values")
    for column in ("unique_pid", "hstay_id", "stay_id"):
        if frame[column].duplicated().any():
            fail("fixed identity is not unique")
    if not frame["outcome_evaluable"].eq(1).all():
        fail("non-evaluable row entered the fixed evaluable matrix")
    y = validate_binary_outcome(frame[OUTCOME_COLUMN].to_numpy())
    if int(y.sum()) != EXPECTED_EVENTS:
        fail("fixed event count differs")
    matrix = frame[predictor_names].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        fail("fixed predictor matrix contains nonfinite values")
    identity = lock.get("identity_lock", {})
    identity_checks = (
        ("evaluable_uniquePid_sha256", frame["unique_pid"], False),
        ("evaluable_hstay_id_sha256", frame["hstay_id"], True),
        ("evaluable_stay_id_sha256", frame["stay_id"], True),
    )
    for key, series, integer in identity_checks:
        if sha_set(series, integer) != identity.get(key):
            fail(f"fixed identity hash mismatch: {key}")
    return y, frame["hospital_id"].to_numpy()


def execute_real_data(
    root: Path,
    restricted_destination: Path,
    external_destination: Path,
    manifests: dict[Path, Path] | None = None,
) -> tuple[Path, Path]:
    # Destination safety is checked before any fixed patient-level artifact is read.
    validate_output_destinations(restricted_destination, external_destination)
    policy, grid, methodology_hash = load_finalized_governing_policy(root / PLAN_PATH)
    sensitivity_grid = policy["sensitivity_thresholds"]
    source_hashes = execution_source_hashes(root)
    verified = verify_hashes(root, manifests)
    predictors, intercept, beta = load_coefficients(root / COEFFICIENT_PATH)
    outcome_manifest = (manifests or {}).get(OUTCOME_LOCK_PATH, OUTCOME_LOCK_PATH)
    manifest_path = verify_study_file(root / outcome_manifest, EXPECTED_HASHES[OUTCOME_LOCK_PATH])
    lock = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    frame = pd.read_parquet(root / EVALUABLE_PATH)
    y, sites = _validate_fixed_frame(frame, predictors, lock)
    probability = score_locked_model(frame, predictors, intercept, beta)

    performance = json.loads((root / PERFORMANCE_PATH).read_text(encoding="utf-8"))
    tolerance = 1e-12
    reproduction = verify_fixed_reproduction(y, probability, performance, tolerance=tolerance)

    result = bootstrap_policy_thresholds(
        y,
        probability,
        sites,
        grid,
        sensitivity_grid,
        n_bootstrap=REAL_BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
        minimum_valid=MINIMUM_VALID_REPETITIONS,
        expected_hospitals=EXPECTED_HOSPITALS,
        enforce_minimum=True,
        retain_samples=False,
    )
    manifest = {
        "created_utc": utc_now(),
        "purpose": "exploratory decision-analytic assessment; not intervention effectiveness",
        "endpoint": "subsequent in-hospital mortality after +24-hour landmark until discharge",
        "fixed_horizon_days": None,
        "clinical_policy_status": policy["status"],
        "clinical_action": policy["action"],
        "threshold_rationale": policy["threshold_rationale"],
        "true_positive_tradeoff": policy["true_positive_tradeoff"],
        "false_positive_tradeoff": policy["false_positive_tradeoff"],
        "thresholds": list(grid),
        "primary_thresholds": list(grid),
        "sensitivity_thresholds": sensitivity_grid,
        "threshold_sets_share_one_deduplicated_bootstrap": True,
        "thresholds_selected_from_observed_dca": False,
        "matches_original_provisional_candidate_grid": (
            grid == CANDIDATE_PRIMARY_THRESHOLDS
        ),
        "model": "unchanged locked harmonized MIMIC Cr/RRT model",
        "no_refit_recalibration_clipping_or_prediction_modification": True,
        "cohort_n": EXPECTED_N,
        "events": EXPECTED_EVENTS,
        "hospitals": EXPECTED_HOSPITALS,
        "bootstrap_repetitions": REAL_BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "minimum_valid_repetitions": MINIMUM_VALID_REPETITIONS,
        "input_sha256": verified,
        "governing_methodology_sha256": methodology_hash,
        "execution_source_sha256": source_hashes,
        "prediction_reproduction_tolerance": tolerance,
        "prediction_reproduction_definitions": {
            "auroc": "Mann-Whitney rank statistic with average ranks for ties",
            "brier": "mean squared error of original locked probability",
            "observed_over_expected": "sum(outcome) / sum(original locked probability)",
        },
        "prediction_reproduction": reproduction,
        "patient_level_output_written": False,
        "restricted_internal_results_only": True,
        "automatic_external_release": "STATUS_ONLY_NO_THRESHOLD_RESULTS",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    return write_separated_outputs(
        restricted_destination, external_destination, result, manifest
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_manifest_arguments(parser)
    parser.add_argument(
        "--execute-real-data",
        action="store_true",
        help="explicitly enter the guarded fixed-cohort analysis path",
    )
    parser.add_argument(
        "--restricted-output-dir",
        type=Path,
        default=Path("work/authorized_results/eicu_dca/restricted_internal"),
        help="new, nonexistent restricted-internal aggregate-output directory",
    )
    parser.add_argument(
        "--external-output-dir",
        type=Path,
        default=Path("work/authorized_results/eicu_dca_external_status"),
        help="new, nonexistent externally shareable status-only directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute_real_data:
        fail(
            "no real-data action requested; run the separate synthetic test suite during preparation"
        )
    restricted = args.restricted_output_dir
    external = args.external_output_dir
    if not restricted.is_absolute():
        restricted = ROOT / restricted
    if not external.is_absolute():
        external = ROOT / external
    published_restricted, published_external = execute_real_data(
        ROOT, restricted, external, manifest_paths(args)
    )
    print(f"RESTRICTED-INTERNAL DCA COMPLETE: {published_restricted}")
    print(f"EXTERNAL STATUS-ONLY PACKAGE COMPLETE: {published_external}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DCAError as exc:
        print(f"DCA HARD STOP: {exc}", file=os.sys.stderr)
        raise SystemExit(2)
