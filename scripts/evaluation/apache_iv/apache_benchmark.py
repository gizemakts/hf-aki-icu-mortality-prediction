#!/usr/bin/env python3
"""Post-hoc eICU APACHE IV/IVa benchmark; fail-closed and aggregate-only.

This module provides an APACHE IV/IVa benchmark for authorized study inputs. It has two modes:

  --preflight  Verify fixed inputs and the approved common-cohort structure.
               It does not score a model or calculate performance metrics.
  --run        Authorized scoring, performance evaluation, paired
               hospital-cluster bootstrap, and aggregate-only reporting.

No mode writes patient-level identifiers, predictions, outcomes, or bootstrap
replicate vectors. --run requires an explicit authorization token and a new,
nonexistent output directory.
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
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from helpers._study_records import (
    OUTCOME_MANIFEST as OUTCOME_LOCK_PATH,
    OUTCOME_MANIFEST_SHA256,
    PERFORMANCE_MANIFEST as PERFORMANCE_LOCK_PATH,
    PERFORMANCE_MANIFEST_SHA256,
    add_manifest_arguments,
    manifest_paths,
    verify_study_file,
)


N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261010
MIN_VALID_BOOTSTRAPS = 1980
CURVE_GRID_N = 120
DESIGN_RANK_RELATIVE_TOLERANCE = 1e-12
MAX_DESIGN_CONDITION_NUMBER = 1e12
MAX_ABSOLUTE_FITTED_LOGIT = 30.0
MAX_ABSOLUTE_COEFFICIENT = 1e4
SMALL_CELL_SUPPRESSION_THRESHOLD = 10
EXPECTED_EVALUABLE_N = 1897
EXPECTED_EVALUABLE_HOSPITALS = 123
EXPECTED_EVALUABLE_EVENTS = 433
EXPECTED_RECORD_AVAILABLE_N = 1699
EXPECTED_COMMON_N = 1582
EXPECTED_COMMON_HOSPITALS = 115
EXPECTED_COMMON_EVENTS = 368
EXPECTED_APACHE_ROWS_PER_VERSION = 148532
AUTHORIZATION_TOKEN = "EXECUTE_POSTHOC_APACHE_BENCHMARK"
VERSIONS = ("IV", "IVa")
MODEL_ORDER = ("Locked harmonized MIMIC Cr/RRT", "APACHE IV", "APACHE IVa")
IDENTITY = ("unique_pid", "hstay_id", "stay_id", "hospital_id")
OUTCOME = "hospital_expire_flag"
MEMBERSHIP_ENCODING = (
    "positive base-10 int64 stay_id; numerically ascending; ASCII digits; "
    "one identifier per line; LF separator; terminal LF; SHA-256 over exact bytes"
)
EXPECTED_COMMON_MEMBERSHIP_SHA256 = (
    "a7cc19756c7831c7b1b3ceeaa35c78575562607928cc2a94df22064484ba505a"
)

EVALUABLE_PATH = Path(
    "outputs/08_external_transport/eicu_outcome_attachment/"
    "eicu_external_validation_evaluable.parquet"
)
COEFFICIENT_PATH = Path(
    "outputs/08_external_transport/mimic_cr_rrt_harmonized_refit/"
    "harmonized_cr_rrt_locked_coefficient_vector.csv"
)
APACHE_PATH = Path(
    "data/external/eicu-collaborative-research-database-2.0/"
    "apachePatientResult.csv.gz"
)
SCORING_REFERENCE_PATH = Path(
    "evaluation/external_validation/"
    "external_performance.py"
)
ORIGINAL_PERFORMANCE_PATH = Path(
    "outputs/08_external_transport/eicu_locked_external_validation/"
    "EICU_LOCKED_PRIMARY_PERFORMANCE.json"
)
MEMBERSHIP_LOCK_PATH = Path(
    "evaluation/apache_iv/"
    "EICU_APACHE_COMMON_COHORT_MEMBERSHIP_LOCK.json"
)

EXPECTED_HASHES = {
    EVALUABLE_PATH: "3971f51cd11e83ad85b38fcd7a4cf7e3e5943aca52e668911f61783b08052e8a",
    COEFFICIENT_PATH: "3773d8ce03655e014e0e0ea578ba85888470ae248070552b089fd94c27afb513",
    APACHE_PATH: "f9870a5ad58b1519cf07f51cbe2717a8bf390f74159e2310e2bac3b6bb7a070a",
    SCORING_REFERENCE_PATH: "4ecb75ab2ce6e4dd88cba761451f627e1fb45216468fad4ae3313cb7cc3f7307",
    ORIGINAL_PERFORMANCE_PATH: "96a0d04c7cf2ccdf3747e94837e8206fa59928d5b90198c4b5561fdfe7d600a4",
    OUTCOME_LOCK_PATH: OUTCOME_MANIFEST_SHA256,
    PERFORMANCE_LOCK_PATH: PERFORMANCE_MANIFEST_SHA256,
}


class BenchmarkError(RuntimeError):
    """Fail-closed benchmark exception."""


class CalibrationError(BenchmarkError):
    """Calibration failure with a stable, aggregate-only taxonomy."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def fail(message: str) -> None:
    raise BenchmarkError(message)


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
            raise BenchmarkError(str(exc)) from exc
        verified[relative.as_posix()] = expected
    return verified


def sha_set(series: pd.Series, integer: bool) -> str:
    values = [int(v) if integer else str(v).strip() for v in series if pd.notna(v)]
    payload = "\n".join(str(v) for v in sorted(set(values))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_membership_bytes(stay_ids) -> bytes:
    values = []
    for value in stay_ids:
        if pd.isna(value):
            fail("common-cohort membership contains a missing stay identifier")
        numeric = int(value)
        if numeric <= 0 or float(value) != numeric:
            fail("common-cohort membership contains a non-positive or noninteger stay identifier")
        values.append(numeric)
    if len(values) != len(set(values)):
        fail("common-cohort membership contains duplicate stay identifiers")
    return "".join(f"{value}\n" for value in sorted(values)).encode("ascii")


def membership_sha256(stay_ids) -> str:
    return hashlib.sha256(canonical_membership_bytes(stay_ids)).hexdigest()


def verify_membership_lock(
    lock: dict, stay_ids, verified_hashes: dict[str, str]
) -> dict[str, str | int]:
    if lock.get("canonical_identity_encoding") != MEMBERSHIP_ENCODING:
        fail("common-cohort membership encoding differs from the pinned contract")
    expected_sources = lock.get("source_sha256", {})
    for relative in (EVALUABLE_PATH, APACHE_PATH):
        key = str(relative).replace("\\", "/")
        if expected_sources.get(key) != verified_hashes.get(key):
            fail("common-cohort membership lock source hash mismatch")
    values = list(stay_ids)
    if lock.get("expected_member_count") != EXPECTED_COMMON_N:
        fail("common-cohort membership lock expected count differs from the pinned contract")
    if len(values) != EXPECTED_COMMON_N:
        fail("common-cohort membership lock count mismatch")
    if lock.get("expected_hospital_count") != EXPECTED_COMMON_HOSPITALS:
        fail("common-cohort membership lock hospital count differs from the pinned contract")
    locked_digest = lock.get("common_cohort_membership_sha256")
    if locked_digest != EXPECTED_COMMON_MEMBERSHIP_SHA256:
        fail("membership-lock JSON digest differs from the pinned expected SHA-256")
    actual = membership_sha256(values)
    if actual != EXPECTED_COMMON_MEMBERSHIP_SHA256:
        fail("freshly derived common-cohort membership SHA-256 mismatch")
    if lock.get("independent_cross_check_status") != "PASS":
        fail("common-cohort independent membership cross-check is not PASS")
    details = lock.get("cross_check_details", {})
    required_details = {
        "primary_member_count": EXPECTED_COMMON_N,
        "independent_member_count": EXPECTED_COMMON_N,
        "member_sets_identical": True,
        "version_valid_sets_identical_in_both_derivations": True,
        "digests_identical": True,
        "record_available_count_in_both_derivations": EXPECTED_RECORD_AVAILABLE_N,
    }
    if any(details.get(key) != value for key, value in required_details.items()):
        fail("common-cohort independent cross-check details differ from the pinned contract")
    return {
        "status": "PASS_AUTOMATIC_FAIL_CLOSED",
        "membership_sha256": actual,
        "member_count": len(values),
        "expected_hospital_count": EXPECTED_COMMON_HOSPITALS,
    }


def validate_binary_outcome(y: np.ndarray, label: str = "outcome") -> np.ndarray:
    raw = np.asarray(y)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number):
        fail(f"{label}: outcome vector is malformed or nonnumeric")
    numeric = raw.astype(np.float64, copy=False)
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        fail(f"{label}: outcomes must be finite and exactly coded 0/1")
    return numeric.astype(np.int8)


def assert_identity_unique(frame: pd.DataFrame, label: str) -> None:
    for column in IDENTITY:
        if column not in frame.columns or frame[column].isna().any():
            fail(f"{label}: missing identity values in {column}")
    for column in ("unique_pid", "hstay_id", "stay_id"):
        if frame[column].duplicated().any():
            fail(f"{label}: duplicate identity in {column}")


def assert_same_identity(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    """Compare identity without emitting any identifier value on failure."""
    if len(left) != len(right):
        fail(f"{label}: identity row-count mismatch")
    a = left.sort_values("stay_id", kind="stable").reset_index(drop=True)
    b = right.sort_values("stay_id", kind="stable").reset_index(drop=True)
    for column in IDENTITY:
        if column not in a or column not in b:
            fail(f"{label}: identity column missing")
        if column == "unique_pid":
            av = a[column].astype("string").str.strip().to_numpy(dtype=str)
            bv = b[column].astype("string").str.strip().to_numpy(dtype=str)
        else:
            av = pd.to_numeric(a[column], errors="raise").to_numpy(dtype=np.int64)
            bv = pd.to_numeric(b[column], errors="raise").to_numpy(dtype=np.int64)
        if np.count_nonzero(av != bv):
            fail(f"{label}: identity mismatch in {column}")


def load_coefficients(path: Path) -> tuple[list[str], float, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 20 or set(rows[0]) != {"term", "locked_coefficient"}:
        fail("locked coefficient format differs from the fixed contract")
    names = [row["term"] for row in rows]
    if names[0] != "const" or len(set(names)) != 20:
        fail("locked coefficient order/names are invalid")
    values = np.asarray([float(row["locked_coefficient"]) for row in rows], dtype=np.float64)
    if not np.isfinite(values).all():
        fail("locked coefficients contain nonfinite values")
    return names[1:], float(values[0]), values[1:]


def validate_evaluable_schema(frame: pd.DataFrame, predictor_names: list[str]) -> None:
    expected_columns = list(IDENTITY) + predictor_names + ["outcome_evaluable", OUTCOME]
    if list(frame.columns) != expected_columns:
        fail("fixed evaluable schema or predictor order differs")


def load_fixed_evaluable(
    root: Path, predictor_names: list[str], manifest: Path = OUTCOME_LOCK_PATH,
) -> pd.DataFrame:
    frame = pd.read_parquet(root / EVALUABLE_PATH)
    validate_evaluable_schema(frame, predictor_names)
    assert_identity_unique(frame, "fixed evaluable matrix")
    if len(frame) != EXPECTED_EVALUABLE_N:
        fail("fixed evaluable row count differs")
    if frame["hospital_id"].nunique() != EXPECTED_EVALUABLE_HOSPITALS:
        fail("fixed evaluable hospital count differs")
    if not frame["outcome_evaluable"].eq(1).all():
        fail("non-evaluable row entered fixed evaluable matrix")
    if not frame[OUTCOME].isin([0, 1]).all():
        fail("fixed outcome is not binary")
    if int(frame[OUTCOME].sum()) != EXPECTED_EVALUABLE_EVENTS:
        fail("fixed outcome event count differs")
    matrix = frame[predictor_names].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        fail("fixed model matrix contains nonfinite values")

    manifest_path = verify_study_file(root / manifest, EXPECTED_HASHES[OUTCOME_LOCK_PATH])
    lock = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    identity = lock["identity_lock"]
    checks = (
        ("evaluable_uniquePid_sha256", frame["unique_pid"], False),
        ("evaluable_hstay_id_sha256", frame["hstay_id"], True),
        ("evaluable_stay_id_sha256", frame["stay_id"], True),
    )
    for key, series, integer in checks:
        if sha_set(series, integer) != identity[key]:
            fail(f"fixed identity hash mismatch: {key}")
    return frame


def classify_probability_text(series: pd.Series) -> pd.DataFrame:
    text = series.astype("string").str.strip()
    blank = text.isna() | text.eq("")
    numeric = pd.to_numeric(text, errors="coerce")
    nonnumeric = (~blank) & numeric.isna()
    sentinel = numeric.eq(-1)
    outside = numeric.notna() & (~numeric.between(0, 1, inclusive="both")) & (~sentinel)
    boundary = numeric.isin([0.0, 1.0])
    valid = numeric.between(0, 1, inclusive="both")
    return pd.DataFrame(
        {
            "prediction_text": text,
            "prediction": numeric,
            "blank": blank,
            "nonnumeric": nonnumeric,
            "sentinel": sentinel,
            "outside": outside,
            "boundary": boundary,
            "valid": valid,
        },
        index=series.index,
    )


def validate_apache_structure(
    apache: pd.DataFrame, fixed_stays: set[int]
) -> tuple[pd.DataFrame, dict, dict[str, set[int]]]:
    required = {
        "apachepatientresultsid",
        "patientunitstayid",
        "apacheversion",
        "predictedhospitalmortality",
    }
    if set(apache.columns) != required:
        fail("APACHE source columns differ from the required minimal schema")

    frame = apache.copy()
    for column in required:
        frame[column] = frame[column].astype("string").str.strip()
    stay = pd.to_numeric(frame["patientunitstayid"], errors="coerce")
    result_id = pd.to_numeric(frame["apachepatientresultsid"], errors="coerce")
    if (
        stay.isna().any()
        or result_id.isna().any()
        or (~np.isfinite(stay)).any()
        or (~np.isfinite(result_id)).any()
        or (stay <= 0).any()
        or (result_id <= 0).any()
        or (stay.mod(1) != 0).any()
        or (result_id.mod(1) != 0).any()
    ):
        fail("APACHE contains invalid linkage or result identifiers")
    frame["stay_id"] = stay.astype("int64")
    frame["apache_result_id"] = result_id.astype("int64")
    labels = set(frame["apacheversion"].dropna().unique())
    if labels != set(VERSIONS):
        fail("unexpected or missing APACHE version label")
    if frame["apacheversion"].isna().any():
        fail("missing APACHE version label")

    group_sizes = frame.groupby(["stay_id", "apacheversion"], dropna=False).size()
    if (group_sizes > 1).any():
        fail("within-version duplicate APACHE record detected; adjudication required")
    if frame["apache_result_id"].duplicated().any():
        fail("duplicate APACHE result identifier detected")
    version_counts = frame.groupby("apacheversion").size().to_dict()
    if any(version_counts.get(v) != EXPECTED_APACHE_ROWS_PER_VERSION for v in VERSIONS):
        fail("local APACHE version record count differs from the feasibility lock")

    classified = classify_probability_text(frame["predictedhospitalmortality"])
    frame = pd.concat([frame, classified], axis=1)
    matched = frame.loc[frame["stay_id"].isin(fixed_stays)].copy()
    if matched["blank"].any() or matched["nonnumeric"].any() or matched["outside"].any():
        fail("study-cohort APACHE prediction is blank, nonnumeric, or outside [0,1]")
    invalid = matched.loc[~matched["valid"], "prediction"]
    if len(invalid) and not invalid.eq(-1).all():
        fail("unexpected study-cohort APACHE invalid-prediction value")
    record_sets = {
        version: set(matched.loc[matched["apacheversion"].eq(version), "stay_id"])
        for version in VERSIONS
    }
    if record_sets["IV"] != record_sets["IVa"]:
        fail("APACHE version record-availability populations differ")
    if len(record_sets["IV"]) != EXPECTED_RECORD_AVAILABLE_N:
        fail("APACHE record-available population differs")
    valid_sets = {
        version: set(
            matched.loc[
                matched["apacheversion"].eq(version) & matched["valid"], "stay_id"
            ]
        )
        for version in VERSIONS
    }
    if valid_sets["IV"] != valid_sets["IVa"]:
        fail("APACHE version valid populations differ")
    if len(valid_sets["IV"]) != EXPECTED_COMMON_N:
        fail("APACHE common valid population differs")

    wide = matched.pivot(index="stay_id", columns="apacheversion", values="prediction")
    common_ids = sorted(valid_sets["IV"])
    wide = wide.loc[common_ids, list(VERSIONS)]
    if wide.isna().any().any() or not np.isfinite(wide.to_numpy(dtype=float)).all():
        fail("common APACHE prediction matrix is incomplete or nonfinite")
    if not ((wide >= 0) & (wide <= 1)).all().all():
        fail("common APACHE probability outside the closed unit interval")

    represented = record_sets["IV"]
    sentinel_sets = {
        version: set(
            matched.loc[
                matched["apacheversion"].eq(version) & matched["sentinel"], "stay_id"
            ]
        )
        for version in VERSIONS
    }
    if sentinel_sets["IV"] != sentinel_sets["IVa"]:
        fail("APACHE version sentinel populations differ")
    membership = {
        "full_fixed_evaluable": set(fixed_stays),
        "record_available": set(represented),
        "no_record": set(fixed_stays) - set(represented),
        "nonpredictive_sentinel": set(sentinel_sets["IV"]),
        "valid_common": set(valid_sets["IV"]),
        "without_valid_prediction": set(fixed_stays) - set(valid_sets["IV"]),
    }
    if membership["valid_common"] | membership["nonpredictive_sentinel"] != membership["record_available"]:
        fail("APACHE record-availability categories do not reconcile")
    if membership["record_available"] | membership["no_record"] != membership["full_fixed_evaluable"]:
        fail("APACHE full-population availability categories do not reconcile")

    audit = {
        "local_rows": int(len(frame)),
        "version_rows": {v: int(version_counts[v]) for v in VERSIONS},
        "record_available_n": int(len(record_sets["IV"])),
        "common_valid_n": int(len(common_ids)),
        "invalid_sentinel_n_each_version": {
            v: int(
                matched.loc[
                    matched["apacheversion"].eq(v) & matched["sentinel"], "stay_id"
                ].nunique()
            )
            for v in VERSIONS
        },
        "boundary_n_each_version": {
            v: int(
                matched.loc[
                    matched["apacheversion"].eq(v) & matched["boundary"], "stay_id"
                ].nunique()
            )
            for v in VERSIONS
        },
        "source_outside_or_unparseable_rows": int(
            (frame["blank"] | frame["nonnumeric"] | frame["outside"]).sum()
        ),
        "source_probability_classification_rows": {
            "blank": int(frame["blank"].sum()),
            "nonnumeric": int(frame["nonnumeric"].sum()),
            "sentinel_minus_one": int(frame["sentinel"].sum()),
            "boundary_zero_or_one": int(frame["boundary"].sum()),
            "outside_closed_interval_excluding_sentinel": int(frame["outside"].sum()),
            "valid_closed_interval": int(frame["valid"].sum()),
        },
        "study_probability_classification_rows": {
            "blank": int(matched["blank"].sum()),
            "nonnumeric": int(matched["nonnumeric"].sum()),
            "sentinel_minus_one": int(matched["sentinel"].sum()),
            "boundary_zero_or_one": int(matched["boundary"].sum()),
            "outside_closed_interval_excluding_sentinel": int(matched["outside"].sum()),
            "valid_closed_interval": int(matched["valid"].sum()),
        },
        "within_version_duplicate_groups": 0,
    }
    return wide, audit, membership


def load_apache_minimal(
    root: Path, fixed_stays: set[int]
) -> tuple[pd.DataFrame, dict, dict[str, set[int]]]:
    columns = [
        "apachepatientresultsid",
        "patientunitstayid",
        "apacheversion",
        "predictedhospitalmortality",
    ]
    frame = pd.read_csv(
        root / APACHE_PATH,
        usecols=columns,
        dtype="string",
        low_memory=False,
    )
    return validate_apache_structure(frame, fixed_stays)


def score_locked_model(
    frame: pd.DataFrame, predictor_names: list[str], intercept: float, beta: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    matrix = frame[predictor_names].to_numpy(dtype=np.float64)
    eta = intercept + matrix @ beta
    probability = expit(eta)
    if not np.isfinite(eta).all() or not np.isfinite(probability).all():
        fail("locked harmonized score is nonfinite")
    if not np.all((probability >= 0) & (probability <= 1)):
        fail("locked harmonized probability is outside [0,1]")
    return eta, probability


def validate_probability_vector(probability: np.ndarray, label: str) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        fail(f"{label}: probability vector is malformed or nonfinite")
    if not np.all((values >= 0) & (values <= 1)):
        fail(f"{label}: probability outside the closed unit interval")
    return values


def calibration_eligibility(probability: np.ndarray) -> tuple[bool, str | None]:
    p = validate_probability_vector(probability, "calibration input")
    if np.any((p == 0.0) | (p == 1.0)):
        return False, "NOT_ESTIMABLE_BOUNDARY_PROBABILITY_NO_CLIPPING"
    return True, None


def auc_rank(y: np.ndarray, probability: np.ndarray) -> float:
    outcome = validate_binary_outcome(y, "AUROC outcome")
    positive = outcome == 1
    n_positive = int(positive.sum())
    n_negative = int(len(outcome) - n_positive)
    if not n_positive or not n_negative:
        fail("AUROC requires both outcome classes")
    ranks = rankdata(probability, method="average")
    value = (ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (
        n_positive * n_negative
    )
    if not np.isfinite(value):
        fail("AUROC is nonfinite")
    return float(value)


def _validated_design(design: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(design, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise CalibrationError("nonfinite_design", f"{label}: design is malformed or nonfinite")
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0 or singular[0] == 0:
        raise CalibrationError("rank_deficiency", f"{label}: design has zero rank")
    tolerance = DESIGN_RANK_RELATIVE_TOLERANCE * singular[0]
    rank = int(np.sum(singular > tolerance))
    if rank != matrix.shape[1]:
        raise CalibrationError("rank_deficiency", f"{label}: design is not full rank")
    condition = float(singular[0] / singular[-1])
    if not np.isfinite(condition) or condition > MAX_DESIGN_CONDITION_NUMBER:
        raise CalibrationError("unstable_design", f"{label}: design condition number is excessive")
    return matrix


def _fit_calibration_glm(y: np.ndarray, design: np.ndarray, label: str, offset=None):
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import (
        ConvergenceWarning,
        HessianInversionWarning,
        PerfectSeparationError,
        PerfectSeparationWarning,
        SingularMatrixWarning,
    )

    outcome = validate_binary_outcome(y, f"{label} outcome")
    if np.unique(outcome).size < 2:
        raise CalibrationError("single_outcome_class", f"{label}: both outcome classes required")
    matrix = _validated_design(design, label)
    try:
        with warnings.catch_warnings():
            for warning_type in (
                RuntimeWarning,
                PerfectSeparationWarning,
                SingularMatrixWarning,
                ConvergenceWarning,
                HessianInversionWarning,
            ):
                warnings.filterwarnings("error", category=warning_type)
            fit = sm.GLM(
                outcome, matrix, offset=offset, family=sm.families.Binomial()
            ).fit(maxiter=100, tol=1e-10, disp=0)
    except PerfectSeparationError as exc:
        raise CalibrationError("perfect_separation", f"{label}: perfect separation") from exc
    except (RuntimeWarning, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise CalibrationError("numerical_error", f"{label}: numerical fitting failure") from exc
    except Warning as exc:
        if isinstance(exc, PerfectSeparationWarning):
            reason = "perfect_separation"
        elif isinstance(exc, SingularMatrixWarning):
            reason = "rank_deficiency"
        elif isinstance(exc, ConvergenceWarning):
            reason = "nonconvergence"
        else:
            reason = "numerical_error"
        raise CalibrationError(reason, f"{label}: fitting warning treated as failure") from exc
    if not fit.converged:
        raise CalibrationError("nonconvergence", f"{label}: optimization did not converge")
    if not np.isfinite(fit.params).all():
        raise CalibrationError("nonfinite_estimate", f"{label}: coefficients are nonfinite")
    model_component = matrix @ fit.params
    fitted_logit = model_component
    if offset is not None:
        fitted_logit = fitted_logit + np.asarray(offset, dtype=np.float64)
    if not np.isfinite(fitted_logit).all():
        raise CalibrationError("nonfinite_estimate", f"{label}: fitted logits are nonfinite")
    if (
        np.max(np.abs(fit.params)) > MAX_ABSOLUTE_COEFFICIENT
        or np.max(np.abs(model_component)) > MAX_ABSOLUTE_FITTED_LOGIT
    ):
        raise CalibrationError("quasi_separation", f"{label}: prospectively defined instability limit exceeded")
    return fit


def calibration_intercept(y: np.ndarray, eta: np.ndarray) -> float:
    fit = _fit_calibration_glm(
        y, np.ones((len(y), 1), dtype=float), "calibration_intercept", offset=eta
    )
    return float(fit.params[0])


def calibration_slope(y: np.ndarray, eta: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y), dtype=float), eta])
    fit = _fit_calibration_glm(y, design, "calibration_slope")
    return float(fit.params[1])


def scalar_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float | str | None]:
    p = validate_probability_vector(probability, "metric input")
    outcome = validate_binary_outcome(y, "metric outcome")
    if np.unique(outcome).size < 2:
        fail("scalar metrics require both outcome classes")
    expected = float(p.sum())
    values = {
        "auroc": auc_rank(outcome, p),
        "brier": float(np.mean((outcome - p) ** 2)),
        "observed_over_expected": None if expected == 0 else float(outcome.sum() / expected),
        "observed_over_expected_status": "ESTIMABLE"
        if expected > 0
        else "NOT_ESTIMABLE_ZERO_EXPECTED_EVENTS",
        "observed_mortality": float(outcome.mean()),
        "mean_predicted_mortality": float(p.mean()),
        "observed_events": int(outcome.sum()),
        "expected_events": expected,
    }
    eligible, reason = calibration_eligibility(p)
    calibration_jobs = {}
    if eligible:
        eta = logit(p)
        calibration_jobs = {
            "calibration_intercept": lambda: calibration_intercept(outcome, eta),
            "calibration_slope": lambda: calibration_slope(outcome, eta),
        }
    for metric in ("calibration_intercept", "calibration_slope"):
        status_key = f"{metric}_status"
        reason_key = f"{metric}_reason"
        if not eligible:
            values[metric] = None
            values[status_key] = "NOT_ESTIMABLE"
            values[reason_key] = reason
            continue
        try:
            estimate = float(calibration_jobs[metric]())
            if not np.isfinite(estimate):
                raise CalibrationError(
                    "nonfinite_estimate", f"{metric}: point estimate is nonfinite"
                )
            values[metric] = estimate
            values[status_key] = "ESTIMABLE"
            values[reason_key] = None
        except CalibrationError as exc:
            values[metric] = None
            values[status_key] = "NOT_ESTIMABLE"
            values[reason_key] = exc.reason
        except Exception:
            values[metric] = None
            values[status_key] = "NOT_ESTIMABLE"
            values[reason_key] = "numerical_error"
    statuses = {
        values["calibration_intercept_status"], values["calibration_slope_status"]
    }
    values["calibration_status"] = (
        "ESTIMABLE" if statuses == {"ESTIMABLE"} else "NOT_FULLY_ESTIMABLE"
    )
    numeric_values = [v for v in values.values() if isinstance(v, (int, float, np.number))]
    if not all(np.isfinite(v) for v in numeric_values):
        fail("nonfinite scalar metric")
    return values


def paired_differences(
    harmonized: dict[str, float], comparator: dict[str, float]
) -> dict[str, float]:
    return {
        "auroc_difference": float(harmonized["auroc"] - comparator["auroc"]),
        "brier_difference": float(comparator["brier"] - harmonized["brier"]),
    }


def rcs_design(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    k = np.asarray(knots, dtype=float)
    if len(k) != 4 or not np.all(np.diff(k) > 1e-10):
        fail("calibration spline knots are not distinct")
    span = k[-1] - k[0]
    bases = [np.ones_like(x), x]
    for j in range(2):
        basis = (
            np.maximum(x - k[j], 0) ** 3
            - np.maximum(x - k[-2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
            + np.maximum(x - k[-1], 0) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])
        ) / (span * span)
        bases.append(basis)
    return np.column_stack(bases)


def calibration_curve(
    y: np.ndarray, eta: np.ndarray, knots: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    design = rcs_design(eta, knots)
    fit = _fit_calibration_glm(y, design, "calibration_curve")
    grid_logit = rcs_design(grid, knots) @ fit.params
    if not np.isfinite(grid_logit).all() or np.max(np.abs(grid_logit)) > MAX_ABSOLUTE_FITTED_LOGIT:
        raise CalibrationError("degenerate_curve", "calibration curve grid is unstable")
    curve = expit(grid_logit)
    if not np.isfinite(curve).all() or np.any((curve <= 0.0) | (curve >= 1.0)):
        raise CalibrationError("degenerate_curve", "calibration curve is nonfinite or saturated")
    return curve


def cluster_index_from_draw(
    sites: np.ndarray, hospitals: np.ndarray, chosen_positions: np.ndarray
) -> np.ndarray:
    site_indices = [np.flatnonzero(sites == hospital) for hospital in hospitals]
    if len(chosen_positions) != len(hospitals):
        fail("cluster draw must contain one selection per observed hospital")
    if np.any(chosen_positions < 0) or np.any(chosen_positions >= len(hospitals)):
        fail("cluster draw position is outside the hospital index")
    return np.concatenate([site_indices[int(position)] for position in chosen_positions])


def percentile_interval(
    values: np.ndarray, minimum_valid: int = MIN_VALID_BOOTSTRAPS
) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if len(valid) < minimum_valid:
        fail("insufficient valid bootstrap repetitions for percentile interval")
    low, high = np.percentile(valid, [2.5, 97.5])
    return float(low), float(high)


def verify_original_locked_reproduction(
    root: Path,
    y: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    recorded = json.loads((root / ORIGINAL_PERFORMANCE_PATH).read_text(encoding="utf-8"))
    reproduction = scalar_metrics(y, probability)
    tolerance = 1e-12
    expected_pairs = {
        "auroc": recorded["metrics"]["auroc"]["point"],
        "brier": recorded["metrics"]["brier"]["point"],
        "calibration_intercept": recorded["metrics"]["calibration_intercept"]["point"],
        "calibration_slope": recorded["metrics"]["calibration_slope"]["point"],
        "observed_over_expected": recorded["metrics"]["observed_over_expected"]["point"],
        "mean_predicted_mortality": recorded["mean_locked_predicted_risk"],
        "expected_events": recorded["expected_event_count"],
    }
    for metric in ("calibration_intercept", "calibration_slope"):
        if reproduction.get(f"{metric}_status") != "ESTIMABLE":
            fail(
                "full-cohort locked-model integrity reproduction requires an estimable "
                f"{metric}; reason={reproduction.get(f'{metric}_reason')}"
            )
    differences = assert_reproduction_within_tolerance(reproduction, expected_pairs, tolerance)
    return {"tolerance": tolerance, "maximum_absolute_difference": max(differences.values())}


def assert_reproduction_within_tolerance(
    observed: dict, expected: dict, tolerance: float
) -> dict[str, float]:
    if set(observed).issuperset(expected) is False:
        fail("integrity reproduction is missing an expected statistic")
    differences = {}
    for key, value in expected.items():
        try:
            observed_value = float(observed[key])
            expected_value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BenchmarkError(
                f"integrity reproduction statistic is missing or nonnumeric: {key}"
            ) from exc
        if not np.isfinite(observed_value) or not np.isfinite(expected_value):
            fail(f"integrity reproduction statistic is nonfinite: {key}")
        differences[key] = abs(observed_value - expected_value)
    if not differences or not np.isfinite(list(differences.values())).all():
        fail("integrity reproduction comparison is nonfinite or empty")
    if max(differences.values()) > tolerance:
        fail("full-cohort locked-model integrity reproduction does not match fixed performance")
    return differences


def bootstrap_analysis(
    y: np.ndarray,
    sites: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    point_metrics: dict[str, dict] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    minimum_valid: int = MIN_VALID_BOOTSTRAPS,
    seed: int = BOOTSTRAP_SEED,
    expected_hospitals: int = EXPECTED_COMMON_HOSPITALS,
) -> dict:
    outcome = validate_binary_outcome(y, "bootstrap outcome")
    if len(outcome) != len(sites):
        fail("bootstrap outcome and hospital vectors differ in length")
    if set(predictions) != set(MODEL_ORDER):
        fail("bootstrap prediction model labels differ from the fixed model set")
    validated_predictions = {
        model: validate_probability_vector(probability, model)
        for model, probability in predictions.items()
    }
    if any(len(probability) != len(outcome) for probability in validated_predictions.values()):
        fail("bootstrap prediction vectors differ in length")
    hospitals = np.unique(sites)
    if len(hospitals) != expected_hospitals:
        fail("common-cohort hospital count differs before bootstrap")
    rng = np.random.Generator(np.random.PCG64(seed))

    scalar_names = (
        "auroc",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "observed_over_expected",
    )
    samples = {
        f"{model}::{metric}": np.full(n_bootstrap, np.nan, dtype=float)
        for model in MODEL_ORDER
        for metric in scalar_names
    }
    paired_keys = []
    for comparator in ("APACHE IV", "APACHE IVa"):
        for metric in ("auroc_difference", "brier_difference"):
            key = f"{comparator}::{metric}"
            paired_keys.append(key)
            samples[key] = np.full(n_bootstrap, np.nan, dtype=float)

    calibration_status = {}
    calibration_scalar_status: dict[str, dict[str, str]] = {}
    calibration_scalar_reason: dict[str, dict[str, str | None]] = {}
    calibration_curve_status: dict[str, str] = {}
    calibration_curve_reason: dict[str, str | None] = {}
    active_sample_keys = {
        f"{model}::{metric}"
        for model in MODEL_ORDER
        for metric in ("auroc", "brier")
    } | set(paired_keys)
    observed_over_expected_status = {}

    curve_spec: dict[str, dict[str, np.ndarray]] = {}
    point_curves: dict[str, np.ndarray] = {}
    curves: dict[str, np.ndarray] = {}
    for model, probability in validated_predictions.items():
        model_points = (
            point_metrics[model]
            if point_metrics is not None
            else scalar_metrics(outcome, probability)
        )
        calibration_scalar_status[model] = {}
        calibration_scalar_reason[model] = {}
        for metric in ("calibration_intercept", "calibration_slope"):
            metric_status = model_points.get(f"{metric}_status")
            metric_reason = model_points.get(f"{metric}_reason")
            if metric_status not in {"ESTIMABLE", "NOT_ESTIMABLE"}:
                fail(f"invalid point-estimate status for {model} {metric}")
            point_value = model_points.get(metric)
            if metric_status == "ESTIMABLE" and (
                point_value is None or not np.isfinite(float(point_value))
            ):
                fail(f"estimable point calibration statistic is missing for {model} {metric}")
            if metric_status == "NOT_ESTIMABLE" and point_value is not None:
                fail(f"non-estimable point calibration statistic has a value for {model} {metric}")
            calibration_scalar_status[model][metric] = metric_status
            calibration_scalar_reason[model][metric] = metric_reason
            if metric_status == "ESTIMABLE":
                active_sample_keys.add(f"{model}::{metric}")
        if probability.sum() > 0:
            observed_over_expected_status[model] = "ESTIMABLE"
            active_sample_keys.add(f"{model}::observed_over_expected")
        else:
            observed_over_expected_status[model] = "NOT_ESTIMABLE_ZERO_EXPECTED_EVENTS"
        eligible, reason = calibration_eligibility(probability)
        calibration_status[model] = "ESTIMABLE" if eligible else reason
        if not eligible:
            calibration_curve_status[model] = "NOT_ESTIMABLE"
            calibration_curve_reason[model] = "boundary_probability_no_clipping"
            continue
        eta = logit(probability)
        knots = np.quantile(eta, [0.05, 0.35, 0.65, 0.95])
        grid = np.linspace(float(np.quantile(eta, 0.01)), float(np.quantile(eta, 0.99)), CURVE_GRID_N)
        try:
            point_curve = calibration_curve(outcome, eta, knots, grid)
        except CalibrationError as exc:
            calibration_curve_status[model] = "NOT_ESTIMABLE"
            calibration_curve_reason[model] = exc.reason
            continue
        except BenchmarkError as exc:
            calibration_curve_status[model] = "NOT_ESTIMABLE"
            calibration_curve_reason[model] = (
                "duplicate_knots"
                if "knots are not distinct" in str(exc)
                else "numerical_error"
            )
            continue
        except Exception:
            calibration_curve_status[model] = "NOT_ESTIMABLE"
            calibration_curve_reason[model] = "numerical_error"
            continue
        calibration_curve_status[model] = "ESTIMABLE"
        calibration_curve_reason[model] = None
        curve_spec[model] = {"knots": knots, "grid": grid}
        point_curves[model] = point_curve
        curves[model] = np.full((n_bootstrap, CURVE_GRID_N), np.nan, dtype=float)

    failures: Counter[str] = Counter()
    failure_events: list[dict[str, int | str]] = []

    def record_failure(repetition: int, statistic: str, reason: str) -> None:
        failures[f"{statistic}::{reason}"] += 1
        failure_events.append(
            {"repetition": int(repetition), "statistic": statistic, "reason": reason}
        )

    for repetition in range(n_bootstrap):
        chosen = rng.integers(0, len(hospitals), size=len(hospitals))
        index = cluster_index_from_draw(sites, hospitals, chosen)
        yy = outcome[index]
        if np.unique(yy).size < 2:
            for key in sorted(active_sample_keys):
                record_failure(repetition, key, "single_outcome_class")
            for model in curves:
                record_failure(repetition, f"{model}::calibration_curve", "single_outcome_class")
            continue

        replicate_metrics: dict[str, dict[str, float]] = {}
        for model in MODEL_ORDER:
            pp = validated_predictions[model][index]
            model_metrics: dict[str, float] = {}
            jobs = {
                "auroc": lambda: auc_rank(yy, pp),
                "brier": lambda: float(np.mean((yy - pp) ** 2)),
            }
            if observed_over_expected_status[model] == "ESTIMABLE":
                jobs["observed_over_expected"] = lambda: float(yy.sum() / pp.sum())
            if calibration_status[model] == "ESTIMABLE":
                eta = logit(pp)
                if calibration_scalar_status[model]["calibration_intercept"] == "ESTIMABLE":
                    jobs["calibration_intercept"] = lambda: calibration_intercept(yy, eta)
                if calibration_scalar_status[model]["calibration_slope"] == "ESTIMABLE":
                    jobs["calibration_slope"] = lambda: calibration_slope(yy, eta)
            for metric, job in jobs.items():
                try:
                    value = float(job())
                    if not np.isfinite(value):
                        fail("nonfinite bootstrap statistic")
                    samples[f"{model}::{metric}"][repetition] = value
                    model_metrics[metric] = value
                except CalibrationError as exc:  # recorded; never redrawn
                    record_failure(repetition, f"{model}::{metric}", exc.reason)
                except Exception as exc:  # recorded; never redrawn
                    reason = "nonfinite_estimate" if "nonfinite" in str(exc).lower() else "numerical_error"
                    record_failure(repetition, f"{model}::{metric}", reason)
            replicate_metrics[model] = model_metrics

            if model not in curves:
                continue
            try:
                spec = curve_spec[model]
                curves[model][repetition] = calibration_curve(
                    yy, eta, spec["knots"], spec["grid"]
                )
            except CalibrationError as exc:  # recorded; never redrawn
                record_failure(repetition, f"{model}::calibration_curve", exc.reason)
            except Exception:  # recorded; never redrawn
                record_failure(repetition, f"{model}::calibration_curve", "numerical_error")

        harmonized = replicate_metrics["Locked harmonized MIMIC Cr/RRT"]
        for comparator in ("APACHE IV", "APACHE IVa"):
            other = replicate_metrics[comparator]
            if "auroc" in harmonized and "auroc" in other:
                samples[f"{comparator}::auroc_difference"][repetition] = (
                    harmonized["auroc"] - other["auroc"]
                )
            else:
                record_failure(repetition, f"{comparator}::auroc_difference", "constituent_failure")
            if "brier" in harmonized and "brier" in other:
                samples[f"{comparator}::brier_difference"][repetition] = (
                    other["brier"] - harmonized["brier"]
                )
            else:
                record_failure(repetition, f"{comparator}::brier_difference", "constituent_failure")

    valid_counts = {key: int(np.isfinite(values).sum()) for key, values in samples.items()}
    curve_valid_counts = {
        model: int(np.isfinite(values).all(axis=1).sum()) for model, values in curves.items()
    }
    inadequate = {
        key: value
        for key, value in valid_counts.items()
        if key in active_sample_keys and value < minimum_valid
    }
    inadequate.update(
        {
            f"{model}::calibration_curve": value
            for model, value in curve_valid_counts.items()
            if value < minimum_valid
        }
    )
    if inadequate:
        fail(
            "bootstrap validity threshold failed; "
            f"valid_counts={inadequate}; failures={dict(failures)}"
        )

    intervals = {
        key: percentile_interval(values, minimum_valid)
        for key, values in samples.items()
        if key in active_sample_keys
    }
    curve_intervals = {}
    for model, values in curves.items():
        lower, upper = np.nanpercentile(values, [2.5, 97.5], axis=0)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            fail("nonfinite calibration-curve interval")
        curve_intervals[model] = {"lower": lower, "upper": upper}
    return {
        "samples": samples,
        "intervals": intervals,
        "valid_counts": valid_counts,
        "active_sample_keys": sorted(active_sample_keys),
        "curve_spec": curve_spec,
        "point_curves": point_curves,
        "curves": curves,
        "curve_intervals": curve_intervals,
        "curve_valid_counts": curve_valid_counts,
        "failures": dict(failures),
        "failure_events": failure_events,
        "calibration_status": calibration_status,
        "calibration_scalar_status": calibration_scalar_status,
        "calibration_scalar_reason": calibration_scalar_reason,
        "calibration_curve_status": calibration_curve_status,
        "calibration_curve_reason": calibration_curve_reason,
        "observed_over_expected_status": observed_over_expected_status,
        "n_bootstrap": n_bootstrap,
        "minimum_valid": minimum_valid,
        "seed": seed,
    }


def write_csv_exclusive(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        fail(f"refusing to overwrite output: {path.name}")
    frame.to_csv(path, index=False)


def write_json_exclusive(obj: dict, path: Path) -> None:
    if path.exists():
        fail(f"refusing to overwrite output: {path.name}")
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


EXPECTED_OUTPUT_NAMES = {
    "MAIN_COMPARISON_TABLE.csv",
    "PAIRED_COMPARISON_TABLE.csv",
    "PATIENT_FLOW.csv",
    "SECONDARY_CALIBRATION_UNCERTAINTY.csv",
    "CALIBRATION_CURVES.png",
    "CALIBRATION_CURVE_DATA.csv",
    "APACHE_AVAILABILITY_SELECTION_SUMMARY.csv",
    "BOOTSTRAP_FAILURE_SUMMARY.json",
    "ANALYSIS_MANIFEST.json",
}


def validate_completed_staging(staging: Path) -> None:
    names = {path.name for path in staging.iterdir() if path.is_file()}
    if names != EXPECTED_OUTPUT_NAMES:
        fail("staging output set is incomplete or contains unexpected files")
    manifest_path = staging / "ANALYSIS_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPUTED" or manifest.get("verification_scope") != "calculation_only":
        fail("staging completion manifest does not declare completion")
    expected_hashes = manifest.get("output_hashes", {})
    hash_names = EXPECTED_OUTPUT_NAMES - {"ANALYSIS_MANIFEST.json"}
    if set(expected_hashes) != hash_names:
        fail("completion manifest output inventory differs")
    for name in sorted(hash_names):
        if sha256_file(staging / name) != expected_hashes[name]:
            fail("staging output hash verification failed")


def transactional_publish(final_dir: Path, producer) -> Path:
    """Publish an already authorized aggregate result atomically after staged validation.

    Computation and file creation are not transactional; only completed-directory
    publication is atomic. A rejected staging directory is kept separate from published results.
    """
    if final_dir.exists():
        fail("output directory already exists; refusing to overwrite or append")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_dir.name}.working-", dir=final_dir.parent)
    )
    try:
        producer(staging)
        validate_completed_staging(staging)
        if final_dir.exists():
            fail("final output destination appeared during staged generation")
        os.replace(staging, final_dir)
        return final_dir
    except BaseException as exc:
        marker = staging / "RUN_REJECTED.json"
        if staging.exists() and not marker.exists():
            try:
                write_json_exclusive(
                    {
                        "status": "REJECTED_NOT_FOR_SCIENTIFIC_USE",
                        "failed_utc": utc_now(),
                        "failure_type": type(exc).__name__,
                        "contains_patient_level_information": False,
                    },
                    marker,
                )
            except Exception:
                pass
        raise


def derive_selection_summary(
    evaluable: pd.DataFrame, membership: dict[str, set[int]]
) -> pd.DataFrame:
    labels = (
        ("full fixed evaluable population", "full_fixed_evaluable"),
        ("APACHE record available", "record_available"),
        ("no APACHE record", "no_record"),
        ("APACHE record with non-predictive sentinel", "nonpredictive_sentinel"),
        ("valid common comparison population", "valid_common"),
        ("all patients without a valid APACHE prediction", "without_valid_prediction"),
    )
    stays = set(evaluable["stay_id"].astype(np.int64))
    if membership["full_fixed_evaluable"] != stays:
        fail("selection summary full population does not match fixed evaluable membership")
    if membership["valid_common"] | membership["without_valid_prediction"] != stays:
        fail("valid and non-valid APACHE categories do not reconcile to the full population")
    if membership["valid_common"] & membership["without_valid_prediction"]:
        fail("valid and non-valid APACHE categories overlap")
    if membership["record_available"] | membership["no_record"] != stays:
        fail("record-availability categories do not reconcile to the full population")
    if membership["record_available"] & membership["no_record"]:
        fail("record-availability categories overlap")
    if membership["valid_common"] | membership["nonpredictive_sentinel"] != membership["record_available"]:
        fail("APACHE sentinel and valid categories do not reconcile to record availability")

    rows = []
    raw_counts = {}
    for label, key in labels:
        subset = evaluable.loc[evaluable["stay_id"].isin(membership[key])]
        n = int(len(subset))
        deaths = int(subset[OUTCOME].sum())
        survivors = n - deaths
        hospitals = int(subset["hospital_id"].nunique())
        raw_counts[key] = (n, deaths)
        suppress = min(n, deaths, survivors) < SMALL_CELL_SUPPRESSION_THRESHOLD
        rows.append(
            {
                "group": label,
                "n": n,
                "deaths": None if suppress else deaths,
                "survivors": None if suppress else survivors,
                "hospitals": hospitals,
                "mortality_proportion": None if suppress else float(deaths / n),
                "small_cell_suppressed": suppress,
            }
        )
    total_n, total_deaths = raw_counts["full_fixed_evaluable"]
    for left, right in (
        ("record_available", "no_record"),
        ("valid_common", "without_valid_prediction"),
    ):
        if raw_counts[left][0] + raw_counts[right][0] != total_n:
            fail("selection patient counts fail reconciliation")
        if raw_counts[left][1] + raw_counts[right][1] != total_deaths:
            fail("selection death counts fail reconciliation")
    if (
        raw_counts["valid_common"][0] + raw_counts["nonpredictive_sentinel"][0]
        != raw_counts["record_available"][0]
        or raw_counts["valid_common"][1] + raw_counts["nonpredictive_sentinel"][1]
        != raw_counts["record_available"][1]
    ):
        fail("APACHE sentinel selection counts fail reconciliation")
    return pd.DataFrame(rows)


def prepare_inputs(root: Path, manifests: dict[Path, Path] | None = None) -> dict:
    verified = verify_hashes(root, manifests)
    predictors, intercept, beta = load_coefficients(root / COEFFICIENT_PATH)
    outcome_manifest = (manifests or {}).get(OUTCOME_LOCK_PATH, OUTCOME_LOCK_PATH)
    evaluable = load_fixed_evaluable(root, predictors, outcome_manifest)
    apache, apache_audit, selection_membership = load_apache_minimal(
        root, set(evaluable["stay_id"].astype(int))
    )
    common_ids = apache.index.to_numpy(dtype=np.int64)
    common = evaluable.set_index("stay_id").loc[common_ids].reset_index()
    assert_identity_unique(common, "common comparison cohort")
    if len(common) != EXPECTED_COMMON_N:
        fail("common comparison row count differs")
    if common["hospital_id"].nunique() != EXPECTED_COMMON_HOSPITALS:
        fail("common comparison hospital count differs")
    if int(common[OUTCOME].sum()) != EXPECTED_COMMON_EVENTS:
        fail("common comparison event count differs")
    if not np.array_equal(common["stay_id"].to_numpy(dtype=np.int64), common_ids):
        fail("common cohort and APACHE identity order differ")
    lock_path = root / MEMBERSHIP_LOCK_PATH
    if not lock_path.is_file():
        fail("pinned common-cohort membership lock is missing")
    membership_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    membership_verification = verify_membership_lock(
        membership_lock, common_ids, verified
    )
    return {
        "verified_hashes": verified,
        "predictors": predictors,
        "intercept": intercept,
        "beta": beta,
        "evaluable": evaluable,
        "apache": apache,
        "apache_audit": apache_audit,
        "selection_membership": selection_membership,
        "common": common,
        "membership_lock": membership_lock,
        "membership_verification": membership_verification,
    }


def preflight(root: Path, manifests: dict[Path, Path] | None = None) -> dict:
    implemented = prepare_inputs(root, manifests)
    return {
        "status": "PASS; NO MODEL SCORES OR PERFORMANCE METRICS COMPUTED",
        "verified_file_hashes": implemented["verified_hashes"],
        "fixed_evaluable_n": int(len(implemented["evaluable"])),
        "fixed_evaluable_hospitals": int(implemented["evaluable"]["hospital_id"].nunique()),
        "fixed_evaluable_events_verified_for_identity_gate": int(implemented["evaluable"][OUTCOME].sum()),
        "apache_structure": implemented["apache_audit"],
        "common_n": int(len(implemented["common"])),
        "common_hospitals": int(implemented["common"]["hospital_id"].nunique()),
        "common_events_verified_for_identity_gate": int(implemented["common"][OUTCOME].sum()),
        "common_membership_lock": implemented["membership_verification"],
    }


def _run_authorized_to_directory(
    root: Path, output_dir: Path, manifests: dict[Path, Path] | None = None,
) -> None:
    import statsmodels

    implemented = prepare_inputs(root, manifests)
    evaluable = implemented["evaluable"]
    common = implemented["common"]
    predictors = implemented["predictors"]

    _, full_locked_probability = score_locked_model(
        evaluable, predictors, implemented["intercept"], implemented["beta"]
    )
    integrity_reproduction = verify_original_locked_reproduction(
        root,
        evaluable[OUTCOME].to_numpy(dtype=np.int8),
        full_locked_probability,
    )

    _, common_locked_probability = score_locked_model(
        common, predictors, implemented["intercept"], implemented["beta"]
    )
    apache = implemented["apache"]
    predictions = {
        "Locked harmonized MIMIC Cr/RRT": common_locked_probability,
        "APACHE IV": validate_probability_vector(apache["IV"].to_numpy(), "APACHE IV"),
        "APACHE IVa": validate_probability_vector(apache["IVa"].to_numpy(), "APACHE IVa"),
    }
    y = common[OUTCOME].to_numpy(dtype=np.int8)
    sites = common["hospital_id"].to_numpy(dtype=np.int64)
    if any(len(values) != len(y) for values in predictions.values()):
        fail("prediction and outcome vector lengths differ")

    points = {model: scalar_metrics(y, probability) for model, probability in predictions.items()}
    paired = {
        comparator: paired_differences(
            points["Locked harmonized MIMIC Cr/RRT"], points[comparator]
        )
        for comparator in ("APACHE IV", "APACHE IVa")
    }
    bootstrap = bootstrap_analysis(y, sites, predictions, point_metrics=points)

    main_rows = []
    for model in MODEL_ORDER:
        metric = points[model]
        main_rows.append(
            {
                "model": model,
                "n": len(y),
                "deaths": int(y.sum()),
                "hospitals": int(np.unique(sites).size),
                "observed_mortality": metric["observed_mortality"],
                "mean_predicted_mortality": metric["mean_predicted_mortality"],
                "auroc": metric["auroc"],
                "auroc_ci95_low": bootstrap["intervals"][f"{model}::auroc"][0],
                "auroc_ci95_high": bootstrap["intervals"][f"{model}::auroc"][1],
                "brier": metric["brier"],
                "brier_ci95_low": bootstrap["intervals"][f"{model}::brier"][0],
                "brier_ci95_high": bootstrap["intervals"][f"{model}::brier"][1],
                "calibration_intercept": metric["calibration_intercept"],
                "calibration_intercept_status": metric["calibration_intercept_status"],
                "calibration_intercept_reason": metric["calibration_intercept_reason"],
                "calibration_slope": metric["calibration_slope"],
                "calibration_slope_status": metric["calibration_slope_status"],
                "calibration_slope_reason": metric["calibration_slope_reason"],
                "observed_over_expected": metric["observed_over_expected"],
                "calibration_status": metric["calibration_status"],
                "population_note": "restricted common APACHE-valid survivor-conditioned +24h HF-AKI cohort",
                "apache_timing_note": "native APACHE predictions originate from first-day assessment; exact +24h temporal equivalence is not established",
            }
        )
    write_csv_exclusive(pd.DataFrame(main_rows), output_dir / "MAIN_COMPARISON_TABLE.csv")

    paired_rows = []
    for comparator in ("APACHE IV", "APACHE IVa"):
        paired_rows.append(
            {
                "comparator": comparator,
                "paired_auroc_difference_harmonized_minus_apache": paired[comparator]["auroc_difference"],
                "paired_auroc_difference_ci95_low": bootstrap["intervals"][f"{comparator}::auroc_difference"][0],
                "paired_auroc_difference_ci95_high": bootstrap["intervals"][f"{comparator}::auroc_difference"][1],
                "paired_brier_difference_apache_minus_harmonized": paired[comparator]["brier_difference"],
                "paired_brier_difference_ci95_low": bootstrap["intervals"][f"{comparator}::brier_difference"][0],
                "paired_brier_difference_ci95_high": bootstrap["intervals"][f"{comparator}::brier_difference"][1],
                "orientation": "positive values favor locked harmonized model",
                "interpretation": "post-hoc exploratory benchmark; no p-value or model selection",
            }
        )
    write_csv_exclusive(pd.DataFrame(paired_rows), output_dir / "PAIRED_COMPARISON_TABLE.csv")

    flow = pd.DataFrame(
        [
            {"step": "fixed outcome-evaluable", "n": len(evaluable)},
            {
                "step": "APACHE IV and IVa records available",
                "n": len(implemented["selection_membership"]["record_available"]),
            },
            {"step": "valid identical common comparison cohort", "n": len(common)},
        ]
    )
    write_csv_exclusive(flow, output_dir / "PATIENT_FLOW.csv")

    secondary_rows = []
    for model in MODEL_ORDER:
        for metric in ("calibration_intercept", "calibration_slope", "observed_over_expected"):
            key = f"{model}::{metric}"
            estimable = key in bootstrap["intervals"] and points[model][metric] is not None
            if metric == "observed_over_expected":
                metric_status = bootstrap["observed_over_expected_status"][model]
                metric_reason = (
                    None if metric_status == "ESTIMABLE" else "zero_expected_events"
                )
            else:
                metric_status = points[model][f"{metric}_status"]
                metric_reason = points[model][f"{metric}_reason"]
            secondary_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "estimate": points[model][metric],
                    "ci95_low": bootstrap["intervals"][key][0] if estimable else None,
                    "ci95_high": bootstrap["intervals"][key][1] if estimable else None,
                    "valid_bootstrap_repetitions": (
                        bootstrap["valid_counts"][key]
                        if metric_status == "ESTIMABLE"
                        else None
                    ),
                    "status": "ESTIMABLE" if estimable else metric_status,
                    "reason": None if estimable else metric_reason,
                    "note": "original probabilities unchanged; no clipping, recalibration, or patient exclusion",
                }
            )
    write_csv_exclusive(
        pd.DataFrame(secondary_rows),
        output_dir / "SECONDARY_CALIBRATION_UNCERTAINTY.csv",
    )

    curve_rows = []
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    colors = {
        "Locked harmonized MIMIC Cr/RRT": "navy",
        "APACHE IV": "darkorange",
        "APACHE IVa": "seagreen",
    }
    for model in MODEL_ORDER:
        if model not in bootstrap["curve_spec"]:
            curve_rows.append(
                {
                    "model": model,
                    "curve_status": bootstrap["calibration_curve_status"][model],
                    "curve_reason": bootstrap["calibration_curve_reason"][model],
                    "predicted_probability": None,
                    "fitted_observed_probability": None,
                    "pointwise_ci95_low": None,
                    "pointwise_ci95_high": None,
                    "plotted_prediction_min": None,
                    "plotted_prediction_max": None,
                }
            )
            continue
        spec = bootstrap["curve_spec"][model]
        point_curve = bootstrap["point_curves"][model]
        lower = bootstrap["curve_intervals"][model]["lower"]
        upper = bootstrap["curve_intervals"][model]["upper"]
        x = expit(spec["grid"])
        plotted_min = float(x[0])
        plotted_max = float(x[-1])
        axis.plot(
            x,
            point_curve,
            color=colors[model],
            linewidth=2,
            label=f"{model} [{plotted_min:.3f}, {plotted_max:.3f}]",
        )
        axis.fill_between(x, lower, upper, color=colors[model], alpha=0.12)
        for xx, yy, lo, hi in zip(x, point_curve, lower, upper):
            curve_rows.append(
                {
                    "model": model,
                    "curve_status": "ESTIMABLE",
                    "curve_reason": None,
                    "predicted_probability": float(xx),
                    "fitted_observed_probability": float(yy),
                    "pointwise_ci95_low": float(lo),
                    "pointwise_ci95_high": float(hi),
                    "plotted_prediction_min": plotted_min,
                    "plotted_prediction_max": plotted_max,
                }
            )
    distribution_axis = axis.inset_axes([0.57, 0.08, 0.39, 0.22])
    for model in MODEL_ORDER:
        distribution_axis.hist(
            predictions[model],
            bins=24,
            range=(0, 1),
            density=True,
            histtype="step",
            linewidth=1,
            color=colors[model],
        )
    distribution_axis.set_title("Original risk distributions", fontsize=8)
    distribution_axis.set_xlabel("Predicted risk", fontsize=7)
    distribution_axis.set_ylabel("Density", fontsize=7)
    distribution_axis.tick_params(labelsize=7)
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Ideal")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Original predicted hospital mortality probability")
    axis.set_ylabel("Fitted observed subsequent in-hospital mortality")
    axis.legend(loc="best", title="Model [plotted prediction-risk range]")
    axis.set_title("Calibration in the fixed APACHE-valid common cohort")
    axis.text(
        0.01,
        -0.16,
        "Descriptive only: survivor-conditioned +24h HF-AKI cohort. APACHE is a native first-day assessment; "
        "curves show only each model's 1st-99th percentile prediction range, not unsupported extreme tails.",
        transform=axis.transAxes,
        fontsize=8,
        va="top",
        wrap=True,
    )
    nonestimable_curves = [
        f"{model}: {bootstrap['calibration_curve_reason'][model]}"
        for model in MODEL_ORDER
        if bootstrap["calibration_curve_status"][model] != "ESTIMABLE"
    ]
    if nonestimable_curves:
        axis.text(
            0.01,
            0.99,
            "Calibration curve NOT_ESTIMABLE -- " + "; ".join(nonestimable_curves),
            transform=axis.transAxes,
            fontsize=8,
            va="top",
            color="darkred",
            wrap=True,
        )
    figure.tight_layout()
    figure_path = output_dir / "CALIBRATION_CURVES.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    write_csv_exclusive(pd.DataFrame(curve_rows), output_dir / "CALIBRATION_CURVE_DATA.csv")

    selection = derive_selection_summary(
        evaluable, implemented["selection_membership"]
    )
    write_csv_exclusive(
        selection, output_dir / "APACHE_AVAILABILITY_SELECTION_SUMMARY.csv"
    )

    failure_summary = {
        "bootstrap_repetitions": N_BOOTSTRAP,
        "minimum_valid_required": MIN_VALID_BOOTSTRAPS,
        "valid_scalar_and_difference_counts": {
            key: bootstrap["valid_counts"][key]
            for key in bootstrap["active_sample_keys"]
        },
        "valid_complete_curve_counts": bootstrap["curve_valid_counts"],
        "failures_by_reason": bootstrap["failures"],
        "failure_events": bootstrap["failure_events"],
        "logit_input_eligibility_status": bootstrap["calibration_status"],
        "point_calibration_scalar_status": bootstrap["calibration_scalar_status"],
        "point_calibration_scalar_reason": bootstrap["calibration_scalar_reason"],
        "point_calibration_curve_status": bootstrap["calibration_curve_status"],
        "point_calibration_curve_reason": bootstrap["calibration_curve_reason"],
        "observed_over_expected_status": bootstrap["observed_over_expected_status"],
        "no_failed_draw_redrawn": True,
    }
    write_json_exclusive(
        failure_summary, output_dir / "BOOTSTRAP_FAILURE_SUMMARY.json"
    )

    outputs = [path for path in output_dir.iterdir() if path.is_file()]
    output_hashes = {path.name: sha256_file(path) for path in sorted(outputs)}
    manifest = {
        "status": "COMPUTED",
        "verification_scope": "calculation_only",
        "created_utc": utc_now(),
        "protocol_character": "post-hoc exploratory benchmark",
        "models": list(MODEL_ORDER),
        "common_population": {
            "n": len(y),
            "events": int(y.sum()),
            "hospitals": int(np.unique(sites).size),
        },
        "bootstrap": {
            "repetitions": N_BOOTSTRAP,
            "seed": BOOTSTRAP_SEED,
            "unit": "hospital",
            "interval": "two-sided 95% percentile",
            "same_draws_for_all_models_and_differences": True,
            "rng": "numpy.random.Generator(numpy.random.PCG64)",
            "minimum_valid": MIN_VALID_BOOTSTRAPS,
        },
        "calibration": {
            "boundary_policy": "zero/one retained for primary metrics; logit calibration NOT ESTIMABLE; no clipping",
            "rank_relative_tolerance": DESIGN_RANK_RELATIVE_TOLERANCE,
            "maximum_design_condition_number": MAX_DESIGN_CONDITION_NUMBER,
            "maximum_absolute_fitted_logit": MAX_ABSOLUTE_FITTED_LOGIT,
            "maximum_absolute_coefficient": MAX_ABSOLUTE_COEFFICIENT,
            "knots_quantiles": [0.05, 0.35, 0.65, 0.95],
            "grid_quantiles": [0.01, 0.99],
            "grid_points": CURVE_GRID_N,
            "knots_and_grids_fixed_across_bootstrap": True,
            "logit_input_eligibility_status": bootstrap["calibration_status"],
            "point_scalar_status": bootstrap["calibration_scalar_status"],
            "point_scalar_reason": bootstrap["calibration_scalar_reason"],
            "point_curve_status": bootstrap["calibration_curve_status"],
            "point_curve_reason": bootstrap["calibration_curve_reason"],
        },
        "input_hashes": implemented["verified_hashes"],
        "preparation_hashes": {
            "analysis_script": sha256_file(Path(__file__).resolve()),
            "protocol": sha256_file(
                Path(__file__).resolve().with_name("EICU_APACHE_BENCHMARK_PROTOCOL.md")
            ),
            "synthetic_test_suite": sha256_file(
                Path(__file__).resolve().with_name("EICU_APACHE_BENCHMARK_TESTS.py")
            ),
            "membership_lock": sha256_file(root / MEMBERSHIP_LOCK_PATH),
        },
        "output_hashes": output_hashes,
        "original_locked_integrity_reproduction": integrity_reproduction,
        "common_membership_lock": implemented["membership_verification"],
        "no_model_refit": True,
        "no_imputer_refit": True,
        "no_recalibration": True,
        "no_apache_native_actual_mortality_used": True,
        "no_patient_level_output": True,
        "software": {
            "python": platform.python_version(),
            "python_full": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": __import__("matplotlib").__version__,
            "numpy_configuration": str(np.show_config(mode="dicts")),
        },
        "limitations": [
            "post-hoc exploratory comparison in the restricted common APACHE-valid cohort, not the full 1,897-patient external-validation population",
            "population is conditioned on survival and continued eligibility to ICU admission plus 24 hours",
            "native APACHE first-day timing is not established as exactly equivalent to the study +24-hour landmark",
            "hospital bootstrap conditions on fixed predictions, coefficients, and imputation and does not propagate development uncertainty",
            "calibration curves are descriptive over model-specific 1st-99th percentile prediction ranges",
        ],
    }
    write_json_exclusive(manifest, output_dir / "ANALYSIS_MANIFEST.json")
    return None


def run_authorized(
    root: Path, output_dir: Path, authorization_token: str | None = None,
    manifests: dict[Path, Path] | None = None,
) -> None:
    if authorization_token != AUTHORIZATION_TOKEN:
        fail(
            "real-outcome execution is not authorized: exact accidental-execution guard required; "
            "this token is not authentication or a security boundary"
        )
    published = transactional_publish(
        output_dir, lambda staging: _run_authorized_to_directory(root, staging, manifests)
    )
    print(f"AUTHORIZED AGGREGATE ANALYSIS COMPLETE: {published}")
    print("APACHE BENCHMARK: computational outputs created; numerical verification is a separate procedure")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    add_manifest_arguments(parser)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--authorization-token")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    try:
        if args.preflight:
            if args.output_dir is not None or args.authorization_token is not None:
                fail("preflight accepts neither an output directory nor authorization token")
            result = preflight(root, manifest_paths(args))
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if args.authorization_token != AUTHORIZATION_TOKEN:
            fail("real-outcome execution is not authorized: exact token required")
        if args.output_dir is None:
            fail("authorized run requires --output-dir")
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = root / output_dir
        output_dir = output_dir.resolve()
        allowed_base = (root / "work" / "authorized_results").resolve()
        if output_dir == allowed_base or not output_dir.is_relative_to(allowed_base):
            fail("output directory must be a new child of work/authorized_results")
        run_authorized(root, output_dir, args.authorization_token, manifest_paths(args))
        return 0
    except BenchmarkError as exc:
        print(f"HARD STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
