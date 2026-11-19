#!/usr/bin/env python3
"""Public harmonized-model source check and restricted harmonized-model audit.

Public mode checks declared model terms and imputation specifications.
Restricted mode checks the authorized harmonized-model artifacts when supplied.
Neither mode uses eICU outcomes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

EXPECTED_SOURCE_SCRIPT_SHA256 = "3b0350bb0059f5084f7522368e87e13dac18d57322275b56c29d9dfbd957fdec"
EXPECTED_HARMONIZED_MODEL_MANIFEST_SHA256 = "e598d921adaf499efd0081fdca78f56563c9982c0acc4d084fff5f53287e01a8"
EXPECTED_N = 2098
EXPECTED_EVENTS = 537
EXPECTED_STAGE_COUNTS = {1: 1516, 2: 237, 3: 345}
OUTCOME = "hospital_expire_flag"
PROHIBITED_PREOUTCOME_TOKENS = (
    "death", "mortality", "expire", "outcome", "deathtime",
    "survival", "brier", "auc", "auroc", "calibration",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_manifest_record(manifest: dict) -> dict:
    """Extract a unique pre-outcome record using its documented field structure.

    This checks the authenticated manifest without requiring a particular
    wrapper key. An absent or ambiguous record is an error, never a silent skip.
    """
    required = {
        "cohort_path", "predictor_matrix_path",
        "subject_id_set_sha256_sorted_newline", "stay_id_set_sha256_sorted_newline",
    }
    matches = [v for v in manifest.values() if isinstance(v, dict) and required.issubset(v)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one pre-outcome record in the authorized manifest")
    return matches[0]


def get_authorization_flag(manifest: dict) -> tuple[str, bool]:
    """Find the unique gate decision flag in a checksum-authenticated manifest."""
    matches = [(k, v) for k, v in manifest.items() if k.startswith("gate_") and type(v) is bool]
    if len(matches) != 1:
        raise ValueError("Expected exactly one boolean gate decision in the authorized manifest")
    return matches[0]


def sha256_sorted_ids(values) -> str:
    payload = "\n".join(str(int(x)) for x in sorted(set(values))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def assert_preoutcome_columns(df: pd.DataFrame, label: str) -> None:
    bad = [
        c for c in df.columns
        if any(tok in str(c).lower() for tok in PROHIBITED_PREOUTCOME_TOKENS)
    ]
    if bad:
        raise AssertionError(f"{label}: prohibited pre-outcome columns: {bad}")


def import_source_module(path: Path):
    spec = importlib.util.spec_from_file_location("harmonized_model_source_for_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_check(rows, name: str, passed: bool, detail: str) -> None:
    rows.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    if not passed:
        raise AssertionError(f"{name}: {detail}")


def public_source_audit(project: Path) -> None:
    """Check only the distributed source structure; do not import/run scientific code."""
    source = require_file(project / "scripts" / "modeling" / "harmonized" / "fit_harmonized_model.py")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    assignments = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    assignments[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError, SyntaxError, RecursionError):
                    pass

    predictors = assignments.get("PRIMARY_PREDICTORS")
    targets = assignments.get("FCE_TARGETS")
    required_functions = {"mice_pmm_imputations", "fit_logistic_mi", "rubin_pool", "main"}
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    checks = {
        "harmonized source is valid Python": True,  # ast.parse completed
        "19 unique model terms": isinstance(predictors, list)
        and len(predictors) == len(set(predictors)) == 19,
        "14 unique continuous imputation targets": isinstance(targets, list)
        and len(targets) == len(set(targets)) == 14,
        "imputation targets are model terms": isinstance(predictors, list)
        and isinstance(targets, list) and set(targets).issubset(predictors),
        "30 completed datasets": assignments.get("N_IMPUTATIONS") == 30,
        "10 imputation cycles": assignments.get("MICE_CYCLES") == 10,
        "5 PMM donors": assignments.get("PMM_DONORS") == 5,
        "mortality endpoint variable": assignments.get("OUTCOME") == "hospital_expire_flag",
        "core fitting functions present": required_functions.issubset(functions),
    }
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {label}")
    if not all(checks.values()):
        raise SystemExit("Public harmonized source consistency: FAILED")
    print(f"SHA-256 (published source, informational): {sha256_file(source)}")
    print("PUBLIC SOURCE CHECK: PASS (structural checks only)")
    print("Scope: source structure only; numerical and artifact verification is a separate procedure")
    print("The source check requires no clinical or study-derived data.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Public source check by default; restricted reproducibility assessment on explicit request."
    )
    ap.add_argument(
        "--project-root", default=None,
        help="Repository root (default: inferred from script location).",
    )
    ap.add_argument(
        "--mode", choices=("public", "restricted"), default="public",
        help="Source structure check or restricted artifact verification; default: public.",
    )
    ap.add_argument("--mimic-root", default=None, help="Authorized raw MIMIC path (restricted mode only).")
    ap.add_argument("--manifest-path", type=Path, default=None,
                    help="Path to the authorized harmonized model manifest (restricted mode only).")
    ap.add_argument("--reference-source", default=None,
                    help="Path to the authorized reference fit script (restricted mode only); "
                         "required for comparison against the authorized reference.")
    args = ap.parse_args()

    project = (Path(args.project_root).resolve() if args.project_root
               else Path(__file__).resolve().parents[3])
    if args.mode == "public":
        public_source_audit(project)
        return

    # Restricted assessment requires the authorized source and manifest.
    # Both must pass their original SHA-256 checks before any protected results
    # are read. These inputs are not distributed in the repository.
    source_script = require_file(Path(args.reference_source).resolve() if args.reference_source
                                else project / "scripts" / "modeling" / "harmonized" / "fit_harmonized_model.py")
    harmonized_model_dir = project / "outputs" / "08_external_transport" / "mimic_cr_rrt_harmonized_refit"
    harmonized_model_qc = project / "quality_control" / "eicu_external_validation" / "mimic_cr_rrt_harmonized_refit"
    try:
        source_manifest_path = project / (args.manifest_path or harmonized_model_qc / "HARMONIZED_MODEL_MANIFEST.json")
        require_file(source_manifest_path)
        if sha256_file(source_manifest_path) != EXPECTED_HARMONIZED_MODEL_MANIFEST_SHA256:
            raise ValueError("Manifest failed SHA-256 verification")
    except (FileNotFoundError, ValueError) as exc:
        ap.error(f"Authorized harmonized model manifest not found or invalid: {exc}")

    # Import data/scientific dependencies only for the restricted reproduction.
    global np, pd, scipy, sm, duckdb, pyarrow
    import numpy as np
    import pandas as pd
    import scipy
    import statsmodels.api as sm
    import duckdb
    import pyarrow

    observed_sha = sha256_file(source_script)
    if observed_sha != EXPECTED_SOURCE_SCRIPT_SHA256:
        ap.error(
            "Authorized reference source SHA-256 mismatch: "
            f"expected {EXPECTED_SOURCE_SCRIPT_SHA256}, observed {observed_sha}. "
            "Provide the matching authorized reference source using --reference-source."
        )
    review_dir = harmonized_model_qc / "integrity_review"
    if review_dir.exists() and any(review_dir.iterdir()):
        raise FileExistsError(
            f"{review_dir} already exists and is non-empty. "
            "Integrity review outputs are immutable; review before replacing."
        )
    review_dir.mkdir(parents=True, exist_ok=True)

    if args.mimic_root:
        mimic_root = Path(args.mimic_root).resolve()
    else:
        import sys
        if str(project) not in sys.path:
            sys.path.insert(0, str(project))
        from helpers._runtime import RAW_DATA_ROOT
        mimic_root = Path(RAW_DATA_ROOT).resolve()

    print("=" * 88)
    print("HARMONIZED MODEL — INTEGRITY ASSESSMENT")
    print("=" * 88)
    print("Project root:", project)
    print("MIMIC root  :", mimic_root)
    print("eICU        : NOT ACCESSED")

    checks = []

    source_script_sha = sha256_file(source_script)
    add_check(
        checks, "source_script_sha256",
        source_script_sha == EXPECTED_SOURCE_SCRIPT_SHA256,
        f"observed={source_script_sha}",
    )

    source_manifest_sha = sha256_file(source_manifest_path)
    add_check(
        checks, "source_harmonized_model_manifest_sha256",
        source_manifest_sha == EXPECTED_HARMONIZED_MODEL_MANIFEST_SHA256,
        f"observed={source_manifest_sha}",
    )

    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    add_check(checks, "source_manifest_status", manifest.get("status") == "PASS", str(manifest.get("status")))
    gate_field, gate_value = get_authorization_flag(manifest)
    add_check(checks, "source_manifest_gate_decision", gate_value is False,
              f"{gate_field}={gate_value}")
    preoutcome = get_manifest_record(manifest)
    add_check(checks, "source_manifest_no_eicu_data", manifest.get("eicu_data_read") is False, f"eicu_data_read={manifest.get('eicu_data_read')}")
    add_check(checks, "source_manifest_no_eicu_outcome", manifest.get("eicu_outcome_accessed") is False, f"eicu_outcome_accessed={manifest.get('eicu_outcome_accessed')}")
    add_check(checks, "source_manifest_outcome_not_authorized", manifest.get("outcome_attachment_authorized") is False, f"outcome_attachment_authorized={manifest.get('outcome_attachment_authorized')}")

    # Verify every file governed by the source Harmonized model execution manifest.
    for rel, expected in manifest["generated_output_hashes"].items():
        p = require_file(project / Path(rel))
        got = sha256_file(p)
        add_check(checks, f"generated_hash::{rel}", got == expected, f"observed={got}")

    for rel, expected in manifest["canonical_input_artifact_hashes"].items():
        p = require_file(project / Path(rel))
        got = sha256_file(p)
        add_check(checks, f"canonical_input_hash::{rel}", got == expected, f"observed={got}")

    for rel, expected in manifest["governing_document_hashes"].items():
        p = require_file(project / Path(rel))
        got = sha256_file(p)
        add_check(checks, f"governing_doc_hash::{rel}", got == expected, f"observed={got}")

    g8 = import_source_module(source_script)
    predictors = list(g8.PRIMARY_PREDICTORS)
    fce_targets = list(g8.FCE_TARGETS)

    p_cohort = require_file(project / preoutcome["cohort_path"])
    p_matrix = require_file(project / preoutcome["predictor_matrix_path"])
    p_dev = require_file(harmonized_model_dir / "harmonized_cr_rrt_development_matrix_with_outcome.parquet")

    cohort = pd.read_parquet(p_cohort)
    matrix = pd.read_parquet(p_matrix)
    dev = pd.read_parquet(p_dev)

    assert_preoutcome_columns(cohort, "harmonized cohort")
    assert_preoutcome_columns(matrix, "harmonized predictor matrix")

    add_check(checks, "cohort_n", len(cohort) == EXPECTED_N, f"n={len(cohort)}")
    add_check(checks, "cohort_unique_subject", cohort["subject_id"].notna().all() and not cohort["subject_id"].duplicated().any(), "subject_id unique/nonmissing")
    add_check(checks, "cohort_unique_stay", cohort["stay_id"].notna().all() and not cohort["stay_id"].duplicated().any(), "stay_id unique/nonmissing")
    add_check(checks, "matrix_n", len(matrix) == EXPECTED_N, f"n={len(matrix)}")
    add_check(checks, "matrix_unique_subject", matrix["subject_id"].notna().all() and not matrix["subject_id"].duplicated().any(), "subject_id unique/nonmissing")
    add_check(checks, "matrix_unique_stay", matrix["stay_id"].notna().all() and not matrix["stay_id"].duplicated().any(), "stay_id unique/nonmissing")

    expected_matrix_cols = ["subject_id", "stay_id"] + predictors
    add_check(checks, "matrix_schema", matrix.columns.tolist() == expected_matrix_cols, f"columns={matrix.columns.tolist()}")
    expected_dev_cols = ["subject_id", "stay_id", OUTCOME] + predictors
    add_check(checks, "development_schema", dev.columns.tolist() == expected_dev_cols, f"columns={dev.columns.tolist()}")

    add_check(checks, "cohort_matrix_subject_order", cohort["subject_id"].reset_index(drop=True).equals(matrix["subject_id"].reset_index(drop=True)), "ordered subject_id equality")
    add_check(checks, "cohort_matrix_stay_order", cohort["stay_id"].reset_index(drop=True).equals(matrix["stay_id"].reset_index(drop=True)), "ordered stay_id equality")
    add_check(checks, "matrix_dev_subject_order", matrix["subject_id"].reset_index(drop=True).equals(dev["subject_id"].reset_index(drop=True)), "ordered subject_id equality")
    add_check(checks, "matrix_dev_stay_order", matrix["stay_id"].reset_index(drop=True).equals(dev["stay_id"].reset_index(drop=True)), "ordered stay_id equality")

    predictor_equal = True
    predictor_maxdiff = 0.0
    for c in predictors:
        a = pd.to_numeric(matrix[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(dev[c], errors="coerce").to_numpy(float)
        ok = np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True)
        if not ok.all():
            predictor_equal = False
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            predictor_maxdiff = max(predictor_maxdiff, float(np.max(np.abs(a[finite] - b[finite]))))
    add_check(checks, "outcome_attachment_predictors_unchanged", predictor_equal, f"max_abs_diff={predictor_maxdiff}")

    y = pd.to_numeric(dev[OUTCOME], errors="raise").astype(int)
    add_check(checks, "development_outcome_binary", set(y.unique()).issubset({0, 1}), f"values={sorted(y.unique().tolist())}")
    add_check(checks, "development_events", int(y.sum()) == EXPECTED_EVENTS, f"events={int(y.sum())}")

    subject_set_sha = sha256_sorted_ids(cohort["subject_id"].astype("int64"))
    stay_set_sha = sha256_sorted_ids(cohort["stay_id"].astype("int64"))
    add_check(
        checks, "subject_set_sha",
        subject_set_sha == preoutcome["subject_id_set_sha256_sorted_newline"],
        f"observed={subject_set_sha}",
    )
    add_check(
        checks, "stay_set_sha",
        stay_set_sha == preoutcome["stay_id_set_sha256_sorted_newline"],
        f"observed={stay_set_sha}",
    )

    stage = pd.to_numeric(cohort["cr_rrt_stage"], errors="raise").astype(int)
    stage_counts = stage.value_counts().to_dict()
    add_check(checks, "stage_counts", all(int(stage_counts.get(k, 0)) == v for k, v in EXPECTED_STAGE_COUNTS.items()), f"observed={stage_counts}")
    add_check(checks, "stage2_indicator", np.array_equal(matrix["kdigo_stage_2_indicator"].to_numpy(int), stage.eq(2).astype(int).to_numpy()), "indicator matches cr_rrt_stage==2")
    add_check(checks, "stage3_indicator", np.array_equal(matrix["kdigo_stage_3_indicator"].to_numpy(int), stage.eq(3).astype(int).to_numpy()), "indicator matches cr_rrt_stage==3")

    # Reproduce deterministic multiple imputation from the saved development matrix.
    print("Reproducing deterministic MI30 from saved harmonized development matrix...")
    imps = g8.mice_pmm_imputations(
        dev,
        predictors,
        OUTCOME,
        g8.N_IMPUTATIONS,
        g8.MICE_CYCLES,
        g8.PMM_DONORS,
        g8.MODEL_SEED,
        verbose=True,
    )
    fits, _, reproduction_coef = g8.fit_logistic_mi(imps, predictors, OUTCOME)

    saved_coef = pd.read_csv(harmonized_model_dir / "harmonized_cr_rrt_coefficients_by_imputation.csv")
    cc = reproduction_coef.merge(saved_coef, on=["imputation", "term"], suffixes=("_reproduction", "_saved"), validate="1:1")
    cc["abs_diff"] = (cc["coefficient_reproduction"] - cc["coefficient_saved"]).abs()
    coef_reproduction_maxdiff = float(cc["abs_diff"].max())
    add_check(checks, "harmonized_coefficient_reproduction", coef_reproduction_maxdiff <= 1e-10, f"max_abs_diff={coef_reproduction_maxdiff:.3e}")

    numerical_rows = []
    for imp_no, (dat, fit) in enumerate(zip(imps, fits), start=1):
        Xdf = sm.add_constant(dat[predictors].astype(float), has_constant="add")
        X = Xdf.to_numpy(float)
        p = np.asarray(fit.predict(Xdf), float)
        w = p * (1.0 - p)
        info = X.T @ (w[:, None] * X)
        design_rank = int(np.linalg.matrix_rank(X))
        info_rank = int(np.linalg.matrix_rank(info))
        design_cond = float(np.linalg.cond(X))
        info_cond = float(np.linalg.cond(info))
        params = fit.params.to_numpy(float)
        row = {
            "component": "harmonized_logistic",
            "target_predictor": "",
            "imputation": imp_no,
            "n": int(X.shape[0]),
            "p_including_intercept": int(X.shape[1]),
            "design_rank": design_rank,
            "design_condition_number": design_cond,
            "information_rank": info_rank,
            "information_condition_number": info_cond,
            "converged": bool(getattr(fit, "converged", True)),
            "nonfinite_coefficient_n": int((~np.isfinite(params)).sum()),
            "max_abs_coefficient": float(np.max(np.abs(params))),
            "coefficient_reproduction_max_abs_diff": float(
                cc.loc[cc["imputation"].eq(imp_no), "abs_diff"].max()
            ),
        }
        numerical_rows.append(row)
        if design_rank != X.shape[1]:
            raise AssertionError(f"Logistic design rank deficient at imputation {imp_no}: rank={design_rank}, p={X.shape[1]}")
        if info_rank != X.shape[1]:
            raise AssertionError(f"Logistic information matrix rank deficient at imputation {imp_no}: rank={info_rank}, p={X.shape[1]}")
        if not np.isfinite(design_cond) or not np.isfinite(info_cond):
            raise AssertionError(f"Non-finite condition number at logistic imputation {imp_no}")
        if not getattr(fit, "converged", True) or not np.isfinite(params).all():
            raise AssertionError(f"Unstable/non-converged logistic fit at imputation {imp_no}")

    saved_fce = pd.read_csv(harmonized_model_dir / "fce_conditional_coefficients_by_imputation.csv")
    max_fce_repro_diff = 0.0
    for target_name in fce_targets:
        covariates = [c for c in predictors if c != target_name] + [OUTCOME]
        for imp_no, dat in enumerate(imps, start=1):
            Xdf = sm.add_constant(dat[covariates].astype(float), has_constant="add")
            X = Xdf.to_numpy(float)
            yy = dat[target_name].astype(float).to_numpy()
            rank = int(np.linalg.matrix_rank(X))
            cond = float(np.linalg.cond(X))
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            if rank != X.shape[1]:
                raise AssertionError(f"FCE rank deficient: target={target_name}, imputation={imp_no}, rank={rank}, p={X.shape[1]}")
            if not np.isfinite(cond) or not np.isfinite(beta).all():
                raise AssertionError(f"FCE numerical non-finite: target={target_name}, imputation={imp_no}")

            saved = saved_fce.loc[
                saved_fce["target_predictor"].eq(target_name)
                & saved_fce["imputation"].eq(imp_no),
                ["term", "coefficient"],
            ].set_index("term")["coefficient"]
            terms = list(Xdf.columns)
            diffs = [abs(float(b) - float(saved.loc[t])) for t, b in zip(terms, beta)]
            this_diff = float(max(diffs)) if diffs else 0.0
            max_fce_repro_diff = max(max_fce_repro_diff, this_diff)
            numerical_rows.append({
                "component": "fce_conditional_ols",
                "target_predictor": target_name,
                "imputation": imp_no,
                "n": int(X.shape[0]),
                "p_including_intercept": int(X.shape[1]),
                "design_rank": rank,
                "design_condition_number": cond,
                "information_rank": "",
                "information_condition_number": "",
                "converged": True,
                "nonfinite_coefficient_n": int((~np.isfinite(beta)).sum()),
                "max_abs_coefficient": float(np.max(np.abs(beta))),
                "coefficient_reproduction_max_abs_diff": this_diff,
            })

    add_check(checks, "fce_coefficient_reproduction", max_fce_repro_diff <= 1e-10, f"max_abs_diff={max_fce_repro_diff:.3e}")

    # Verify FCE initialization values are exactly the mean of the 30 completed-data means.
    init_saved = pd.read_csv(harmonized_model_dir / "fce_initialization_values.csv").set_index("target_predictor")
    max_init_diff = 0.0
    for target_name in fce_targets:
        mean_of_means = float(np.mean([pd.to_numeric(dat[target_name], errors="raise").mean() for dat in imps]))
        d = abs(mean_of_means - float(init_saved.loc[target_name, "initialization_value"]))
        max_init_diff = max(max_init_diff, d)
    add_check(checks, "fce_initialization_reproduction", max_init_diff <= 1e-12, f"max_abs_diff={max_init_diff:.3e}")

    numerical = pd.DataFrame(numerical_rows)
    provenance_path = review_dir / "HARMONIZED_MODEL_INTEGRITY_REVIEW_PROVENANCE.json"

    # Bind the directly accessed raw MIMIC files into supplemental provenance.
    raw_paths = {
        "hosp/patients.csv.gz": require_file(mimic_root / "hosp" / "patients.csv.gz"),
        "icu/chartevents.csv.gz": require_file(mimic_root / "icu" / "chartevents.csv.gz"),
        "hosp/labevents.csv.gz": require_file(mimic_root / "hosp" / "labevents.csv.gz"),
        "icu/inputevents.csv.gz": require_file(mimic_root / "icu" / "inputevents.csv.gz"),
        "hosp/admissions.csv.gz": require_file(mimic_root / "hosp" / "admissions.csv.gz"),
    }
    raw_hashes = {}
    print("Hashing the five raw MIMIC files directly used by Harmonized model...")
    for rel, p in raw_paths.items():
        print("  SHA256", rel)
        raw_hashes[rel] = {"sha256": sha256_file(p), "size_bytes": int(p.stat().st_size)}

    logistic = numerical.loc[numerical["component"].eq("harmonized_logistic")].copy()
    fce_num = numerical.loc[numerical["component"].eq("fce_conditional_ols")].copy()

    provenance = {
        "status": "PASS",
        "purpose": "Harmonized model integrity assessment",
        "source_harmonized_model_manifest_path": (
            str(source_manifest_path.relative_to(project)).replace("\\", "/")
            if source_manifest_path.is_relative_to(project)
            else source_manifest_path.name
        ),
        "source_harmonized_model_manifest_sha256": source_manifest_sha,
        "source_harmonized_model_script_path": (str(source_script.relative_to(project)).replace("\\", "/")
                                     if source_script.is_relative_to(project)
                                     else "external_reference_source"),
        "source_harmonized_model_script_sha256": source_script_sha,
        "source_harmonized_model_generated_artifact_hashes_verified": True,
        "source_harmonized_model_canonical_input_hashes_verified": True,
        "source_harmonized_model_governing_document_hashes_verified": True,
        "direct_raw_mimic_input_hashes": raw_hashes,
        "semantic_checks_passed": int(len(checks)),
        "harmonized_logistic": {
            "imputations_checked": int(len(logistic)),
            "all_designs_full_rank": bool((logistic["design_rank"] == logistic["p_including_intercept"]).all()),
            "all_information_matrices_full_rank": bool((logistic["information_rank"] == logistic["p_including_intercept"]).all()),
            "all_converged": bool(logistic["converged"].astype(bool).all()),
            "coefficient_reproduction_max_abs_diff": coef_reproduction_maxdiff,
            "design_condition_number_min": float(logistic["design_condition_number"].min()),
            "design_condition_number_max": float(logistic["design_condition_number"].max()),
            "information_condition_number_min": float(logistic["information_condition_number"].min()),
            "information_condition_number_max": float(logistic["information_condition_number"].max()),
        },
        "fce": {
            "conditional_designs_checked": int(len(fce_num)),
            "all_designs_full_rank": bool((fce_num["design_rank"] == fce_num["p_including_intercept"]).all()),
            "coefficient_reproduction_max_abs_diff": max_fce_repro_diff,
            "initialization_reproduction_max_abs_diff": max_init_diff,
            "design_condition_number_min": float(fce_num["design_condition_number"].min()),
            "design_condition_number_max": float(fce_num["design_condition_number"].max()),
        },
        "semantic_checks": checks,
        "numerical_checks": numerical_rows,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": sm.__version__,
            "duckdb": duckdb.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "eicu_data_read": False,
        "eicu_outcome_accessed": False,
        "harmonized_model_fixed_by_this_audit": False,
        "outcome_attachment_authorized": False,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print()
    print("=" * 88)
    print("HARMONIZED MODEL INTEGRITY ASSESSMENT: PASS")
    print("=" * 88)
    print("Semantic checks                         :", len(checks), "PASS")
    print("Logistic imputations full-rank          :", len(logistic), "/", g8.N_IMPUTATIONS)
    print("Logistic coefficient reproduction max abs diff:", f"{coef_reproduction_maxdiff:.3e}")
    print("FCE conditional designs full-rank       :", len(fce_num), "/", len(fce_targets) * g8.N_IMPUTATIONS)
    print("FCE coefficient reproduction max abs diff      :", f"{max_fce_repro_diff:.3e}")
    print("FCE initialization reproduction max abs diff   :", f"{max_init_diff:.3e}")
    print("Review directory                         :", review_dir)
    print("eICU data/outcome accessed               : NO")
    print("Model authorization                        : separate approval required")
    print("Outcome attachment authorized            : NO")


if __name__ == "__main__":
    main()
