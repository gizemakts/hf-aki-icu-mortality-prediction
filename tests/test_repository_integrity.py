"""Repository layout, source integrity, and metadata checks."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE = REPO / "scripts"


class RepositoryIntegrityTests(unittest.TestCase):
    def test_source_tree_and_metadata(self):
        for name in ("README.md", "LICENSE", "CITATION.cff", "requirements.txt", ".gitignore"):
            with self.subTest(name=name):
                self.assertTrue((REPO / name).is_file())
        for name in ("cohort", "preprocessing", "modeling", "evaluation", "sensitivity"):
            with self.subTest(name=name):
                self.assertTrue((CODE / name).is_dir())
        self.assertTrue((REPO / "helpers" / "_runtime.py").is_file())

    def test_all_python_files_parse(self):
        files = sorted(REPO.rglob("*.py"))
        self.assertGreaterEqual(len(files), 70)
        for file in files:
            with self.subTest(file=file.relative_to(REPO).as_posix()):
                ast.parse(file.read_text(encoding="utf-8"), filename=str(file))

    def test_root_discovery_uses_repository_not_scripts(self):
        representative = [
            "cohort/mimic_iv/build_primary_cohort.py",
            "preprocessing/aki/combined_kdigo.py",
            "modeling/mimic_iv/fit_canonical_models.py",
            "evaluation/decision_curve/decision_curve_analysis.py",
            "sensitivity/mimic_iv/expanded_heart_failure.py",
        ]
        for path in representative:
            with self.subTest(path=path):
                source = (CODE / path).read_text(encoding="utf-8")
                self.assertIn('Path(__file__).resolve().parents[3]', source)
                self.assertNotIn('Path(__file__).resolve().parents[2]', source)


    def test_source_hash_references(self):
        checks = [
            ("scripts/evaluation/mimic_iv/canonical_internal_validation.py",
             "scripts/evaluation/mimic_iv/bootstrap_internal_validation.py"),
            ("scripts/sensitivity/posthoc/outcome_free_imputation_sensitivity.py",
             "scripts/cohort/eicu_crd/final_preoutcome_build.py"),
            ("scripts/sensitivity/posthoc/outcome_free_imputation_sensitivity.py",
             "scripts/modeling/harmonized/fit_harmonized_model.py"),
            ("scripts/evaluation/apache_iv/verify_apache_benchmark.py",
             "scripts/evaluation/apache_iv/apache_benchmark.py"),
            ("scripts/evaluation/decision_curve/verify_decision_curve.py",
             "scripts/evaluation/decision_curve/decision_curve_analysis.py"),
        ]
        for verifying, source in checks:
            with self.subTest(verifying=verifying, source=source):
                recorded = (REPO / verifying).read_text(encoding="utf-8")
                observed = hashlib.sha256((REPO / source).read_bytes()).hexdigest()
                self.assertIn(observed, recorded)

    def test_scientific_invariant_contract_is_required(self):
        script = CODE / "evaluation/mimic_iv/scientific_integrity_checks.py"
        help_result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPO, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--invariants", help_result.stdout)

        result = subprocess.run(
            [sys.executable, str(script), "--invariants", "absent-scientific-contract.json"],
            cwd=REPO, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Authorized scientific-invariant contract is required", result.stderr)

    def test_harmonized_manifest_structure(self):
        source = CODE / "modeling/harmonized/audit_harmonized_model_lock.py"
        spec = importlib.util.spec_from_file_location("harmonized_integrity", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        record = {
            "cohort_path": "cohort.parquet", "predictor_matrix_path": "matrix.parquet",
            "subject_id_set_sha256_sorted_newline": "a" * 64,
            "stay_id_set_sha256_sorted_newline": "b" * 64,
        }
        payload = {"status": "PASS", "gate_decision": False, "site_record": record}
        self.assertEqual(module.get_manifest_record(payload), record)
        self.assertEqual(module.get_authorization_flag(payload), ("gate_decision", False))
        with self.assertRaises(ValueError):
            module.get_manifest_record({"first": record, "second": record})
        with self.assertRaises(ValueError):
            module.get_authorization_flag({"gate_first": False, "gate_second": True})

    def test_restricted_mode_requires_authorized_manifest(self):
        result = subprocess.run(
            [sys.executable, str(CODE / "modeling/harmonized/audit_harmonized_model_lock.py"),
             "--mode", "restricted"],
            cwd=REPO, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Authorized harmonized model manifest not found", result.stderr)

    def test_eicu_sensitivity_cli_matches_documented_contract(self):
        script = CODE / "sensitivity/eicu_crd/external_sensitivity_performance.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPO, capture_output=True, text=True, timeout=30, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for option in ("--project-root", "--package-root", "--preflight", "--ratify", "--run"):
            self.assertIn(option, result.stdout)

    def test_verifiers_require_authorized_inputs(self):
        """Status-only invocation cannot report a successful numerical verification."""
        for relative in (
            "evaluation/apache_iv/verify_apache_benchmark.py",
            "evaluation/decision_curve/verify_decision_curve.py",
        ):
            with self.subTest(relative=relative):
                result = subprocess.run(
                    [sys.executable, str(CODE / relative)],
                    cwd=REPO, capture_output=True, text=True,
                    timeout=30, check=False,
                )
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("INPUTS_REQUIRED", result.stdout)

    def test_harmonized_data_free_audit(self):
        result = subprocess.run(
            [sys.executable, str(CODE / "modeling/harmonized/audit_harmonized_model_lock.py")],
            cwd=REPO, capture_output=True, text=True, timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PUBLIC SOURCE CHECK: PASS", result.stdout)
        self.assertIn("structural checks only", result.stdout)


if __name__ == "__main__":
    unittest.main()
