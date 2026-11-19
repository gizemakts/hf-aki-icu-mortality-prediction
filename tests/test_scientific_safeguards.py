"""Data-free tests of numerical calculations and source-level disclosure safeguards.

Only synthetic arrays and synthetic aggregate rows are used. No clinical data or
study-derived patient records are required.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScientificSafeguardsTests(unittest.TestCase):
    def test_preoutcome_input_integrity_guard(self):
        """Outcome attachment must verify the pre-outcome lock and file hash."""
        path = ROOT / "scripts/cohort/eicu_crd/build_outcome_evaluable_attachment.py"
        source = path.read_text(encoding="utf-8")
        ast.parse(source, filename=str(path))
        self.assertIn("lock=Path(args.preoutcome_lock_manifest)", source)
        self.assertIn("verify_hash(verification,EXPECTED_PREOUTCOME_VERIFICATION_SHA", source)

    def test_small_cell_suppression_policy(self):
        """Source-level check of published small-count redaction, not a data test."""
        design = (ROOT / "scripts/preprocessing/predictors/model_design.py").read_text(encoding="utf-8")
        tree = ast.parse(design)
        policies = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "rrt_note"
                    for target in node.targets)
        ]
        self.assertEqual(len(policies), 1)
        policy = ast.literal_eval(policies[0])
        for key in (
            "isolated_226457_uf_output_only_target_stays",
            "selected_kdigo_stage_changes_when_suppressed",
        ):
            self.assertEqual(policy[key], "<11")

        # Percentages in a public source can inadvertently disclose a small
        # suppressed count. Check for over-precise missingness summaries.
        cohort_size = 3457  # Historical design cohort size documented by the source.
        pattern = re.compile(
            r"(?<![\d.])(\d+(?:\.\d+)?)\s*%\s+(?:component\s+)?missing(?:ness)?\b",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(design):
            self.assertGreaterEqual(cohort_size * float(match.group(1)) / 100.0, 11)

        suppressed = (ROOT / "scripts/sensitivity/mimic_iv/isolated_uf_rrt_suppression.py").read_text(encoding="utf-8")
        verified = (ROOT / "scripts/sensitivity/mimic_iv/isolated_uf_result_validation.py").read_text(encoding="utf-8")
        self.assertIn("'source_verified_affected_stays': '<11'", suppressed)
        self.assertIn('"source_verified_affected_stays": "<11"', verified)

    def test_sparse_histogram_bins_on_synthetic_risks(self):
        """Execute the source's actual histogram bin-merging logic on fake risks."""
        import numpy as np

        path = ROOT / "scripts/evaluation/eicu_crd/external_performance.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        run = next(n for n in tree.body
                   if isinstance(n, ast.FunctionDef) and n.name == "primary_run")
        statements = run.body
        start = next(i for i, n in enumerate(statements)
                     if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "initial_edges"
                             for t in n.targets))
        end = next(i for i, n in enumerate(statements[start:], start)
                   if isinstance(n, ast.If) and "Histogram includes a disclosive small cell"
                   in ast.unparse(n))
        executable = ast.fix_missing_locations(ast.Module(
            body=statements[start:end + 1], type_ignores=[]
        ))
        code = compile(executable, str(path), "exec")

        # The original function's code is executed; it is not reimplemented.
        risks = np.array([0.01] * 12 + [0.45] * 3 + [0.98] * 20)
        scope = {"np": np, "p": risks}
        exec(code, scope)
        self.assertEqual(int(sum(scope["safe_counts"])), len(risks))
        self.assertTrue(all(n == 0 or n >= 11 for n in scope["safe_counts"]))
        self.assertEqual(scope["safe_edges"][0], 0.0)
        self.assertEqual(scope["safe_edges"][-1], 1.0)

        # An entire synthetic sample below the suppression threshold is not
        # safe to plot and must be explicitly rejected.
        with self.assertRaisesRegex(AssertionError, "disclosive small cell"):
            exec(code, {"np": np, "p": np.array([0.05] * 10)})

    def test_apache_completion_requires_explicit_verification_metadata(self):
        """A completed aggregate calculation is not a verified study result."""
        module = load_module(
            "scripts/evaluation/apache_iv/apache_benchmark.py",
            "apache_benchmark_manifest_test",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {}
            for name in module.EXPECTED_OUTPUT_NAMES - {"ANALYSIS_MANIFEST.json"}:
                data = f"synthetic output for {name}\n".encode("utf-8")
                (root / name).write_bytes(data)
                expected[name] = hashlib.sha256(data).hexdigest()
            manifest = {
                "status": "COMPUTED",
                "verification_scope": "calculation_only",
                "output_hashes": expected,
            }
            path = root / "ANALYSIS_MANIFEST.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            module.validate_completed_staging(root)
            manifest["status"] = "PASS"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "completion manifest"):
                module.validate_completed_staging(root)
            manifest["status"] = "COMPUTED"
            manifest["verification_scope"] = "independently_verified"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "completion manifest"):
                module.validate_completed_staging(root)

    def test_apache_failed_staging_is_not_published(self):
        """A synthetic writer failure must not publish a result directory."""
        module = load_module(
            "scripts/evaluation/apache_iv/apache_benchmark.py", "apache_atomic_test"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = root / "aggregate_results"

            def reject(staging):
                (staging / "synthetic_marker.txt").write_text("test only", encoding="utf-8")
                raise RuntimeError("synthetic writer failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic writer failure"):
                module.transactional_publish(final, reject)
            self.assertFalse(final.exists())
            candidates = list(root.glob(".aggregate_results.working-*"))
            self.assertEqual(len(candidates), 1)
            marker = candidates[0] / "RUN_REJECTED.json"
            self.assertTrue(marker.is_file())
            record = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "REJECTED_NOT_FOR_SCIENTIFIC_USE")

    def test_dca_import_and_protected_pair_publication(self):
        """DCA imports, preserves immutable result metadata, and protects pair commits."""
        module = load_module(
            "scripts/evaluation/decision_curve/decision_curve_analysis.py", "dca_pair_test"
        )
        result = module.DCAResult([], [], {}, False)
        with self.assertRaises(AttributeError):
            result.release_ready = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "DCA_ANALYSIS_MANIFEST.json"
            pair = "a" * 64
            path.write_text(json.dumps({
                "status": "STAGED_PAIRED_PUBLICATION",
                "publication_pair_sha256": pair,
            }), encoding="utf-8")
            with self.assertRaises(module.DCAError):
                module.finalize_restricted_publication(root, "b" * 64)
            self.assertEqual(json.loads(path.read_text())["status"], "STAGED_PAIRED_PUBLICATION")
            module.finalize_restricted_publication(root, pair)
            self.assertEqual(json.loads(path.read_text())["status"], "RESTRICTED_INTERNAL_AGGREGATES_PUBLISHED")
            self.assertFalse((root / "DCA_ANALYSIS_MANIFEST.atomic.json").exists())

    def test_auroc_ranks_with_ties_and_invalid_single_class(self):
        """Evaluate the real AUROC calculation on small synthetic vectors."""
        import numpy as np

        model = load_module(
            "scripts/evaluation/eicu_crd/external_performance.py",
            "external_performance_test",
        )
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(model.auc_ranks(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)
        self.assertAlmostEqual(model.auc_ranks(y, np.array([0.8, 0.9, 0.1, 0.2])), 0.0)
        self.assertAlmostEqual(model.auc_ranks(y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)
        with self.assertRaisesRegex(RuntimeError, "both outcome classes"):
            model.auc_ranks(np.array([1, 1, 1]), np.array([0.2, 0.3, 0.4]))

    def test_auroc_percentile_interpolation(self):
        """Check exact linear percentile interpolation and input validation."""
        module = load_module("tools/derive_canonical_auroc_intervals.py", "auroc_intervals_test")
        values = [0.9, 0.3, 0.1, 0.7, 0.5]
        self.assertAlmostEqual(module.linear_percentile(values, 2.5), 0.12)
        self.assertAlmostEqual(module.linear_percentile(values, 97.5), 0.88)
        self.assertEqual(module.linear_percentile(values, 0), 0.1)
        self.assertEqual(module.linear_percentile(values, 100), 0.9)
        for bad_values, bad_pct in (([], 50), ([0.2], -1), ([0.2], 101)):
            with self.subTest(values=bad_values, percentage=bad_pct):
                with self.assertRaises(ValueError):
                    module.linear_percentile(bad_values, bad_pct)

    def test_interval_calculation_and_replicate_integrity(self):
        """Use 500 synthetic aggregate replicates to exercise full validation."""
        module = load_module("tools/derive_canonical_auroc_intervals.py", "aggregate_interval_test")
        fields = ["model", "replicate", "bootstrap_apparent_auroc", "optimism_auroc"]
        rows = [
            {"model": "PRIMARY", "replicate": str(i),
             "bootstrap_apparent_auroc": str(0.6 + i / 10000),
             "optimism_auroc": "0.01"}
            for i in range(1, 501)
        ]
        summary = {
            "mean_bootstrap_optimism": "0.01",
            "optimism_corrected_estimate": "0.69",
            "apparent_estimate": "0.70",
            "corrected_ci95": "NA",
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "synthetic_bootstrap.csv"

            def write_records(records):
                with path.open("w", encoding="utf-8", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(records)

            write_records(rows)
            point, low, high, n = module.calculate(path, "PRIMARY", summary)
            apparent = [float(r["bootstrap_apparent_auroc"]) for r in rows]
            self.assertEqual(n, 500)
            self.assertAlmostEqual(point, 0.69)
            self.assertAlmostEqual(low, module.linear_percentile(apparent, 2.5) - 0.01)
            self.assertAlmostEqual(high, module.linear_percentile(apparent, 97.5) - 0.01)
            # Previous restricted summary files retain their original
            # missing-interval encoding; no numerical CI is fabricated.
            reference_summary = {**summary, "corrected_ci95": "NOT_" + "ESTIMATED"}
            self.assertEqual(module.calculate(path, "PRIMARY", reference_summary)[0], point)

            broken = [r.copy() for r in rows]
            broken[-1]["replicate"] = "499"
            write_records(broken)
            with self.assertRaisesRegex(ValueError, "duplicate or missing"):
                module.calculate(path, "PRIMARY", summary)

            write_records(rows)
            with self.assertRaisesRegex(ValueError, "optimism does not match"):
                module.calculate(path, "PRIMARY", {**summary, "mean_bootstrap_optimism": "0.02"})
            with self.assertRaisesRegex(ValueError, "must not contain a separately estimated"):
                module.calculate(path, "PRIMARY", {**summary, "corrected_ci95": "0.1-0.9"})


if __name__ == "__main__":
    unittest.main()
