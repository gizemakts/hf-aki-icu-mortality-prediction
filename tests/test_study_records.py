"""Study input validation using synthetic files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from helpers._study_records import OUTCOME_MANIFEST, verify_study_file


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StudyRecordTests(unittest.TestCase):
    def test_file_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            payload = b'{"status":"PASS"}'
            digest = hashlib.sha256(payload).hexdigest()
            with self.assertRaises(FileNotFoundError):
                verify_study_file(path, digest)
            path.write_bytes(payload)
            self.assertEqual(verify_study_file(path, digest), path)
            path.write_bytes(b'{"status":"FAIL"}')
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                verify_study_file(path, digest)

    def test_verifiers_use_specified_manifest(self):
        for name, script in (
            ("apache", "apache_iv/verify_apache_benchmark.py"),
            ("decision_curve", "decision_curve/verify_decision_curve.py"),
        ):
            module = load_module(name + "_input_test", f"scripts/evaluation/{script}")
            with self.subTest(verifier=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                selected = root / "study_inputs" / "outcomes.json"
                selected.parent.mkdir()
                payload = b'{"identity_lock":{}}'
                selected.write_bytes(payload)
                digest = hashlib.sha256(payload).hexdigest()
                with patch.object(module, "PATHS", {"outcome_lock": OUTCOME_MANIFEST}), \
                     patch.object(module, "PINNED_HASHES", {"outcome_lock": digest}):
                    with self.assertRaises(module.VerificationFailure):
                        module.authenticate_inputs(root, module.Recorder())
                    supplied = {OUTCOME_MANIFEST: selected}
                    self.assertEqual(
                        module.authenticate_inputs(root, module.Recorder(), supplied),
                        {"outcome_lock": selected},
                    )
                    selected.write_bytes(b'{"identity_lock":null}')
                    with self.assertRaises(module.VerificationFailure):
                        module.authenticate_inputs(root, module.Recorder(), supplied)

    def test_sensitivity_methods_record_bindings(self):
        module = load_module(
            "sensitivity_methods_test", "scripts/sensitivity/eicu_crd/external_sensitivity_performance.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "methods"
            package.mkdir()
            note = package / "EICU_SENSITIVITY_PERFORMANCE_METHODS.md"
            note.write_text("Methods description for this test.", encoding="utf-8")
            pre = {"note_sha": module.sha(note), "script_sha": module.sha(Path(module.__file__))}

            module.record_methods(root, package, pre)
            receipt_path = root / module.NEW_METHOD_DIR / module.NEW_RECEIPT
            self.assertEqual(module.check_ratification(root, pre), module.sha(receipt_path))
            with self.assertRaisesRegex(RuntimeError, "script_sha256"):
                module.check_ratification(root, {**pre, "script_sha": "0" * 64})

            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            del receipt["primary_lock_zip_sha256"]
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "primary_lock_zip_sha256"):
                module.check_ratification(root, pre)

    def test_sensitivity_record_is_selected_by_fields(self):
        module = load_module(
            "sensitivity_records_test", "scripts/sensitivity/eicu_crd/external_sensitivity_performance.py"
        )
        record = {"cohort": "external", "membership_count": 10}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.zip"
            with ZipFile(path, "w") as z:
                z.writestr("data.json", json.dumps(record))
                z.writestr("metadata.json", "{}")

            def matches(candidate):
                return {"cohort", "membership_count"} <= candidate.keys()

            self.assertEqual(module.read_json_record(path, matches), record)
            with ZipFile(path, "a") as z:
                z.writestr("duplicate.json", json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, "Expected one matching study record"):
                module.read_json_record(path, matches)
