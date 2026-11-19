#!/usr/bin/env python3
"""Validate isolated-ultrafiltration sensitivity outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


def main() -> None:
    out = _REPOSITORY_ROOT / "outputs" / "06_sensitivity" / "isolated_uf"
    audit_path = out / "rrt_source_isolation_audit.csv"
    affected_path = out / "affected_stays.csv"
    changes_path = out / "kdigo_indicator_changes.csv"
    summary_path = out / "internal_validation_summary.csv"
    manifest_path = out / "isolated_uf_sensitivity_manifest.json"

    for path in [audit_path, affected_path, changes_path, summary_path, manifest_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    audit = pd.read_csv(audit_path)
    affected = pd.read_csv(affected_path)
    changes = pd.read_csv(changes_path)
    summary = pd.read_csv(summary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if len(audit) != 3457 or audit["stay_id"].nunique() != 3457:
        raise AssertionError("RRT source-isolation audit must contain exactly 3,457 unique stays.")
    if int(pd.to_numeric(audit["has_226457"], errors="raise").sum()) != 73:
        raise AssertionError("The primary cohort must contain 73 stays with itemid 226457 by the landmark.")
    isolated_mask = pd.to_numeric(audit["isolated_226457_output_only"], errors="raise").eq(1)
    if not 0 < int(isolated_mask.sum()) < 11:
        raise AssertionError("Isolated-UF cell must be nonzero and smaller than 11.")

    isolated_ids = set(pd.to_numeric(audit.loc[isolated_mask, "stay_id"], errors="raise").astype("int64"))
    affected_ids = set(pd.to_numeric(affected["stay_id"], errors="raise").astype("int64"))
    change_ids = set(pd.to_numeric(changes["stay_id"], errors="raise").astype("int64"))
    if isolated_ids != affected_ids or affected_ids != change_ids:
        raise AssertionError("Isolated-UF stay identities disagree across source audit, affected-stay, and KDIGO-change outputs.")
    if len(affected) != int(isolated_mask.sum()) or int(pd.to_numeric(affected["stage_changed"], errors="raise").sum()) != len(affected):
        raise AssertionError("Source-verified group and KDIGO stage changes are inconsistent.")
    if len(summary) != 5 or set(summary["metric"].astype(str)) != {
        "auroc", "brier", "calibration_intercept", "calibration_slope", "observed_expected_ratio"
    }:
        raise AssertionError("Isolated-UF internal-validation summary is incomplete.")

    expected_manifest = {
        "status": "PASS",
        "source_verified_affected_stays": "<11",
        "selected_kdigo_stage_changes": "<11",
        "membership_changed": False,
        "timezero_changed": False,
        "primary_n": 3457,
        "primary_events": 707,
        "bootstrap_resamples": 500,
    }
    for key, expected in expected_manifest.items():
        observed=manifest.get(key)
        # Historical restricted manifests may contain precise counts, while
        # publication-facing manifests use the display suppression token.
        if key in ("source_verified_affected_stays", "selected_kdigo_stage_changes"):
            if observed != "<11" and observed != len(affected):
                raise AssertionError(f"Isolated-UF source count mismatch for {key}")
        elif observed != expected:
            raise AssertionError(f"Isolated-UF manifest mismatch for {key}")

    print("ISOLATED-UF RESULT VALIDATION: PASS")


if __name__ == "__main__":
    main()
