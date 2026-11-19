#!/usr/bin/env python3
"""Derive approximate canonical AUROC intervals from *authorized* aggregate records.

Reads completed post hoc outcome-safe bootstrap outputs but never writes
input rows or patient-level values. DO NOT publish restricted input files.

Example, from repository root (with local authorized input files):
  python tools/derive_canonical_auroc_intervals.py \
    --summary /authorized/full_500/internal_validation_summary.csv \
    --primary /authorized/full_500/primary_bootstrap_replicates.csv \
    --secondary /authorized/full_500/secondary_bootstrap_replicates.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean


def linear_percentile(values: list[float], percentage: float) -> float:
    """NumPy/pandas default linear (type 7) percentile interpolation."""
    if not values or not 0 <= percentage <= 100:
        raise ValueError("Invalid percentile input")
    data = sorted(values)
    index = (len(data) - 1) * percentage / 100.0
    low, high = math.floor(index), math.ceil(index)
    return data[low] + (index - low) * (data[high] - data[low])


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row.get("metric") == "auroc"]
    if len(rows) != 2 or {r["model"] for r in rows} != {"PRIMARY", "SECONDARY"}:
        raise ValueError("Expected one AUROC row for each canonical model")
    return {r["model"]: r for r in rows}


def calculate(path: Path, model: str, summary: dict[str, str]) -> tuple[float, float, float, int]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 500 or {row["model"] for row in rows} != {model}:
        raise ValueError(f"{model}: expected exactly 500 model-matched bootstrap rows")
    replicate_ids = [int(row["replicate"]) for row in rows]
    if len(set(replicate_ids)) != 500 or set(replicate_ids) != set(range(1, 501)):
        raise ValueError(f"{model}: duplicate or missing replicate identifiers")
    apparent = [float(r["bootstrap_apparent_auroc"]) for r in rows]
    optimism = mean(float(r["optimism_auroc"]) for r in rows)
    expected_optimism = float(summary["mean_bootstrap_optimism"])
    if not math.isclose(optimism, expected_optimism, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{model}: optimism does not match authorized summary")
    corrected = float(summary["optimism_corrected_estimate"])
    if not math.isclose(
        corrected,
        float(summary["apparent_estimate"]) - optimism,
        rel_tol=0,
        abs_tol=1e-10,
    ):
        raise ValueError(f"{model}: corrected point does not match summary")
    raw_ci = str(summary.get("corrected_ci95") or "").strip().upper()
    # A missing value identifies a summary without separately estimated intervals.
    missing_ci = raw_ci in {"", "NA", "N/A"} or (
        raw_ci.startswith("NOT_") and raw_ci[4:] == "ESTIMATED"
    )
    if not missing_ci:
        raise ValueError(f"{model}: summary must not contain a separately estimated corrected interval")
    return (corrected, linear_percentile(apparent, 2.5) - optimism,
            linear_percentile(apparent, 97.5) - optimism, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    args = parser.parse_args()
    summary = read_summary(args.summary)
    print("Supplementary post hoc outcome-safe validation; approximate shifted AUROC intervals")
    for model, path in (("PRIMARY", args.primary), ("SECONDARY", args.secondary)):
        point, low, high, n = calculate(path, model, summary[model])
        print(f"{model}: n={n}; corrected AUROC={point:.7f}; "
              f"approximate 95% interval={low:.6f}–{high:.6f}; "
              f"Table 2: {point:.3f} ({low:.3f}–{high:.3f})")
    print("The source summaries do not contain independently estimated corrected confidence intervals.")


if __name__ == "__main__":
    main()
