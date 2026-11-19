"""Deterministic logical fingerprints for resumable scientific computations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def logical_frame_sha256(frame: pd.DataFrame, *, sort_by: list[str]) -> str:
    """Hash logical table content without paths or storage-format byte details."""
    ordered = frame.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(_canonical_json({
        "columns": [str(column) for column in ordered.columns],
        "dtypes": [str(dtype) for dtype in ordered.dtypes],
        "rows": int(len(ordered)),
        "sort_by": sort_by,
    }))
    row_hashes = pd.util.hash_pandas_object(ordered, index=False, categorize=False)
    digest.update(row_hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def analysis_fingerprint(contract: dict[str, Any], frames: dict[str, tuple[pd.DataFrame, list[str]]]) -> tuple[str, dict[str, Any]]:
    """Return a SHA-256 and transparent contract for a resumable analysis."""
    payload = {
        "contract": contract,
        "logical_inputs": {
            name: logical_frame_sha256(frame, sort_by=sort_by)
            for name, (frame, sort_by) in sorted(frames.items())
        },
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest(), payload


def fingerprinted_resume_dir(base: Path, fingerprint: str, payload: dict[str, Any]) -> Path:
    """Create an input-bound namespace and persist its non-path contract."""
    target = base / fingerprint
    target.mkdir(parents=True, exist_ok=True)
    contract_path = target / "resume_fingerprint_contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if _canonical_json(existing) != _canonical_json(payload):
            raise RuntimeError(f"Resume fingerprint collision or corrupt contract: {contract_path}")
    else:
        contract_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target
