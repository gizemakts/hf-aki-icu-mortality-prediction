"""Portable runtime support shared by the analysis programs."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_raw_data_root(value: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a usable MIMIC root, falling back from stale overrides."""
    project_default = (PROJECT_ROOT / "data" / "raw" / "mimic-iv-3.1").resolve()
    configured = value if value is not None else os.environ.get("MIMIC_ROOT")
    if configured:
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            configured_path = PROJECT_ROOT / configured_path
        configured_path = configured_path.resolve()
        if configured_path.is_dir() or not project_default.is_dir():
            return configured_path
    return project_default


RAW_DATA_ROOT = resolve_raw_data_root()
TEMP_ROOT = Path(tempfile.gettempdir()) / "hfaki_pipeline"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)


def display_table(value: Any) -> None:
    """Render a concise text representation in non-interactive command-line runs."""
    to_string = getattr(value, "to_string", None)
    print(to_string() if callable(to_string) else value)


def logical_source_path(
    value: str | os.PathLike[str],
    *,
    external_root: str | os.PathLike[str] | None = None,
    external_label: str = "external",
) -> str:
    """Return a stable provenance identifier without embedding a workstation path."""
    path = Path(value).expanduser().resolve()
    roots = (
        (PROJECT_ROOT.resolve(), ""),
        (RAW_DATA_ROOT.resolve(), "data/raw/mimic-iv-3.1"),
    )
    for root, label in roots:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        return f"{label}/{relative}".lstrip("/")
    if external_root is not None:
        root = Path(external_root).expanduser().resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            pass
        else:
            return f"{external_label.rstrip('/')}://{relative}"
    return f"{external_label.rstrip('/')}://{path.name}"
