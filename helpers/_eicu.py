"""Shared recursive eICU table discovery contract."""
from __future__ import annotations

from pathlib import Path


def discover_eicu_tables(root: Path, stems: list[str]) -> dict[str, Path | None]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    discovered: dict[str, Path | None] = {}
    for stem in stems:
        wanted = {f"{stem}.csv.gz".lower(), f"{stem}.csv".lower()}
        matches = sorted(
            (path.resolve() for path in root.rglob("*") if path.is_file() and path.name.lower() in wanted),
            key=lambda path: path.as_posix().lower(),
        )
        if len(matches) > 1:
            raise RuntimeError(f"Multiple {stem} tables under {root}:\n" + "\n".join(map(str, matches)))
        discovered[stem] = matches[0] if matches else None
    return discovered
