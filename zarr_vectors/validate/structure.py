"""Level 1 structural validation — verify the store layout on disk."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    """Accumulated validation outcome."""

    level: int
    passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def add_pass(self, msg: str) -> None:
        self.passed.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def merge(self, other: "ValidationResult") -> None:
        self.passed.extend(other.passed)
        self.warnings.extend(other.warnings)
        self.errors.extend(other.errors)

    def summary(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        parts = [
            f"Level {self.level} validation: {status}",
            f"  {len(self.passed)} passed, {len(self.warnings)} warnings, {len(self.errors)} errors",
        ]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN:  {w}")
        return "\n".join(parts)


def validate_structure(store_path: str | Path) -> ValidationResult:
    """Level 1: verify store directory layout."""
    result = ValidationResult(level=1)
    root = Path(store_path)

    if not root.exists():
        result.add_error(f"Store path does not exist: {root}")
        return result
    if not root.is_dir():
        result.add_error(f"Store path is not a directory: {root}")
        return result
    result.add_pass("Store root exists and is a directory")

    has_meta = any((root / f).exists() for f in [".zattrs", "zarr.json", "metadata.json"])
    if has_meta:
        result.add_pass("Root metadata file found")
    else:
        result.add_error("No root metadata found (expected .zattrs, zarr.json, or metadata.json)")

    level_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name.startswith("resolution_"))
    if not level_dirs:
        result.add_error("No resolution level directories found")
        return result
    result.add_pass(f"Found {len(level_dirs)} resolution level(s)")

    for level_dir in level_dirs:
        ln = level_dir.name
        vd = level_dir / "vertices"
        if vd.exists() and vd.is_dir():
            result.add_pass(f"{ln}/vertices/ exists")
        else:
            result.add_error(f"{ln}/vertices/ missing")

        vgo = level_dir / "vertex_group_offsets"
        if vgo.exists() and vgo.is_dir():
            result.add_pass(f"{ln}/vertex_group_offsets/ exists")
        else:
            result.add_warning(f"{ln}/vertex_group_offsets/ missing")

        if any((level_dir / f).exists() for f in [".zattrs", "zarr.json"]):
            result.add_pass(f"{ln}/ has metadata")
        else:
            result.add_warning(f"{ln}/ has no metadata file")

        for opt in ["links", "attributes", "object_index", "object_attributes",
                     "groupings", "cross_chunk_links"]:
            if (level_dir / opt).exists():
                result.add_pass(f"{ln}/{opt}/ exists")

    if (root / "parametric").exists():
        result.add_pass("parametric/ group exists")

    return result
