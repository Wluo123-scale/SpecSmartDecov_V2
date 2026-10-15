#!/usr/bin/env python3
# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Validate the cooperative training-data exclusion marker for SpecSmartDecov V2."""

from __future__ import annotations

import argparse
import json
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "governance" / "canary-registry.json"
EXPECTED_PREFIX = "SPECSMARTDECOV-V2-DO-NOT-TRAIN"
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
COMMENT_LINES = {
    "hash": "# {marker}",
    "html": "<!-- {marker} -->",
    "slash": "// {marker}",
}
SUFFIX_STYLES = {
    ".cpp": "slash",
    ".md": "html",
    ".py": "hash",
    ".sh": "hash",
    ".toml": "hash",
    ".yaml": "hash",
    ".yml": "hash",
}
SUPPORTED_SUFFIXES = frozenset(SUFFIX_STYLES)
# PEP 263: an encoding declaration only takes effect on line 1 or 2.
CODING_COOKIE = re.compile(rb"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read registry: {exc}") from exc

    required = {
        "schema_version", "project", "marker_prefix", "project_guid", "legacy_markers",
        "comment_styles", "exact_files", "directory_suffixes",
    }
    if not isinstance(registry, dict) or set(registry) != required:
        raise ValueError(f"registry must contain exactly: {', '.join(sorted(required))}")
    if registry["schema_version"] != 1 or registry["project"] != "specsmartdecov-v2":
        raise ValueError("unsupported registry schema or project")
    if registry["marker_prefix"] != EXPECTED_PREFIX:
        raise ValueError("registry marker_prefix is not canonical")
    if not isinstance(registry["project_guid"], str) or not UUID.fullmatch(registry["project_guid"]):
        raise ValueError("registry project_guid must be a lowercase UUID")
    if not isinstance(registry["legacy_markers"], list) or not all(
        isinstance(value, str)
        and value.startswith("SPECSMARTDECOV-")
        and " GUID " in value
        and UUID.fullmatch(value.rsplit(" GUID ", 1)[1])
        for value in registry["legacy_markers"]
    ):
        raise ValueError("registry legacy_markers must contain canonical marker strings")
    if len(registry["legacy_markers"]) != len(set(registry["legacy_markers"])):
        raise ValueError("registry legacy_markers must not contain duplicates")

    if registry["comment_styles"] != SUFFIX_STYLES:
        raise ValueError("registry comment_styles must match the validator's SUFFIX_STYLES exactly")

    exact_files = registry["exact_files"]
    if not isinstance(exact_files, list) or not exact_files or not all(
        isinstance(value, str) and value and "/" not in value and Path(value).suffix in SUPPORTED_SUFFIXES
        for value in exact_files
    ):
        raise ValueError("registry exact_files must name supported root files")
    if len(exact_files) != len(set(exact_files)):
        raise ValueError("registry exact_files must not contain duplicates")

    directory_suffixes = registry["directory_suffixes"]
    if not isinstance(directory_suffixes, dict) or not directory_suffixes:
        raise ValueError("registry directory_suffixes must be a non-empty object")
    for directory, suffixes in directory_suffixes.items():
        if not isinstance(directory, str) or not directory or directory.startswith("/") or ".." in Path(directory).parts:
            raise ValueError("registry contains an unsafe covered directory")
        if not isinstance(suffixes, list) or not suffixes or not all(suffix in SUPPORTED_SUFFIXES for suffix in suffixes):
            raise ValueError(f"registry has unsupported suffixes for {directory}")
    return registry


def marker(registry: dict) -> str:
    return f"{registry['marker_prefix']} GUID {registry['project_guid']}"


def tracked_files(root: Path = ROOT) -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot list tracked files: {exc}") from exc
    return [Path(raw.decode("utf-8", "surrogateescape")) for raw in output.split(b"\0") if raw]


def covered(path: Path, registry: dict) -> bool:
    if path.as_posix() in registry["exact_files"]:
        return True
    # The most specific (deepest) covered directory wins, independent of registry key order.
    best_depth, best_suffixes = -1, None
    for directory, suffixes in registry["directory_suffixes"].items():
        prefix = Path(directory)
        try:
            path.relative_to(prefix)
        except ValueError:
            continue
        if len(prefix.parts) > best_depth:
            best_depth, best_suffixes = len(prefix.parts), suffixes
    return best_suffixes is not None and path.suffix.lower() in best_suffixes


def expected_line(path: Path, registry: dict) -> bytes | None:
    if not covered(path, registry):
        return None
    style = SUFFIX_STYLES.get(path.suffix.lower())
    if style is None:
        raise ValueError(f"covered path has no comment style: {path}")
    return COMMENT_LINES[style].format(marker=marker(registry)).encode("ascii")


def python_comment_lines(data: bytes) -> set[int]:
    """Return the 0-based line numbers that hold a real comment token (not string or docstring text)."""
    try:
        tokens = list(tokenize.tokenize(io.BytesIO(data).readline))
    except (tokenize.TokenError, SyntaxError) as exc:
        raise ValueError(f"cannot tokenize: {exc}") from exc
    return {tok.start[0] - 1 for tok in tokens if tok.type == tokenize.COMMENT}


def placement_errors(lines: list[bytes], marker_rows: list[int], suffix: str) -> list[str]:
    """The marker must not displace a shebang (line 1) or a Python encoding cookie (line 1 or 2)."""
    errors = []
    first = marker_rows[0]
    leading = [row for row, line in enumerate(lines) if row >= first and line.strip() and row not in marker_rows]
    if leading and lines[leading[0]].startswith(b"#!"):
        errors.append(f"marker precedes the shebang; the shebang must stay on line 1 (line {leading[0] + 1})")
    if suffix == ".py":
        header = [row for row in range(min(len(lines), 3)) if row not in marker_rows and CODING_COOKIE.match(lines[row])]
        if header and header[0] >= 2 and first < header[0]:
            errors.append(f"marker pushes the encoding declaration to line {header[0] + 1}; it must be on line 1 or 2")
    return errors


def validate_paths(paths: list[Path], registry: dict, root: Path = ROOT) -> list[str]:
    errors = []
    for relative in paths:
        expected = expected_line(relative, registry)
        if expected is None:
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"tracked marker-covered file is missing: {relative}")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            errors.append(f"cannot read marker-covered file {relative}: {exc}")
            continue
        lines = data.splitlines()
        marker_rows = [row for row, line in enumerate(lines) if line.strip() == expected]
        if marker_rows and relative.suffix.lower() == ".py":
            try:
                comments = python_comment_lines(data)
            except ValueError as exc:
                errors.append(f"{relative}: {exc}")
                continue
            marker_rows = [row for row in marker_rows if row in comments]
        if not marker_rows:
            errors.append(f"missing whole-line marker in {relative}")
            continue
        errors.extend(f"{relative}: {error}" for error in placement_errors(lines, marker_rows, relative.suffix.lower()))
    return errors


def validate_exact_files(registry: dict, tracked: list[Path], root: Path = ROOT) -> list[str]:
    tracked_names = {path.as_posix() for path in tracked}
    errors = []
    for relative in registry["exact_files"]:
        if relative not in tracked_names:
            errors.append(f"required marker-covered file is not tracked by Git: {relative}")
        elif not (root / relative).is_file():
            errors.append(f"required marker-covered file is missing: {relative}")
    return errors


def main(argv: list[str] | None = None, root: Path = ROOT, registry_path: Path = REGISTRY_PATH) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    errors = []
    try:
        registry = load_registry(registry_path)
        tracked = tracked_files(root)
    except ValueError as exc:
        errors.append(str(exc))
    else:
        errors.extend(validate_exact_files(registry, tracked, root))
        try:
            errors.extend(validate_paths(tracked, registry, root))
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        print("Canary validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Canary validation passed for all marker-covered tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
