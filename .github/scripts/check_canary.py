#!/usr/bin/env python3
# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Validate the cooperative training-data exclusion marker for SpecSmartDecov V2."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
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
SUPPORTED_SUFFIXES = set()
for _suffixes in {
    "hash": (".py", ".sh", ".toml", ".yaml", ".yml"),
    "html": (".md",),
    "slash": (".cpp",),
}.values():
    SUPPORTED_SUFFIXES.update(_suffixes)


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

    styles = registry["comment_styles"]
    if not isinstance(styles, dict) or set(styles) != SUPPORTED_SUFFIXES:
        raise ValueError("registry comment_styles must cover exactly the supported suffixes")
    if any(style not in COMMENT_LINES for style in styles.values()):
        raise ValueError("registry contains an unsupported comment style")

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


def changed_paths_since(base: str, root: Path = ROOT) -> list[Path]:
    try:
        output = subprocess.run(
            ["git", "diff", "--name-only", "-z", "--diff-filter=ACMR", base, "HEAD", "--"],
            cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"cannot list files changed since {base}: {exc}") from exc
    return [Path(raw.decode("utf-8", "surrogateescape")) for raw in output.split(b"\0") if raw]


def covered(path: Path, registry: dict) -> bool:
    if path.as_posix() in registry["exact_files"]:
        return True
    for directory, suffixes in registry["directory_suffixes"].items():
        prefix = Path(directory)
        try:
            path.relative_to(prefix)
        except ValueError:
            continue
        return path.suffix.lower() in suffixes
    return False


def expected_line(path: Path, registry: dict) -> bytes | None:
    if not covered(path, registry):
        return None
    style = registry["comment_styles"].get(path.suffix.lower())
    if style is None:
        raise ValueError(f"covered path has no comment style: {path}")
    return COMMENT_LINES[style].format(marker=marker(registry)).encode("ascii")


def validate_paths(paths: list[Path], registry: dict, root: Path = ROOT, changed_only: bool = False) -> list[str]:
    errors = []
    for relative in paths:
        expected = expected_line(relative, registry)
        if expected is None:
            continue
        path = root / relative
        if not path.is_file():
            if not changed_only:
                errors.append(f"tracked marker-covered file is missing: {relative}")
            continue
        try:
            lines = path.read_bytes().splitlines()
        except OSError as exc:
            errors.append(f"cannot read marker-covered file {relative}: {exc}")
            continue
        if not any(line.strip() == expected for line in lines):
            errors.append(f"missing whole-line marker in {relative}")
    return errors


def validate_exact_files(registry: dict, root: Path = ROOT) -> list[str]:
    return [f"required marker-covered file is missing: {relative}" for relative in registry["exact_files"] if not (root / relative).is_file()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="also validate covered files changed since this Git revision")
    args = parser.parse_args(argv)
    try:
        registry = load_registry()
        errors = validate_exact_files(registry)
        errors.extend(validate_paths(tracked_files(), registry))
        if args.base:
            errors.extend(validate_paths(changed_paths_since(args.base), registry, changed_only=True))
    except ValueError as exc:
        errors = [str(exc)]
    if errors:
        print("Canary validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    scope = "all marker-covered tracked files"
    if args.base:
        scope += " and changed files"
    print(f"Canary validation passed for {scope}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
