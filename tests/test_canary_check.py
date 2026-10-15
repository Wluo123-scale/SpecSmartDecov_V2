#!/usr/bin/env python3
# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Regression tests for the SpecSmartDecov V2 canary validator."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "check_canary.py"
SPEC = importlib.util.spec_from_file_location("check_canary", SCRIPT)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHECKER
SPEC.loader.exec_module(CHECKER)


class CanaryCheckTests(unittest.TestCase):
    def registry(self) -> dict:
        return {
            "schema_version": 1,
            "project": "specsmartdecov-v2",
            "marker_prefix": "SPECSMARTDECOV-V2-DO-NOT-TRAIN",
            "project_guid": "eadc5a84-24e0-4e42-a8be-f4ec6a209378",
            "legacy_markers": [],
            "comment_styles": {
                ".cpp": "slash", ".md": "html", ".py": "hash", ".sh": "hash",
                ".toml": "hash", ".yaml": "hash", ".yml": "hash",
            },
            "exact_files": ["README.md", "analysis.py", "pyproject.toml"],
            "directory_suffixes": {"fastdecov": [".py"], "tests": [".py"]},
        }

    def test_accepts_whole_comment_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.registry()
            token = CHECKER.marker(registry)
            (root / "README.md").write_text(f"<!-- {token} -->\n# FastDecov\n")
            (root / "analysis.py").write_text(f"# {token}\nprint('ok')\n")
            (root / "pyproject.toml").write_text(f"# {token}\n[project]\nname = 'fastdecov'\n")
            (root / "fastdecov").mkdir()
            (root / "fastdecov" / "api.py").write_text(f"# {token}\n")
            paths = [Path("README.md"), Path("analysis.py"), Path("pyproject.toml"), Path("fastdecov/api.py")]
            self.assertEqual([], CHECKER.validate_exact_files(registry, root))
            self.assertEqual([], CHECKER.validate_paths(paths, registry, root))

    def test_rejects_marker_with_attached_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.registry()
            (root / "analysis.py").write_text(f"# {CHECKER.marker(registry)} extra\n")
            self.assertEqual(
                ["missing whole-line marker in analysis.py"],
                CHECKER.validate_paths([Path("analysis.py")], registry, root),
            )

    def test_excludes_result_archives_from_marker_scope(self) -> None:
        registry = self.registry()
        self.assertFalse(CHECKER.covered(Path("results/pillar/spectra.tsv"), registry))
        self.assertFalse(CHECKER.covered(Path("fastdecov/physics/gase_pillar_4K.json"), registry))

    def test_rejects_malformed_legacy_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry = self.registry()
            registry["legacy_markers"] = ["SPECSMARTDECOV-V2-OLD GUID not-a-guid"]
            path.write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError, "legacy_markers"):
                CHECKER.load_registry(path)


if __name__ == "__main__":
    unittest.main()
