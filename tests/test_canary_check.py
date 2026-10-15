#!/usr/bin/env python3
# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Regression tests for the SpecSmartDecov V2 canary validator."""

import contextlib
import importlib.util
import io
import json
import subprocess
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
            self.assertEqual([], CHECKER.validate_exact_files(registry, paths, root))
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

    def test_real_registry_loads_and_repo_passes(self) -> None:
        CHECKER.load_registry()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            status = CHECKER.main([])
        self.assertEqual(0, status, err.getvalue())

    def test_rejects_marker_above_shebang(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.registry()
            token = CHECKER.marker(registry)
            (root / "fastdecov").mkdir()
            registry["directory_suffixes"]["fastdecov"] = [".py", ".sh"]
            (root / "fastdecov" / "bad.sh").write_text(f"# {token}\n#!/bin/bash\necho hi\n")
            (root / "fastdecov" / "spaced.sh").write_text(f"# {token}\n\n#!/bin/bash\n")
            (root / "fastdecov" / "good.sh").write_text(f"#!/bin/bash\n# {token}\necho hi\n")
            errors = CHECKER.validate_paths(
                [Path("fastdecov/bad.sh"), Path("fastdecov/spaced.sh"), Path("fastdecov/good.sh")], registry, root
            )
            self.assertEqual(2, len(errors), errors)
            self.assertTrue(all("shebang" in error for error in errors))
            self.assertFalse(any("good.sh" in error for error in errors))

    def test_rejects_marker_that_displaces_encoding_cookie(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.registry()
            token = CHECKER.marker(registry)
            (root / "analysis.py").write_text(f"#!/usr/bin/env python3\n# {token}\n# -*- coding: utf-8 -*-\n")
            errors = CHECKER.validate_paths([Path("analysis.py")], registry, root)
            self.assertEqual(1, len(errors), errors)
            self.assertIn("encoding declaration", errors[0])
            (root / "analysis.py").write_text(f"#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n# {token}\n")
            self.assertEqual([], CHECKER.validate_paths([Path("analysis.py")], registry, root))

    def test_rejects_python_marker_inside_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.registry()
            (root / "analysis.py").write_text(f'"""\n# {CHECKER.marker(registry)}\n"""\n')
            self.assertEqual(
                ["missing whole-line marker in analysis.py"],
                CHECKER.validate_paths([Path("analysis.py")], registry, root),
            )

    def test_deepest_covered_directory_wins_regardless_of_order(self) -> None:
        registry = self.registry()
        for ordering in (
            {"manuscript": [".py", ".sh"], "manuscript/determinism": [".py"]},
            {"manuscript/determinism": [".py"], "manuscript": [".py", ".sh"]},
        ):
            registry["directory_suffixes"] = ordering
            self.assertFalse(CHECKER.covered(Path("manuscript/determinism/sweep.sh"), registry))
            self.assertTrue(CHECKER.covered(Path("manuscript/build.sh"), registry))

    def test_exact_file_must_be_tracked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.registry()
            for name in registry["exact_files"]:
                (root / name).write_text("")
            errors = CHECKER.validate_exact_files(registry, [Path("README.md"), Path("analysis.py")], root)
            self.assertEqual(["required marker-covered file is not tracked by Git: pyproject.toml"], errors)

    def test_rejects_comment_style_that_disagrees_with_validator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry = self.registry()
            registry["comment_styles"][".py"] = "html"
            path.write_text(json.dumps(registry))
            with self.assertRaisesRegex(ValueError, "comment_styles"):
                CHECKER.load_registry(path)

    def test_main_reports_every_error_in_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self.registry()
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry))
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / "analysis.py").write_text("print('no marker')\n")
            subprocess.run(["git", "add", "analysis.py"], cwd=root, check=True)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(1, CHECKER.main([], root=root, registry_path=registry_path))
            output = err.getvalue()
            self.assertIn("not tracked by Git: README.md", output)
            self.assertIn("not tracked by Git: pyproject.toml", output)
            self.assertIn("missing whole-line marker in analysis.py", output)


if __name__ == "__main__":
    unittest.main()
