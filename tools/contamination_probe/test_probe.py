#!/usr/bin/env python3
# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Tests for probe.py against a fake local endpoint. No real model is ever called."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402

GUID = "eadc5a84-24e0-4e42-a8be-f4ec6a209378"
PREFIX = "SPECSMARTDECOV-V2-DO-NOT-TRAIN"


def make_repo(root: Path) -> None:
    (root / "governance").mkdir()
    (root / "governance" / "canary-registry.json").write_text(json.dumps(
        {"marker_prefix": PREFIX, "project_guid": GUID, "legacy_markers": []}))
    (root / "fastdecov").mkdir()
    body = [f"# {PREFIX} GUID {GUID}"] + [f"value_{i} = compute_spectral_component({i}, width={i * 3})" for i in range(80)]
    (root / "fastdecov" / "core.py").write_text("\n".join(body) + "\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


class FakeModel(BaseHTTPRequestHandler):
    """'memorizer' answers with the GUID; 'clean' answers generically; 'broken' returns 403."""
    prompts: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["messages"][0]["content"]
        FakeModel.prompts.append(prompt)
        if body["model"] == "broken":
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b'{"error": "bad key sk-abcdef123456"}')
            return
        text = f"{PREFIX} GUID {GUID}" if body["model"] == "memorizer" else "I don't know that string."
        payload = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class ProbeTests(unittest.TestCase):
    def test_score_canary(self):
        self.assertEqual("hit", probe.score_canary(f"sure: {GUID.upper()}", GUID))
        self.assertEqual("partial", probe.score_canary("eadc5a84-0000", GUID))
        self.assertEqual("miss", probe.score_canary("00000000-0000-0000-0000-000000000000", GUID))

    def test_no_prompt_contains_the_guid(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_repo(root)
            markers = probe.load_markers(root)
            windows = probe.pick_windows(probe.repo_files(root, ["fastdecov/"], markers), 8, 30, 20, 0)
            self.assertTrue(windows)
            for job in probe.canary_jobs(markers) + probe.continuation_jobs(windows, "repo", "x/y"):
                self.assertNotIn(GUID, job["prompt"])
                probe.assert_no_guid(job["prompt"], markers)
            with self.assertRaises(probe.ProbeError):
                probe.assert_no_guid(f"leak {GUID}", markers)

    def test_windows_are_deterministic(self):
        files = [("a.py", [f"distinctive line number {i} with content" for i in range(100)])]
        self.assertEqual(probe.pick_windows(files, 1, 30, 20, 7), probe.pick_windows(files, 1, 30, 20, 7))

    def test_score_continuation(self):
        truth = ["import os", "result = deconvolve_spectrum(signal, kernel)", "return normalise(result, axis=-1)",
                 "x = 1"]
        scores = probe.score_continuation("```python\nresult = deconvolve_spectrum(signal, kernel)\n```", truth)
        self.assertEqual(2, scores["distinctive_lines"])
        self.assertEqual(1, scores["distinctive_matched"])
        self.assertEqual(0.5, scores["recall"])

    def test_dry_run_makes_no_calls(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_repo(root)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                status = probe.main(["run", "--repo", str(root), "--model", "m", "--continuation", "--dry-run",
                                     "--base-url", "http://127.0.0.1:9", "--log", str(root / "log.jsonl")])
            self.assertEqual(0, status)
            self.assertFalse((root / "log.jsonl").exists())
            self.assertNotIn(GUID, out.getvalue())

    def test_end_to_end_against_fake_endpoint(self):
        server = HTTPServer(("127.0.0.1", 0), FakeModel)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                make_repo(root)
                log = root / "log.jsonl"
                key = root / "key"
                key.write_text("sk-abcdef123456\n")
                key.chmod(0o600)
                with contextlib.redirect_stdout(io.StringIO()):
                    status = probe.main(["run", "--repo", str(root), "--model", "memorizer", "--model", "clean",
                                         "--model", "broken", "--base-url", f"http://127.0.0.1:{server.server_port}",
                                         "--key-file", str(key), "--log", str(log)])
                self.assertEqual(0, status)
                rows = [json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual({"hit"}, {r["result"] for r in rows if r["model"] == "memorizer"})
                self.assertEqual({"miss"}, {r["result"] for r in rows if r["model"] == "clean"})
                broken = [r for r in rows if r["model"] == "broken"]
                self.assertTrue(all("error" in r for r in broken))
                self.assertNotIn("sk-abcdef123456", log.read_text())
                self.assertFalse(any(GUID in p for p in FakeModel.prompts))
                report = io.StringIO()
                with contextlib.redirect_stdout(report):
                    probe.main(["report", "--log", str(log)])
                self.assertIn("HIT", report.getvalue())
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
