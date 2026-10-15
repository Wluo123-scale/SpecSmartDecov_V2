# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""End-to-end smoke tests of the CLI and SDK on a tiny generated map (two lines, one with a red shoulder)."""
import json
import subprocess
import sys

import numpy as np
import pytest

from fastdecov import api, emitter


def make_map(tmp_path, n=24):
    rng = np.random.default_rng(1)
    x = np.linspace(600, 760, 400)
    d = tmp_path / "spectra"; d.mkdir()
    for i in range(n):
        p = np.array([10.0, 0.0, 800 * (1 + 0.3 * rng.random()), 650.0, 0.3, 4.0,
                      1500 * (1 + 0.3 * rng.random()), 700.0, 0.6, 3.0, 0.15, 1.5, 3.0])
        y = emitter.evaluate(p, x, [0, 1]) + rng.normal(0, 3, x.size)
        np.savetxt(d / f"{i + 1:04d}.txt", np.column_stack([x, y]), fmt="%.6f", delimiter="\t")
    return d


@pytest.mark.parametrize("method", ["v3", "v2"])
def test_cli_fit(tmp_path, method):
    d = make_map(tmp_path); out = tmp_path / f"out_{method}"
    r = subprocess.run([sys.executable, "-m", "fastdecov", "fit", str(d), str(out), "--method", method, "--grid", "4x6",
                        "--threads", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for f in ("basis.json", "emitters.tsv", "spectra.tsv", "emitter_report.md", "emitter_report.json", "emitter_maps.png", "run.json"):
        assert (out / f).is_file(), f
    assert len(list((out / "dc").glob("dc_*.txt"))) == 24
    b = json.loads((out / "basis.json").read_text())
    centres = sorted(b["centers"])
    assert any(abs(c - 650) < 1.0 for c in centres) and any(abs(c - 700) < 1.0 for c in centres)
    q = json.loads((out / "run.json").read_text())["fit_quality"]
    assert q["pass_rate"] > 0.9


def test_sdk_v3_one_emitter_per_line(tmp_path):
    m = api.load_map(make_map(tmp_path))
    res = api.fit_map(m, "v3", threads=2)
    cen = sorted(res.median(0, k) for k in range(res.K))
    assert res.K == 2, cen                        # the red shoulder is a sideband, not a third line
    assert abs(cen[0] - 650) < 0.5 and abs(cen[1] - 700) < 0.5
