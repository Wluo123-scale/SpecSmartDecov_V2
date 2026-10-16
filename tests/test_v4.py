# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""V4: a fixed set of map-wide emitters, seeded from the brightest spectra, colour shifting within the strain window."""
import json
import subprocess
import sys

import numpy as np
import pytest

from fastdecov import api, basis4, cli, emitter

TRUE = (650.0, 690.0, 720.0)


def strained_map(n=48, drift=2.0, seed=3):
    """n spectra on a line scan; every emitter drifts by up to +-drift nm (strain) and its brightness varies (0 in some)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(600, 760, 400)
    xs, ys = [], []
    for i in range(n):
        u = i / (n - 1)
        p = [10.0, 0.0]
        for k, c in enumerate(TRUE):
            shift = drift * np.sin(2 * np.pi * (u + k / 3))
            amp = 4000 * max(0.0, np.cos(np.pi * (u - k / 3))) ** 2 * (0.7 + 0.6 * rng.random())
            p += [amp, c + shift, 0.4, 4.0]
        ys.append(emitter.evaluate(np.array(p), x, [0, 0, 0]) + rng.normal(0, 3, x.size)); xs.append(x)
    return xs, ys


def test_v4_finds_fixed_emitter_set_despite_strain_drift():
    xs, ys = strained_map()
    res = api.fit_map(api.SpectrumMap([f"{i + 1:04d}" for i in range(len(xs))], xs, ys), "v4", threads=2, grid=None)
    assert res.method == "v4" and res.K == 3, res.extra["reference_centers"]
    for c, t in zip(sorted(res.extra["reference_centers"]), TRUE):
        assert abs(c - t) <= 3.0
    lo, hi = np.array(res.extra["center_lo"]), np.array(res.extra["center_hi"])
    assert np.all(hi - lo <= 10.0 + 1e-9) and np.all(lo[1:] >= hi[:-1] - 1e-9)      # <= +-5 nm windows, never overlapping
    # per-spectrum colour follows the strain drift within the window, for every spectrum where the emitter is bright
    for k in range(3):
        m = res.share[:, k] >= 0.1
        assert m.sum() >= 5
        assert np.all((res.table[m, k, 0] >= lo[k] - 1e-6) & (res.table[m, k, 0] <= hi[k] + 1e-6))
        assert np.ptp(res.table[m, k, 0]) > 1.0                                          # it really moves
    assert res.extra["n_seeds"] >= 1 and res.extra["history"][0]["step"] == "seed"


def test_seed_assignment_is_brightest_first():
    """The brightest line fixes the reference; dimmer lines within the window join it; a resolved line founds another."""
    cens = [dict(lines=[dict(center=700.0, width=4.0, height=100.0, share=0.5)]),
            dict(lines=[dict(center=702.5, width=4.0, height=900.0, share=0.5)]),
            dict(lines=[dict(center=700.5, width=4.0, height=50.0, share=0.5),
                        dict(center=712.0, width=4.0, height=40.0, share=0.5)])]
    em = basis4.seed_emitters(cens, [0, 1, 2], (600.0, 760.0), 3.0, 8.0, {**basis4.OPTS4, "min_sep_nm": 3.4})
    refs = sorted((round(e[0], 1), e[1]) for e in em)
    assert [r[1] for r in refs] == [3, 1] and abs(refs[1][0] - 712.0) < 1e-6
    assert 700.0 <= refs[0][0] <= 702.5                         # middle of the colour range of its three lines


def test_seed_merge_one_emitter_seen_at_both_ends_of_its_drift():
    """Brightest line at 688 (one end of the drift): lines at 691-692 lie > 3 nm away, found a second candidate, never
    share a spectrum with the first, and span <= 6 nm together: one emitter."""
    cens = [dict(lines=[dict(center=688.0 + 0.1 * i, width=4.0, height=1000.0 - i, share=0.5)]) for i in range(10)] + \
           [dict(lines=[dict(center=691.5 + 0.05 * i, width=4.0, height=500.0 - i, share=0.5)]) for i in range(10)]
    em = basis4.seed_emitters(cens, list(range(20)), (600.0, 760.0), 3.0, 8.0, {**basis4.OPTS4, "min_sep_nm": 3.4})
    assert len(em) == 1 and em[0][1] == 20 and 689.0 < em[0][0] < 691.5


def test_seed_retains_far_strain_endpoint_when_spacing_exceeds_drift_span():
    """An endpoint outside the first candidate's half-window must remain a provisional candidate even when the learned
    emitter spacing is larger than the endpoint separation; the merge pass then identifies the one non-cooccurring emitter."""
    cens = [dict(lines=[dict(center=700.0, width=4.0, height=10.0, share=0.5)]) for _ in range(10)] + \
           [dict(lines=[dict(center=709.0, width=4.0, height=1000.0, share=0.5)]) for _ in range(20)]
    em = basis4.seed_emitters(cens, list(range(len(cens))), (600.0, 760.0), 5.0, 8.0,
                              {**basis4.OPTS4, "min_sep_nm": 12.0, "window_min_nm": 3.0})
    assert len(em) == 1 and em[0][1] == 30
    assert 703.0 <= em[0][0] <= 706.0


def test_seed_no_merge_when_lines_share_spectra():
    cens = [dict(lines=[dict(center=688.0, width=2.0, height=1000.0, share=0.5),
                        dict(center=692.0, width=2.0, height=800.0, share=0.5)]) for _ in range(10)]
    em = basis4.seed_emitters(cens, list(range(10)), (600.0, 760.0), 3.0, 8.0, {**basis4.OPTS4, "min_sep_nm": 3.4})
    assert len(em) == 2


def test_seed_assignment_ignores_edge_artefacts():
    cens = [dict(lines=[dict(center=598.0, width=10.0, height=5000.0, share=0.5),
                        dict(center=700.0, width=4.0, height=10.0, share=0.5)])]
    em = basis4.seed_emitters(cens, [0], (600.0, 760.0), 3.0, 8.0, {**basis4.OPTS4, "min_sep_nm": 3.4})
    assert [round(e[0]) for e in em] == [700]


@pytest.mark.parametrize("opts, message", [
    ({"window_min_nm": 5.0, "window_max_nm": 3.0}, "window_max_nm"),
    ({"window_min_nm": 0.0}, "window_min_nm"),
    ({"width_max_nm": 0.0}, "width_max_nm"),
    ({"min_sep_nm": -1.0}, "min_sep_nm"),
])
def test_v4_api_rejects_invalid_physical_bounds(opts, message):
    with pytest.raises(ValueError, match=message):
        basis4.fit([], [], opts=opts)


@pytest.mark.parametrize("args", [
    ["--window=5,3"], ["--window=0,5"], ["--width-max", "0"], ["--min-sep", "-1"],
])
def test_cli_rejects_invalid_v4_physical_bounds(args):
    with pytest.raises(SystemExit) as err:
        cli.main(["fit", "input", "output", "--method", "v4", *args])
    assert err.value.code == 2


def test_cli_v4(tmp_path):
    xs, ys = strained_map(n=24)
    d = tmp_path / "spectra"; d.mkdir()
    for i, (x, y) in enumerate(zip(xs, ys)):
        np.savetxt(d / f"{i + 1:04d}.txt", np.column_stack([x, y]), fmt="%.6f", delimiter="\t")
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "fastdecov", "fit", str(d), str(out), "--method", "v4", "--grid", "4x6",
                        "--threads", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    b = json.loads((out / "basis.json").read_text())
    assert b["method"] == "v4" and b["K"] == 3 and "history" in b
    assert b["history"][0]["step"] == "seed" and b["min_sep_nm"] > 0


def test_windows_clip_at_midpoints_and_reopen_when_a_neighbour_goes():
    W = basis4.Windows([700.0, 708.0, 730.0], [5.0, 5.0, 3.0], [5.0, 5.0, 3.0])
    lo, hi = W.bounds()
    assert np.allclose(lo, [695.0, 704.0, 727.0]) and np.allclose(hi, [704.0, 713.0, 733.0])
    lo2, hi2 = W.without(1).bounds()
    assert np.allclose(hi2, [705.0, 733.0])                     # 700 nm opens up to its own +5 nm again
    W3 = basis4.Windows([700.0], [3.0], [3.0]).widened(0, "hi", 1.0, 5.0).widened(0, "hi", 1.0, 5.0).widened(0, "hi", 1.0, 5.0)
    assert np.allclose(W3.bounds()[1], [705.0]) and np.allclose(W3.bounds()[0], [697.0])   # capped at +5, one side only


def test_v4_widens_window_for_an_emitter_drifting_beyond_3nm():
    """Emitters drifting +-4.5 nm pile up on a +-3 nm window edge; v4 opens the window (<= 5 nm) and keeps K = 3."""
    xs, ys = strained_map(drift=4.5)
    res = api.fit_map(api.SpectrumMap([f"{i + 1:04d}" for i in range(len(xs))], xs, ys), "v4", threads=2, grid=None,
                      window_min_nm=3.0, window_max_nm=5.0)
    assert res.K == 3, res.extra["reference_centers"]
    half = (np.array(res.extra["center_hi"]) - np.array(res.extra["center_lo"])) / 2
    assert np.all(half <= 5.0 + 1e-9) and np.any(half > 3.0 + 1e-9), half
    # the windows hold the drift, whether seeding (both drift ends merged) or widening opened them: no colour piles up
    # on a window edge
    assert np.all(np.array(res.extra["edge_fraction"]) <= basis4.OPTS4["widen_edge_frac"]), res.extra["edge_fraction"]


def test_widen_opens_only_the_pinned_side_up_to_the_maximum(monkeypatch):
    """A fit whose colour keeps sitting on the upper window edge (in 60 % of spectra): that side opens 1 nm per round
    up to window_max_nm, the lower side stays, and every step is recorded."""
    from types import SimpleNamespace

    def fit_on_upper_edge(xm, ym, b, starts, o, workers=None, idx=None):
        c = np.where(np.arange(10) < 6, b.hi[0], 700.0)                 # 6 of 10 spectra on the upper edge
        tab = np.zeros((10, 1, 8)); tab[:, 0, 0] = c
        return SimpleNamespace(share=np.ones((10, 1)), tab=tab)
    monkeypatch.setattr(basis4.B3, "fit_set", fit_on_upper_edge)
    W = basis4.Windows([700.0], [3.0], [3.0]); hist = []
    o = {**basis4.OPTS4, "_workers": 1, "starts": (8.0,)}
    fit0 = fit_on_upper_edge(None, None, W.basis(1.0, 8.0), None, o)
    W2, _ = basis4._widen([None] * 10, [None] * 10, W, fit0, o, hist, 1.0, 8.0, 5.0)
    assert np.allclose(W2.hhi, [5.0]) and np.allclose(W2.hlo, [3.0])
    assert [h["step"] for h in hist] == ["widen", "widen"] and all("+" in h["windows"][0] for h in hist)


def test_spacing_is_learned_from_brightest_spectra():
    """Bright spectra: a band of three components 9 nm apart plus a far line; dim spectra: noise only. The learned
    spacing is the within-band one (~9 nm), not the far one and not influenced by the dim spectra."""
    rng = np.random.default_rng(5); x = np.linspace(600, 760, 400); xs, ys = [], []
    for i in range(40):
        bright = i < 10
        p = [10.0, 0.0] + sum(([3000.0 if bright else 0.0, c, 0.3, 4.0] for c in (700.0, 709.0, 718.0, 640.0)), [])
        ys.append(emitter.evaluate(np.array(p), x, [0, 0, 0, 0]) + rng.normal(0, 3, x.size)); xs.append(x)
    snr = np.array([np.ptp(y) / 3 for y in ys])
    sep, st = basis4.learn_min_sep(xs, ys, snr, 8.0, {**basis4.OPTS4, "sep_min_spectra": 10, "sep_min_pairs": 5})
    assert sep is not None and 8.0 <= sep <= 10.0, st
    assert st["spectra"] == 10 and st["within_band_pairs"] >= 10
