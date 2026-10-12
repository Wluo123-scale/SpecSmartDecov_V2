"""Grid auto-detection, clipped-peak masking and the residual-limit report."""
import json
import subprocess
import sys

import numpy as np

from fastdecov import basis as B, grid, report
from test_cli import make_map


def smooth_map(rows, cols, rng):
    """Spectra whose line positions and heights vary smoothly across a rows x cols map (row-major order)."""
    x = np.linspace(600, 760, 300); ys = []
    for r in range(rows):
        for c in range(cols):
            a = 1000 + 800 * np.sin(r / 3.0) * np.cos(c / 4.0); cen = 680 + 0.6 * r - 0.4 * c
            ys.append(a * np.exp(-((x - cen) / 4.0) ** 2) + 300 * np.exp(-((x - 640 - 0.3 * c) / 6.0) ** 2) + rng.normal(0, 3, x.size))
    return x, ys


def test_grid_detect_row_major_and_serpentine():
    rng = np.random.default_rng(0)
    x, ys = smooth_map(9, 12, rng)
    ids = [f"{i + 1:04d}" for i in range(len(ys))]
    g = grid.detect(ids, [x] * len(ys), ys)
    assert (g["rows"], g["cols"], g["snake"]) == (9, 12, False), g
    order = grid.order_for(9, 12, True)                      # file i holds image pixel order[i]
    snake = [ys[p] for p in order]
    g = grid.detect(ids, [x] * len(ys), snake)
    assert (g["rows"], g["cols"], g["snake"]) == (9, 12, True), g


def test_grid_from_file_names_and_no_grid():
    ids = [f"x{c:02d}_y{r:02d}" for r in range(4) for c in range(5)]
    g = grid.detect(ids, None, None)
    assert (g["rows"], g["cols"], g["method"]) == (5, 4, "file names"), g      # last two integers: rows from x, cols from y
    assert grid.detect([f"{i}" for i in range(7)], None, None)["rows"] is None  # 7 spectra: no grid with sides >= 3


def test_clip_keep_masks_plateau_and_edges():
    y = np.r_[np.zeros(20), [100, 500, 900], [1000.0] * 6, [1100, 600, 200], np.zeros(20)]
    keep = B.clip_keep(y)
    assert not keep[22:30].any()              # plateau (23..28) plus one pixel on each side
    assert keep[:22].all() and keep[30:].all()
    assert B.clip_keep(np.sin(np.linspace(0, 3, 50)) * 1000).all()   # no repeated values: nothing masked


def test_residual_limit_rule():
    assert np.allclose(report.residual_limit(np.array([0.0, 100.0, 1111.111, 1e4])), [100, 100, 100, 300])
    assert np.allclose(report.residual_limit(np.array([1e4]), 50, 2), [200])


def test_cli_auto_grid_and_residual_columns(tmp_path):
    d = make_map(tmp_path); out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "fastdecov", "fit", str(d), str(out), "--threads", "2", "--residual-limit", "50,3"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    run = json.loads((out / "run.json").read_text())
    assert "rows" in run["grid"] and run["residual_limit"] == [50.0, 3.0]
    q = run["fit_quality"]["residual_limit"]
    assert q["floor"] == 50.0 and 0.0 <= q["pass_fraction"] <= 1.0
    head = (out / "spectra.tsv").read_text().splitlines()[0].split("\t")
    assert {"clipped_pixels", "max_abs_residual", "residual_limit_ratio", "residual_ok"} <= set(head)


def test_local_emitter_confined_to_patch():
    """A sharp line present only in a 2x2 patch of a 6x6 map is added as a local emitter; its amplitude is zero outside."""
    from fastdecov import api, emitter
    rng = np.random.default_rng(3); x = np.linspace(600, 760, 400)
    ys, ids = [], []
    for r in range(6):
        for c in range(6):
            p = np.array([10.0, 0.0, 1500 * (1 + 0.2 * rng.random()), 700.0 + 0.2 * r, 0.5, 4.0])
            y = emitter.evaluate(p, x, [0])
            if r in (2, 3) and c in (2, 3):
                y = y + emitter.evaluate(np.array([0.0, 0.0, 900.0, 650.0, 0.3, 2.0]), x, [0])
            ys.append(y + rng.normal(0, 3, x.size)); ids.append(f"{r * 6 + c + 1:04d}")
    m = api.SpectrumMap(ids, [x] * 36, ys)
    res = api.fit_map(m, "v3", threads=2, local=True, grid=(6, 6))
    cen = [res.median(0, k) for k in range(res.K)]
    k = int(np.argmin([abs(c - 650) for c in cen]))
    assert abs(cen[k] - 650) < 0.5, cen
    patch = {2 * 6 + 2, 2 * 6 + 3, 3 * 6 + 2, 3 * 6 + 3}
    far = [s for s in range(36) if s not in patch and all(abs(s // 6 - q // 6) > 1 or abs(s % 6 - q % 6) > 1 for q in patch)]
    assert np.all(res.table[far, k, 3] < 1e-3), res.table[far, k, 3]


def test_snr_classes_in_outputs(tmp_path):
    d = make_map(tmp_path); out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "fastdecov", "fit", str(d), str(out), "--threads", "2", "--grid", "4x6",
                        "--snr-classes", "20,300"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [l.split("\t") for l in (out / "spectra.tsv").read_text().splitlines()]
    i = rows[0].index("spectrum_class"); j = rows[0].index("peak_snr")
    for row in rows[1:]:
        snr = float(row[j]); assert row[i] == ("substrate" if snr < 20 else "weak" if snr < 300 else "emitter")
    q = json.loads((out / "emitter_report.json").read_text())["fit_quality"]["snr_classes"]
    assert sum(v["n"] for v in q["by_class"].values()) == 24
    assert (out / "snr_map.png").is_file()
    assert report.classify(np.array([5.0, 20.0, 299.0, 300.0])).tolist() == [0, 1, 1, 2]


def test_implied_huang_rhys():
    assert report.implied_huang_rhys(np.exp(-1.5)) == 1.5
    assert report.implied_huang_rhys(1.0) is None and report.implied_huang_rhys(0.0) is None and report.implied_huang_rhys(None) is None


def test_pair_option_error_is_generic(tmp_path):
    d = make_map(tmp_path)
    r = subprocess.run([sys.executable, "-m", "fastdecov", "fit", str(d), str(tmp_path / "o"), "--snr-classes", "abc"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "two comma-separated numbers" in r.stderr and "residual limit" not in r.stderr


def test_figure_legend_lists_masked_points(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seen = {}
    real_close = plt.close
    def close(fig):
        seen["labels"] = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        real_close(fig)
    monkeypatch.setattr(plt, "close", close)
    x = np.linspace(600, 700, 200); y = 1000 * np.exp(-((x - 650) / 3) ** 2); keep = np.ones(200, bool); keep[100:103] = False
    report.figure(tmp_path / "f", "s1", x, y, keep, np.array([0.0, 0.0, 1000.0, 650.0, 0.5, 4.0]), [0], ("png",), (100, 3))
    assert "masked" in seen["labels"]


def test_pair_rule_narrow_on_broad():
    from fastdecov import basis3
    o = dict(basis3.OPTS3)
    corr_low, corr_high = (lambda: 0.1), (lambda: 0.95)
    # narrow 4 nm line 5 nm from a 25 nm band: resolved under the narrow rule unless the band's map tracks the line
    assert not basis3.pair_unresolved(5.0, 4.0, 25.0, corr_low, o)
    assert basis3.pair_unresolved(5.0, 4.0, 25.0, corr_high, o)          # pedestal that mimics Lorentzian wings
    assert basis3.pair_unresolved(2.0, 4.0, 25.0, corr_low, o)           # closer than 0.85 x the narrow FWHM
    assert basis3.pair_unresolved(5.0, 4.0, 25.0, corr_low, {**o, "pair_rule": "mean"})   # 2.1 rule
    # similar widths: unchanged 2.1 rule (0.85 x mean FWHM)
    assert basis3.pair_unresolved(3.0, 4.0, 4.5, corr_low, o) and not basis3.pair_unresolved(4.0, 4.0, 4.5, corr_low, o)


def test_background_floor():
    """A strong line's tail must not be absorbed by a background that dips below -3 sigma."""
    from fastdecov import basis3, emitter, model
    rng = np.random.default_rng(5); x = np.linspace(550, 800, 500)
    p = np.array([0.0, 0.0, 20000.0, 760.0, 0.1, 8.0, 300.0, 600.0, 0.5, 6.0])
    y = emitter.evaluate(p, x, [0, 0]) + rng.normal(0, 5, x.size)
    b = basis3.EmitterBasis([600.0, 760.0], [597.0, 757.0], [603.0, 763.0], [1.0, 1.0], [20.0, 20.0])
    for floor in (3.0, None):
        f = basis3.fit_set([x], [y], b, ("mid", 5.0), {**basis3.OPTS3, "bg_floor_sigma": floor})
        t = (x - model.X_REF) / model.X_SCALE; bg = f.params[0, 0] + f.params[0, 1] * t
        if floor:
            assert bg.min() >= -3.0 * 5.0 * 1.2, bg.min()


def _replica_map(n=36, with_replica=True, rng=None):
    """6x6 map: one emitter at ~690 nm whose strength and position vary across the map, plus (optionally) an
    optical-phonon replica 31 meV below it, which the emitter model does not include. The emitter is absent from the
    first row, so "present only where its emitter is" is an informative test."""
    from fastdecov import api, emitter
    rng = rng or np.random.default_rng(7); x = np.linspace(620, 760, 450); ys, ids = [], []
    for i in range(n):
        r, c = divmod(i, 6)
        a = 0.0 if r == 0 else 4000 * (0.3 + 0.7 * np.exp(-((r - 2.5) ** 2 + (c - 2.5) ** 2) / 6))   # absent in row 0
        c0 = 690.0 + 0.25 * (r - 2.5)
        y = emitter.evaluate(np.array([5.0, 0.0, a, c0, 0.3, 3.0]), x, [0])
        if with_replica:
            e = 1239.84193 / c0 - 0.031; cr = 1239.84193 / e
            y = y + 0.06 * a * np.exp(-4 * np.log(2) * ((x - cr) / 3.2) ** 2)
        y = y + 60.0 * np.exp(-((x - 650.0) / 8.0) ** 2)            # a weak map-wide band, so no spectrum is empty
        ys.append(y + rng.normal(0, 4, x.size)); ids.append(f"{i + 1:04d}")
    return api.SpectrumMap(ids, [x] * n, ys)


def test_replica_classified_and_absent_without():
    from fastdecov import api, physics, replicas, report
    phys = physics.load()
    for with_rep in (True, False):
        m = _replica_map(with_replica=with_rep)
        res = api.fit_map(m, "v3", threads=2, grid=(6, 6))
        rows = report.ranking(res, phys); kinds = [r["kind"] for r in rows]
        lines = replicas.fitted_line_replicas(res, phys, kinds=kinds)          # replica discovered as a fitted line
        feats = replicas.analyse(res, phys, kinds=kinds)                      # or left in the residual
        offs = [t["offset_meV"] for t in lines.values()] + \
               [f["offset_meV"] for f in feats if f["classification"] == "optical-phonon replica"]
        if with_rep:
            assert len(offs) == 1 and abs(offs[0] - 31.0) < 3.0, (lines, feats)
        else:
            assert not offs, (lines, feats)


def test_voigt_intrinsic_nu_and_grating_choice():
    from fastdecov import lineshape, shortlist, physics
    assert lineshape.nu_from_widths(1.0, 0.0) == 0.0 and lineshape.nu_from_widths(0.0, 1.0) == 1.0
    assert 0.2 < lineshape.nu_from_widths(1.0, 1.0) < 0.8
    phys = physics.load()
    assert shortlist._grating(2.0, phys) == "600 g/mm"         # 0.3 meV <= half of 2 meV
    assert shortlist._grating(0.3, phys) == "2400 g/mm"        # needs the finest grating
    assert shortlist._grating(None, phys) == "2400 g/mm"


def test_shortlist_written_and_excludes_broad_and_replicas(tmp_path):
    d = make_map(tmp_path); out = tmp_path / "out"
    r = subprocess.run([sys.executable, "-m", "fastdecov", "fit", str(d), str(out), "--threads", "2", "--grid", "4x6"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    sl = json.loads((out / "shortlist.json").read_text())["items"]
    rows = {e["emitter"]: e for e in json.loads((out / "emitter_report.json").read_text())["emitters"]}
    for it in sl:
        assert it["tier"] in (1, 2, 3)
        if it["emitter"] is not None:
            assert rows[it["emitter"]]["kind"] == "emitter line"
    assert [it["rank"] for it in sl] == list(range(1, len(sl) + 1))
    assert (out / "shortlist.md").is_file()
