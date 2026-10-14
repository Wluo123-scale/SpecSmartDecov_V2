"""Writers and the emitter ranking shared by the CLI and the SDK.

All methods are reported in the V3 emitter layout (emitter.py): V2 lines are emitters without a phonon sideband,
and legacy fits additionally have a zero background slope.
"""
import json
from pathlib import Path

import numpy as np

from . import basis as B, emitter, grid as G, legacy, lineshape as LS, physics, replicas as RP, shortlist as SL

# Ranking for single-photon (g2) follow-up. nu and the sideband ZPL fraction are reported, not used as gates: at
# 150 g/mm the antibunched ~692 nm emitter of the reference study (g2(0) = 0.43) looks Gaussian-dominated, because the
# exciton/biexciton structure is unresolved and spectral diffusion adds Gaussian broadening.
RANK = dict(strong_snr=50.0, share_for_nu=0.10, sharp_x_instrument=3.0, isolated_x_fwhm=1.5, sampling_px=2.5,
            broad_x_instrument=5.0)
# Components wider than broad_x_instrument x the instrument FWHM are reported as "broad" (e.g. GaSe band-edge emission
# or an unresolved ensemble),
# not as single-emitter lines; they are fitted like any other component but are never g2 candidates.
PASS = dict(rms_peak=0.025, rms_sigma=2.0, max_peak=0.12, max_sigma=8.0)
# Per-pixel residual limit |r| <= max(floor, k_sqrt * sqrt(y)): an absolute floor for the baseline and weak lines, and a
# shot-noise-like allowance that grows with the signal (with counts ~ detected photons, sqrt(y) is the Poisson noise).
RESIDUAL_LIMIT = dict(floor=100.0, k_sqrt=3.0)
# Spectrum classes by peak SNR (peak above the edge baseline / noise sigma): substrate below the census "emitting"
# threshold, weak in between, emitter above. Fit quality is reported per class: on a pillar map most substrate spectra
# meet any residual limit trivially, so map-wide fractions hide how well the emitters themselves are fitted.
SNR_CLASSES = (20.0, 300.0)
CLASS_NAMES = ("substrate", "weak", "emitter")


def peak_snr(res):
    """Peak height above the edge baseline (unmasked pixels) in units of the noise sigma, and the height in counts."""
    from . import model
    snr, height = [], []
    for s in range(len(res.ids)):
        x, y, k = np.asarray(res.xs[s]), np.asarray(res.ys[s]), res.keeps[s]
        b0, b1 = model._edge_baseline(x[k], y[k]); t = (x[k] - model.X_REF) / model.X_SCALE
        h = float((y[k] - (b0 + b1 * t)).max()); height.append(h); snr.append(h / res.sigma[s])
    return np.array(snr), np.array(height)


def classify(snr, bounds=SNR_CLASSES):
    return np.digitize(snr, bounds)          # 0 substrate, 1 weak, 2 emitter


def residual_limit(y, floor=RESIDUAL_LIMIT["floor"], k_sqrt=RESIDUAL_LIMIT["k_sqrt"]):
    return np.maximum(floor, k_sqrt * np.sqrt(np.maximum(y, 0.0)))


def residual_check(res, floor=RESIDUAL_LIMIT["floor"], k_sqrt=RESIDUAL_LIMIT["k_sqrt"]):
    """Per spectrum, over fitted (unmasked) pixels: max |residual|, worst |residual| / limit, and whether every pixel
    is within the limit."""
    mx, ratio = [], []
    for s in range(len(res.ids)):
        k = res.keeps[s]; y = np.asarray(res.ys[s]); r = np.abs(y - res.curve(s))[k]
        mx.append(float(r.max())); ratio.append(float((r / residual_limit(y[k], floor, k_sqrt)).max()))
    mx, ratio = np.array(mx), np.array(ratio)
    return mx, ratio, ratio <= 1.0


def implied_huang_rhys(zpl_fraction):
    """S = -ln(ZPL fraction): the ZPL fraction is the Debye-Waller factor, which equals exp(-S) for linear coupling to a
    single phonon mode at low temperature. A model-dependent estimate: the Gaussian sideband's area is approximate, and an
    unresolved neighbouring emitter inside the sideband would bias it upward. Not used in any decision."""
    if zpl_fraction is None or not np.isfinite(zpl_fraction) or zpl_fraction <= 0 or zpl_fraction >= 1:
        return None
    return round(float(-np.log(zpl_fraction)), 3)


def ranking(res, phys):
    """Per emitter: centre, median nu/FWHM, SNR, sideband, and the strong / isolated / sharp / g2-priority flags."""
    tab, share, K = res.table, res.share, res.K
    step = float(np.median(np.diff(res.xs[0])))
    snr = res.heights / res.sigma[:, None]
    cen = np.array([res.median(0, k) for k in range(K)])
    wall = [float(np.median(tab[:, j, 2])) for j in range(K)]
    rows = []
    for k in range(K):
        m = share[:, k] >= RANK["share_for_nu"]
        nu = float(np.median(tab[m, k, 1])) if m.any() else float("nan")
        w = float(np.median(tab[m, k, 2])) if m.any() else wall[k]
        inst = physics.instrument_fwhm_nm(cen[k], phys)
        zf = float(np.median(tab[m, k, 7])) if (m.any() and res.psb[k]) else 1.0
        isolated = all(abs(cen[j] - cen[k]) >= RANK["isolated_x_fwhm"] * 0.5 * (w + wall[j]) for j in range(K) if j != k)
        r = dict(emitter=k + 1, center_nm=round(float(cen[k]), 2), nu_median=round(nu, 3), fwhm_nm=round(w, 2),
                 fwhm_over_instrument=round(w / inst, 2), max_snr=round(float(snr[:, k].max()), 1),
                 spectra_with_share_ge_10pct=int(m.sum()), phonon_sideband=bool(res.psb[k]), zpl_fraction=round(zf, 3),
                 huang_rhys_implied=implied_huang_rhys(zf) if res.psb[k] else None,
                 strong=bool(snr[:, k].max() >= RANK["strong_snr"]), isolated=bool(isolated),
                 sharp=bool(w <= RANK["sharp_x_instrument"] * inst), nu_reliable=bool(w >= RANK["sampling_px"] * step))
        r["kind"] = "broad" if w > RANK["broad_x_instrument"] * inst else "emitter line"
        r["g2_priority"] = r["strong"] and r["isolated"] and r["sharp"] and r["kind"] == "emitter line"
        rows.append(r)
    return rows


def unexplained_peaks(res, phys=None):
    """Fit audit, per spectrum: resolved peaks of the data (prominence >= 5 x noise) that the fit leaves unexplained (fit
    >= 3 x noise too low there and no fitted component centred within half an instrument FWHM). Returns, per spectrum,
    a list of (centre_nm, height above the fit)."""
    from . import basis3
    phys = phys or physics.load()
    inst = lambda c: physics.instrument_fwhm_nm(c, phys)
    return [[(round(c, 2), round(h, 1)) for c, h, _ in basis3._data_peaks(np.asarray(res.xs[s]), np.asarray(res.ys[s]), res.keeps[s],
                                                                       res.params[s], res.psb, res.sigma[s], inst, basis3.LOCAL)]
            for s in range(len(res.ids))]


def per_spectrum(res, limit=None, snr_classes=None, phys=None):
    """Per-spectrum checks shared by quality() and save(), computed once: residual limit (max |r|, ratio, pass), peak
    SNR, peak height, spectrum class and clipped-pixel count."""
    limit = tuple(limit or (RESIDUAL_LIMIT["floor"], RESIDUAL_LIMIT["k_sqrt"]))
    bounds = tuple(snr_classes or SNR_CLASSES)
    mx, ratio, rok = residual_check(res, *limit)
    psnr, height = peak_snr(res)
    clipped = np.array([0 if res.method == "legacy" else int((~B.clip_keep(y)).sum()) for y in res.ys])
    return dict(limit=limit, bounds=bounds, mx=mx, ratio=ratio, rok=rok, psnr=psnr, height=height,
                cls=classify(psnr, bounds), clipped=clipped, unexplained=unexplained_peaks(res, phys))


def _audit(res, ps):
    un = ps["unexplained"]
    pos = [c for u in un for c, _ in u]
    h, e = np.histogram(pos, bins=np.arange(np.floor(min(pos) / 10) * 10, np.ceil(max(pos) / 10) * 10 + 10, 10)) if pos else ([], [])
    worst = sorted(((h_, res.ids[s], c) for s, u in enumerate(un) for c, h_ in u), reverse=True)[:12]
    return dict(rule="resolved data peak (prominence >= 5 noise) left unexplained by the fit (fit >= 3 noise too low, no component "
                     "centred within half an instrument FWHM); noise = sqrt(sigma^2 + y)",
                n_spectra=int(sum(1 for u in un if u)), n_total=len(un),
                bands=[dict(from_nm=float(e[i]), to_nm=float(e[i + 1]), spectra=int(h[i])) for i in np.argsort(-np.asarray(h))[:6] if h[i]],
                worst=[dict(spectrum_id=sid, center_nm=c, height=hh) for hh, sid, c in worst])


def quality(res, limit=None, snr_classes=None, ps=None, phys=None):
    """Residual pass rate (RMS <= max(2.5 % peak, 2 sigma), max <= max(12 % peak, 8 sigma)), reduced chi^2, and the
    per-pixel residual limit check (see residual_check); limit = (floor, k_sqrt), default RESIDUAL_LIMIT.
    ps: per_spectrum() results, if already computed."""
    ok, rch = [], []
    for s in range(len(res.ids)):
        y, k = res.ys[s], res.keeps[s]
        r = (y - res.curve(s))[k]; yk = y[k]; n = r.size; sg = res.sigma[s]
        pk = float(yk.max() - np.median(yk[:max(5, n // 12)]))
        ok.append(bool(np.sqrt(np.mean(r ** 2)) <= max(PASS["rms_peak"] * pk, PASS["rms_sigma"] * sg) and
                       np.abs(r).max() <= max(PASS["max_peak"] * pk, PASS["max_sigma"] * sg)))
        rch.append(float(r @ r / max(n - res.params.shape[1], 1) / sg ** 2))
    ps = ps or per_spectrum(res, limit, snr_classes, phys)
    (floor, k_sqrt), bounds = ps["limit"], ps["bounds"]
    mx, ratio, rok, cls = ps["mx"], ps["ratio"], ps["rok"], ps["cls"]; okv = np.array(ok)
    by_class = {}
    for c, nm in enumerate(CLASS_NAMES):
        m = cls == c
        by_class[nm] = dict(n=int(m.sum()), pass_rate=float(okv[m].mean()) if m.any() else None,
                            residual_limit_pass=float(rok[m].mean()) if m.any() else None,
                            max_abs_residual_median=float(np.median(mx[m])) if m.any() else None)
    worst = np.argsort(-ratio)[:10]
    return dict(K=res.K, n_sidebands=int(np.sum(res.psb)), pass_rate=float(np.mean(ok)), rchi2_median=float(np.median(rch)),
                residual_limit=dict(rule=f"|r| <= max({floor:g}, {k_sqrt:g} * sqrt(y)) at every fitted pixel", floor=floor,
                                    k_sqrt=k_sqrt, pass_fraction=float(np.mean(rok)), n_fail=int((~rok).sum()),
                                    max_abs_residual_median=float(np.median(mx)), max_abs_residual_max=float(mx.max()),
                                    worst=[dict(spectrum_id=res.ids[i], ratio=round(float(ratio[i]), 3),
                                                max_abs_residual=round(float(mx[i]), 1)) for i in worst if not rok[i]]),
                clipped_spectra=int(np.sum(ps["clipped"] > 0)),
                audit=_audit(res, ps),
                snr_classes=dict(bounds=list(bounds), rule="peak SNR < %g substrate, < %g weak, else emitter" % bounds,
                                 by_class=by_class))


def audit_markdown(q):
    a = q.get("audit")
    if not a:
        return ""
    L = [f"\nFit audit: {a['n_spectra']} of {a['n_total']} spectra still show a resolved peak the fit does not explain "
         f"({a['rule']})."]
    if a["n_spectra"]:
        L.append("Where: " + ", ".join(f"{b['from_nm']:.0f}-{b['to_nm']:.0f} nm ({b['spectra']})" for b in a["bands"]) + ".")
        L.append("Inspect first: " + ", ".join(f"{w['spectrum_id']} at {w['center_nm']} nm (+{w['height']:.0f})" for w in a["worst"]) + ".")
        L.append("Per spectrum: column unexplained_peaks_nm of spectra.tsv. These fits are least reliable; at 150 g/mm they are "
                 "typically crowded lines or sharp lines on band-edge emission.")
    return "\n".join(L) + "\n"


def residual_markdown(q, method):
    r = q["residual_limit"]
    L = [f"\nResidual limit {r['rule']}: {r['pass_fraction']:.0%} of spectra pass ({r['n_fail']} fail); "
         f"max |r| median {r['max_abs_residual_median']:.0f}, worst {r['max_abs_residual_max']:.0f}."]
    sc = q["snr_classes"]
    L.append(f"By spectrum class ({sc['rule']}):")
    L.append("")
    L.append("| class | spectra | fit pass rate | residual limit met | max \\|r\\| median |")
    L.append("|---|---|---|---|---|")
    for nm, v in sc["by_class"].items():
        f = lambda z, p=True: "–" if z is None else (f"{z:.0%}" if p else f"{z:.0f}")
        L.append(f"| {nm} | {v['n']} | {f(v['pass_rate'])} | {f(v['residual_limit_pass'])} | {f(v['max_abs_residual_median'], False)} |")
    L.append("")
    if q["clipped_spectra"] and method != "legacy":
        L.append(f"{q['clipped_spectra']} spectra have a clipped (flat-topped) peak; those pixels are masked, not fitted.")
    if r["worst"]:
        L.append("Worst spectra (|r| / limit): " + ", ".join(f"{w['spectrum_id']} ({w['ratio']})" for w in r["worst"]) + ".")
    return "\n".join(L) + "\n"


def ranking_markdown(name, res, rows, phys):
    yn = lambda v: "yes" if v else "–"
    L = [f"# {name}: {res.method} emitters (K = {res.K}, {int(np.sum(res.psb))} with phonon sideband)", "",
         f"Instrument resolution {phys['instrument_resolution_meV']} meV. g2 priority = strong (>= {RANK['strong_snr']:.0f} sigma) "
         f"AND isolated (no emitter within {RANK['isolated_x_fwhm']} x FWHM) AND sharp (FWHM <= {RANK['sharp_x_instrument']:.0f} x "
         "instrument FWHM). nu and the sideband ZPL fraction are informative only; confirmation needs g2(0) < 0.5 or a "
         "high-resolution spectrum. Implied S = -ln(ZPL fraction) is the Huang-Rhys factor for single-mode linear coupling "
         "at low temperature: a model-dependent estimate (Gaussian sideband area; an unresolved neighbour would raise it). "
         f"type 'broad': FWHM > {RANK['broad_x_instrument']:g} x instrument, e.g. GaSe band-edge emission or an unresolved ensemble; fitted, "
         "never a g2 candidate.", "",
         "| # | ZPL centre (nm) | type | median ν | FWHM (nm) | FWHM / instrument | max SNR | spectra ≥10 % | sideband (ZPL fraction) | implied S | strong | isolated | sharp | **g² priority** |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['emitter']} | {r['center_nm']} | {r['kind']} | {r['nu_median']} | {r['fwhm_nm']} | {r['fwhm_over_instrument']} | {r['max_snr']} | "
                 f"{r['spectra_with_share_ge_10pct']} | {('yes (' + str(r['zpl_fraction']) + ')') if r['phonon_sideband'] else '–'} | "
                 f"{r['huang_rhys_implied'] if r['huang_rhys_implied'] is not None else '–'} | "
                 f"{yn(r['strong'])} | {yn(r['isolated'])} | {yn(r['sharp'])} | **{'YES' if r['g2_priority'] else 'no'}** |")
    return "\n".join(L) + "\n"


def figure(path, sid, x, y, keep, p, psb, formats=("tif",), limit=None, kinds=None, min_share=0.01):
    """Fit figure: data, fit, background, and the components present in this spectrum (>= min_share of its line area):
    each ZPL (solid, labelled #k, fitted centre and kind) and its phonon sideband (dashed); residuals at kept pixels.
    Components of the map basis with (near) zero amplitude here are not drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tot, bg, zpl, sb = emitter.evaluate(p, x, psb, parts=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(x, y, color="#0072bd", lw=1, label="data"); ax[0].plot(x, tot, color="#d95319", lw=1.2, label="fit")
    ax[0].plot(x, bg, color="#edb120", lw=1, label="background")
    tab = emitter.table(p, psb); area = tab[:, 3] + tab[:, 4]
    present = area >= min_share * max(float(area.sum()), 1e-300)
    for k in np.flatnonzero(present):
        kind = f", {kinds[k]}" if kinds else ""
        line, = ax[0].plot(x, bg + zpl[k], lw=0.9, label=f"#{k + 1} {tab[k, 0]:.1f} nm{kind}")
        if psb[k] and tab[k, 4] > 0:
            ax[0].plot(x, bg + sb[k], lw=0.9, ls="--", color=line.get_color(), label=f"#{k + 1} sideband")
    ax[0].set_xlim(x[0], x[-1]); ax[0].set_xlabel("Wavelength (nm)"); ax[0].set_ylabel("Intensity")
    ax[0].set_title(f"{sid}.txt  (dashed: phonon sidebands)")
    if (~keep).any():
        ax[0].plot(x[~keep], y[~keep], "x", color="k", ms=4, label="masked")
    ax[0].legend(fontsize=6, ncol=2)                     # after every labelled artist, including the masked points
    r = (tot - y)[keep]
    ax[1].plot(x[keep], r, "o", ms=2.5, mfc="none", color="b"); ax[1].axhline(0, color="k", lw=0.8)
    if limit:
        lim = residual_limit(y, *limit)
        ax[1].plot(x, lim, "--", color="g", lw=0.8); ax[1].plot(x, -lim, "--", color="g", lw=0.8,
                                                               label=f"limit max({limit[0]:g}, {limit[1]:g}√y)")
        ax[1].legend(fontsize=7)
    ax[1].set_title(f"Residual Plot. Resnorm = {float(r @ r):.6g}"); ax[1].set_xlabel("Wavelength (nm)"); ax[1].set_ylabel("Residuals")
    fig.tight_layout()
    for f in formats:
        fig.savefig(Path(path).with_suffix("." + f), dpi=100)
    plt.close(fig)


def _figure_task(args):
    out, items, formats, limit, kinds = args
    for sid, x, y, k, p, psb in items:
        figure(Path(out) / f"dc_{sid}", sid, x, y, k, p, psb, formats, limit, kinds)
    return len(items)


def maps(path, res, rows, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = min(4, res.K); rws = int(np.ceil(res.K / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(4 * cols, 3.6 * rws), squeeze=False)
    for k in range(rws * cols):
        ax = axs[k // cols, k % cols]
        if k >= res.K:
            ax.axis("off"); continue
        im = ax.imshow(G.image(res.table[:, k, 3], grid), cmap="jet", aspect="auto")
        r = rows[k]
        ax.set_title(f"#{k + 1} {r['center_nm']} nm, ν {r['nu_median']}, FWHM {r['fwhm_nm']}" + ("  g²?" if r["g2_priority"] else ""),
                     fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def feature_maps(path, res, feats, grid):
    """Per residual feature: where it is detected, next to the map of its parent emitter (if any)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(feats), 2, figsize=(8, 3.4 * len(feats)), squeeze=False)
    for i, f in enumerate(feats):
        det = np.zeros(len(res.ids)); det[[res.ids.index(s) for s in f["spectrum_ids"]]] = 1
        axs[i, 0].imshow(G.image(det, grid), cmap="Greys", vmin=0, vmax=1, aspect="auto")
        axs[i, 0].set_title(f"{f['center_nm']} nm bump: {f['classification']}", fontsize=8)
        if "parent_emitter" in f:
            im = axs[i, 1].imshow(G.image(res.table[:, f["parent_emitter"] - 1, 3], grid), cmap="viridis", aspect="auto")
            axs[i, 1].set_title(f"parent #{f['parent_emitter']} {f['parent_center_nm']} nm ZPL area "
                                f"(offset {f['offset_meV']} meV)", fontsize=8)
            fig.colorbar(im, ax=axs[i, 1])
        else:
            axs[i, 1].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def snr_map(path, snr, cls, grid, bounds):
    """Peak-SNR map (log scale) and the substrate / weak / emitter classes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    im = ax[0].imshow(G.image(np.log10(np.maximum(snr, 1.0)), grid), cmap="viridis", aspect="auto")
    ax[0].set_title("log10 peak SNR"); fig.colorbar(im, ax=ax[0])
    im = ax[1].imshow(G.image(cls, grid), cmap=ListedColormap(["#bbbbbb", "#e5a50a", "#c01c28"]), vmin=-0.5, vmax=2.5, aspect="auto")
    ax[1].set_title(f"class: substrate < {bounds[0]:g} ≤ weak < {bounds[1]:g} ≤ emitter")
    cb = fig.colorbar(im, ax=ax[1], ticks=[0, 1, 2]); cb.ax.set_yticklabels(CLASS_NAMES)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def save(res, out, name=None, phys=None, figures=False, formats=("tif",), grid=None, threads=1, residual_limit=None,
         snr_classes=None, residual_features=True):
    """Write every output for a fitted map into `out` (see README, 'Outputs'). residual_limit = (floor, k_sqrt)."""
    phys = phys or physics.load()
    limit = tuple(residual_limit or (RESIDUAL_LIMIT["floor"], RESIDUAL_LIMIT["k_sqrt"]))
    if isinstance(grid, (tuple, list)):
        grid = G.fixed(*grid)
    if isinstance(grid, dict) and not grid.get("rows"):
        grid = None
    ps = per_spectrum(res, limit, snr_classes, phys)       # the run's physics (instrument width) for the fit audit
    bounds = ps["bounds"]
    mx, ratio, rok, psnr, height, cls = ps["mx"], ps["ratio"], ps["rok"], ps["psnr"], ps["height"], ps["cls"]
    out = Path(out); (out / "dc").mkdir(parents=True, exist_ok=True)
    name = name or out.name
    for s, sid in enumerate(res.ids):
        legacy.write_dc(out / "dc" / f"dc_{sid}.txt", res.table[s][:, [0, 1, 2, 3]])
    snr = res.heights / res.sigma[:, None]
    with open(out / "emitters.tsv", "w") as fh:
        fh.write("spectrum_id\temitter\tzpl_center_nm\tnu\tzpl_fwhm_nm\tzpl_area\tpsb_area\tpsb_offset_nm\tpsb_fwhm_nm\t"
                 "zpl_fraction\tzpl_peak_snr\thuang_rhys_implied\n")
        for s, sid in enumerate(res.ids):
            for k in range(res.K):
                hr = implied_huang_rhys(res.table[s, k, 7]) if res.psb[k] else None
                fh.write(f"{sid}\t{k + 1}\t" + "\t".join(f"{v:.10g}" for v in res.table[s, k]) + f"\t{snr[s, k]:.4g}\t"
                         + ("" if hr is None else f"{hr:.4g}") + "\n")
    with open(out / "spectra.tsv", "w") as fh:
        fh.write("spectrum_id\tsse\tbackground_700nm\tbackground_slope_per_nm\tmasked_pixels\tclipped_pixels\t"
                 "max_abs_residual\tresidual_limit_ratio\tresidual_ok\tpeak_counts\tpeak_snr\tspectrum_class\tunexplained_peaks_nm\n")
        for s, sid in enumerate(res.ids):
            fh.write(f"{sid}\t{res.sse[s]:.10g}\t{res.params[s, 0]:.10g}\t{res.params[s, 1] / 100.0:.10g}\t"
                     f"{int((~res.keeps[s]).sum())}\t{ps['clipped'][s]}\t"
                     f"{mx[s]:.6g}\t{ratio[s]:.4f}\t{int(rok[s])}\t{height[s]:.6g}\t{psnr[s]:.4g}\t{CLASS_NAMES[cls[s]]}\t"
                     + " ".join(f"{c:.2f}" for c, _ in ps["unexplained"][s]) + "\n")
    (out / "basis.json").write_text(json.dumps(dict(method=res.method, K=res.K, psb=[int(v) for v in res.psb],
                                                    centers=[round(res.median(0, k), 4) for k in range(res.K)],
                                                    **res.extra), indent=1, default=float) + "\n")
    rows = ranking(res, phys)
    if residual_features:                       # fitted lines that are optical-phonon replicas of a bluer line
        for b, t in RP.fitted_line_replicas(res, phys, kinds=[r["kind"] for r in rows]).items():
            rows[b]["kind"] = f"replica of #{t['emitter']}"; rows[b]["g2_priority"] = False
            rows[b]["replica"] = {k: t[k] for k in ("emitter", "emitter_center_nm", "offset_meV", "offset_iqr_meV",
                                                     "tracking_slope", "width_ratio", "area_ratio_median", "area_rank_corr")}
    shapes = LS.analyse(res, phys, kinds=[r["kind"] for r in rows])
    for r, v in zip(rows, shapes):
        r["line_shape"] = v
    # direction towards an ideal ZPL: among g2-priority emitters, the more Lorentzian (lower nu_intrinsic) comes first
    g2 = [r for r in rows if r["g2_priority"]]
    g2.sort(key=lambda r: (r["line_shape"] or {}).get("nu_intrinsic", 1.0))
    for i, r in enumerate(g2):
        r["g2_order"] = i + 1
    q = quality(res, ps=ps)
    feats = RP.analyse(res, phys, kinds=[r["kind"] for r in rows]) if residual_features else None
    if feats is not None:
        (out / "residual_features.json").write_text(json.dumps(dict(criteria=RP.TEST, min_snr=RP.MIN_SNR,
                                                                    min_spectra=RP.MIN_SPECTRA, features=feats),
                                                               indent=1, default=float) + "\n")
    (out / "emitter_report.json").write_text(json.dumps(dict(map=name, method=res.method, criteria=RANK, physics=phys,
                                                             fit_quality=q, emitters=rows), indent=1, default=float) + "\n")
    (out / "emitter_report.md").write_text(ranking_markdown(name, res, rows, phys) +
                                           f"\nFit quality: pass rate {q['pass_rate']:.2f}, reduced chi^2 median {q['rchi2_median']:.2f}.\n"
                                           + residual_markdown(q, res.method) + audit_markdown(q)
                                           + LS.markdown(shapes, rows)
                                           + (RP.markdown(feats) if feats is not None else ""))
    sl = SL.build(res, rows, feats, phys, grid)
    SL.save(out, sl, name)
    if grid:
        maps(out / "emitter_maps.png", res, rows, grid)
        snr_map(out / "snr_map.png", psnr, cls, grid, bounds)
        if feats:
            feature_maps(out / "residual_features.png", res, feats, grid)
    if figures:
        items = [(sid, res.xs[s], res.ys[s], res.keeps[s], res.params[s], list(res.psb)) for s, sid in enumerate(res.ids)]
        kinds = [r["kind"] for r in rows]
        if threads > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            chunks = [items[i::threads] for i in range(threads) if items[i::threads]]
            with ProcessPoolExecutor(len(chunks), mp_context=multiprocessing.get_context("spawn")) as ex:
                list(ex.map(_figure_task, [(str(out / "dc"), c, formats, limit, kinds) for c in chunks]))
        else:
            _figure_task((str(out / "dc"), items, formats, limit, kinds))
    return dict(ranking=rows, quality=q, shortlist=sl)
