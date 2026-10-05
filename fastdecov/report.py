"""Writers and the emitter ranking shared by the CLI and the SDK.

All methods are reported in the V3 emitter layout (emitter.py): V2 lines are emitters without a phonon sideband,
and legacy fits additionally have a zero background slope.
"""
import json
from pathlib import Path

import numpy as np

from . import emitter, legacy, physics

# Ranking for single-photon (g2) follow-up. nu and the sideband ZPL fraction are reported, not used as gates: at
# 150 g/mm the antibunched ~692 nm emitter of the reference study (g2(0) = 0.43) looks Gaussian-dominated, because the
# exciton/biexciton structure is unresolved and spectral diffusion adds Gaussian broadening.
RANK = dict(strong_snr=50.0, share_for_nu=0.10, sharp_x_instrument=3.0, isolated_x_fwhm=1.5, sampling_px=2.5)
PASS = dict(rms_peak=0.025, rms_sigma=2.0, max_peak=0.12, max_sigma=8.0)


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
                 strong=bool(snr[:, k].max() >= RANK["strong_snr"]), isolated=bool(isolated),
                 sharp=bool(w <= RANK["sharp_x_instrument"] * inst), nu_reliable=bool(w >= RANK["sampling_px"] * step))
        r["g2_priority"] = r["strong"] and r["isolated"] and r["sharp"]
        rows.append(r)
    return rows


def quality(res):
    """Residual pass rate (RMS <= max(2.5 % peak, 2 sigma), max <= max(12 % peak, 8 sigma)) and reduced chi^2."""
    ok, rch = [], []
    for s in range(len(res.ids)):
        x, y, k = res.xs[s], res.ys[s], res.keeps[s]
        r = (y - res.curve(s))[k]; yk = y[k]; n = r.size; sg = res.sigma[s]
        pk = float(yk.max() - np.median(yk[:max(5, n // 12)]))
        ok.append(bool(np.sqrt(np.mean(r ** 2)) <= max(PASS["rms_peak"] * pk, PASS["rms_sigma"] * sg) and
                       np.abs(r).max() <= max(PASS["max_peak"] * pk, PASS["max_sigma"] * sg)))
        rch.append(float(r @ r / max(n - res.params.shape[1], 1) / sg ** 2))
    return dict(K=res.K, n_sidebands=int(np.sum(res.psb)), pass_rate=float(np.mean(ok)), rchi2_median=float(np.median(rch)))


def ranking_markdown(name, res, rows, phys):
    yn = lambda v: "yes" if v else "–"
    L = [f"# {name}: {res.method} emitters (K = {res.K}, {int(np.sum(res.psb))} with phonon sideband)", "",
         f"Instrument resolution {phys['instrument_resolution_meV']} meV. g2 priority = strong (>= {RANK['strong_snr']:.0f} sigma) "
         f"AND isolated (no emitter within {RANK['isolated_x_fwhm']} x FWHM) AND sharp (FWHM <= {RANK['sharp_x_instrument']:.0f} x "
         "instrument FWHM). nu and the sideband ZPL fraction are informative only; confirmation needs g2(0) < 0.5 or a "
         "high-resolution spectrum.", "",
         "| # | ZPL centre (nm) | median ν | FWHM (nm) | FWHM / instrument | max SNR | spectra ≥10 % | sideband (ZPL fraction) | strong | isolated | sharp | **g² priority** |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['emitter']} | {r['center_nm']} | {r['nu_median']} | {r['fwhm_nm']} | {r['fwhm_over_instrument']} | {r['max_snr']} | "
                 f"{r['spectra_with_share_ge_10pct']} | {('yes (' + str(r['zpl_fraction']) + ')') if r['phonon_sideband'] else '–'} | "
                 f"{yn(r['strong'])} | {yn(r['isolated'])} | {yn(r['sharp'])} | **{'YES' if r['g2_priority'] else 'no'}** |")
    return "\n".join(L) + "\n"


def figure(path, sid, x, y, keep, p, psb, formats=("tif",)):
    """Fit figure: data, fit, background, each ZPL (solid) and phonon sideband (dashed); residuals at kept pixels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tot, bg, zpl, sb = emitter.evaluate(p, x, psb, parts=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(x, y, color="#0072bd", lw=1, label="data"); ax[0].plot(x, tot, color="#d95319", lw=1.2, label="fit")
    ax[0].plot(x, bg, color="#edb120", lw=1, label="background")
    for k in range(len(psb)):
        line, = ax[0].plot(x, bg + zpl[k], lw=0.9, label=f"ZPL {k + 1}")
        if psb[k]:
            ax[0].plot(x, bg + sb[k], lw=0.9, ls="--", color=line.get_color())
    ax[0].set_xlim(x[0], x[-1]); ax[0].set_xlabel("Wavelength (nm)"); ax[0].set_ylabel("Intensity")
    ax[0].set_title(f"{sid}.txt  (dashed: phonon sidebands)"); ax[0].legend(fontsize=6, ncol=2)
    r = (tot - y)[keep]
    ax[1].plot(x[keep], r, "o", ms=2.5, mfc="none", color="b"); ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title(f"Residual Plot. Resnorm = {float(r @ r):.6g}"); ax[1].set_xlabel("Wavelength (nm)"); ax[1].set_ylabel("Residuals")
    fig.tight_layout()
    for f in formats:
        fig.savefig(Path(path).with_suffix("." + f), dpi=100)
    plt.close(fig)


def _figure_task(args):
    out, items, formats = args
    for sid, x, y, k, p, psb in items:
        figure(Path(out) / f"dc_{sid}", sid, x, y, k, p, psb, formats)
    return len(items)


def maps(path, res, rows, grid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = grid
    cols = min(4, res.K); rws = int(np.ceil(res.K / cols))
    fig, axs = plt.subplots(rws, cols, figsize=(4 * cols, 3.6 * rws), squeeze=False)
    for k in range(rws * cols):
        ax = axs[k // cols, k % cols]
        if k >= res.K:
            ax.axis("off"); continue
        im = ax.imshow(res.table[:, k, 3].reshape(nr, nc), cmap="jet", aspect="auto")
        r = rows[k]
        ax.set_title(f"#{k + 1} {r['center_nm']} nm, ν {r['nu_median']}, FWHM {r['fwhm_nm']}" + ("  g²?" if r["g2_priority"] else ""),
                     fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


def save(res, out, name=None, phys=None, figures=False, formats=("tif",), grid=None, threads=1):
    """Write every output for a fitted map into `out` (see README, 'Outputs')."""
    phys = phys or physics.load()
    out = Path(out); (out / "dc").mkdir(parents=True, exist_ok=True)
    name = name or out.name
    for s, sid in enumerate(res.ids):
        legacy.write_dc(out / "dc" / f"dc_{sid}.txt", res.table[s][:, [0, 1, 2, 3]])
    snr = res.heights / res.sigma[:, None]
    with open(out / "emitters.tsv", "w") as fh:
        fh.write("spectrum_id\temitter\tzpl_center_nm\tnu\tzpl_fwhm_nm\tzpl_area\tpsb_area\tpsb_offset_nm\tpsb_fwhm_nm\t"
                 "zpl_fraction\tzpl_peak_snr\n")
        for s, sid in enumerate(res.ids):
            for k in range(res.K):
                fh.write(f"{sid}\t{k + 1}\t" + "\t".join(f"{v:.10g}" for v in res.table[s, k]) + f"\t{snr[s, k]:.4g}\n")
    with open(out / "spectra.tsv", "w") as fh:
        fh.write("spectrum_id\tsse\tbackground_700nm\tbackground_slope_per_nm\tmasked_pixels\n")
        for s, sid in enumerate(res.ids):
            fh.write(f"{sid}\t{res.sse[s]:.10g}\t{res.params[s, 0]:.10g}\t{res.params[s, 1] / 100.0:.10g}\t"
                     f"{int((~res.keeps[s]).sum())}\n")
    (out / "basis.json").write_text(json.dumps(dict(method=res.method, K=res.K, psb=[int(v) for v in res.psb],
                                                    centers=[round(res.median(0, k), 4) for k in range(res.K)],
                                                    **res.extra), indent=1, default=float) + "\n")
    rows = ranking(res, phys); q = quality(res)
    (out / "emitter_report.json").write_text(json.dumps(dict(map=name, method=res.method, criteria=RANK, physics=phys,
                                                             fit_quality=q, emitters=rows), indent=1, default=float) + "\n")
    (out / "emitter_report.md").write_text(ranking_markdown(name, res, rows, phys) +
                                           f"\nFit quality: pass rate {q['pass_rate']:.2f}, reduced chi^2 median {q['rchi2_median']:.2f}.\n")
    if grid:
        maps(out / "emitter_maps.png", res, rows, grid)
    if figures:
        items = [(sid, res.xs[s], res.ys[s], res.keeps[s], res.params[s], list(res.psb)) for s, sid in enumerate(res.ids)]
        if threads > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            chunks = [items[i::threads] for i in range(threads) if items[i::threads]]
            with ProcessPoolExecutor(len(chunks), mp_context=multiprocessing.get_context("spawn")) as ex:
                list(ex.map(_figure_task, [(str(out / "dc"), c, formats) for c in chunks]))
        else:
            _figure_task((str(out / "dc"), items, formats))
    return dict(ranking=rows, quality=q)
