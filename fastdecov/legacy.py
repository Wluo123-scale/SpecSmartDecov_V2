"""Legacy dc_<id>.txt tables (writer.m format) and refinement of legacy fits.

writer.m writes, per spectrum, a header line and one row per component:
    n  centre  nu  width(FWHM)  area      (printf '%g\\t%g\\t%g\\t%g\\t%g\\n')
followed by a blank line, with CRLF line endings. Heights and background are not stored; they are rebuilt here
(height from area, background as the best constant) to restart a fit from a legacy result.
"""
from pathlib import Path

import numpy as np

from . import model

DC_HEADER = "Peak #\t\tPeak Center\t\tNu (0=Lorentz<Shape<1=Gauss)\tSigma (Width)\t\tArea Under Peak\r\n"
WINDOWS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
BOUND_HIT_NM = 2e-3


def read_dc(path):
    """-> (NP, 4) array [centre, nu, width, area]."""
    rows = [l.split() for l in Path(path).read_text().splitlines()[1:] if len(l.split()) >= 5]
    if len(rows) != model.NP:
        raise ValueError(f"{path}: expected {model.NP} components, found {len(rows)}")
    return np.array([[float(v) for v in r[1:5]] for r in rows])


def _g(v):
    """MATLAB %g formatting (6 significant digits)."""
    return f"{v:g}"


def write_dc(path, table):
    lines = [DC_HEADER]
    for k, (c, nu, s, a) in enumerate(table, start=1):
        lines.append(f"{k}\t{_g(c)}\t{_g(nu)}\t{_g(s)}\t{_g(a)}\r\n")
    lines.append("\r\n")
    Path(path).write_bytes("".join(lines).encode())   # CRLF, as MATLAB fopen(...,'wt') on Windows


def infer_window(table):
    """rdnpeak half-window used for a legacy fit, from centres sitting on a +-0.5 % bound."""
    c = table[:, 0]
    consistent, hits = [], set()
    for h in WINDOWS:
        x0 = model.initial_centers(h)
        lo, hi = x0 * (1 - model.CENTER_REL_BOUND), x0 * (1 + model.CENTER_REL_BOUND)
        if np.all(c >= lo - BOUND_HIT_NM) and np.all(c <= hi + BOUND_HIT_NM):
            consistent.append(h)
        if np.any(np.abs(c - lo) < BOUND_HIT_NM) or np.any(np.abs(c - hi) < BOUND_HIT_NM):
            hits.add(h)
    for h in sorted(hits):
        if h in consistent:
            return h
    if consistent:
        return consistent[0]
    return min(hits) if hits else model.RDNPEAK_HALF_WINDOW_NM


def params_from_table(table, x, y):
    """Full parameter vector from a legacy table (height from area, best constant background)."""
    c, nu, s, area = table.T
    sa = np.abs(s)
    f3 = nu * model.K_GAUSS / sa + (1 - nu) * model.INV_PI / sa
    p = np.zeros(model.NPAR)
    pk = p[1:].reshape(model.NP, 4)
    pk[:, 0], pk[:, 1], pk[:, 2], pk[:, 3] = 2 * area * f3, c, nu, s
    p[0] = np.clip(np.mean(y - model.evaluate(p, x)), 0, y.max())
    return p


def refine_problem(table, x, y):
    """Start = legacy solution; bounds = legacy box for the inferred window, widened just
    enough to contain the legacy solution (it must be feasible)."""
    h = infer_window(table)
    _, lb, ub = model.setup(x, y, half_window=h)
    p0 = params_from_table(table, x, y)
    lb = np.minimum(lb, p0)
    ub = np.maximum(ub, p0)
    ub[0] = max(ub[0], p0[0])
    return p0, lb, ub, h


# MATLAB default axes colour order (R2014b+), used by plotter.m's 'hold all'
MATLAB_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]


def _num2str(v):
    """MATLAB num2str for a scalar: max(digits before the point, 1) + 4 significant digits."""
    d = max(int(np.floor(np.log10(abs(v)))) + 1, 1) + 4 if v != 0 else 5
    return f"{v:.{d}g}"


def write_tif(path, name, x, y, p, formats=("tif",)):
    """Figure as plotter.m saves it: left = data, fit, background and peaks over background;
    right = residual (fit - data, as lsqcurvefit returns it). 875 x 656 px, 150 dpi, PackBits TIFF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fit = model.evaluate(p, x)
    comps = model.components(p, x)
    resid = fit - y
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(875 / 150, 656 / 150), dpi=150)
    a1.set_prop_cycle(color=MATLAB_COLORS)
    a1.plot(x, y, lw=0.6)
    a1.plot(x, fit, lw=0.6)
    a1.plot(x, np.full_like(x, p[0]), lw=0.6)
    for c in comps:
        a1.plot(x, p[0] + c, lw=0.6)
    a1.set_xlim(550, 850)
    a1.set_ylim(min(0.0, 0.0), y.max() * 1.1)
    a1.set_title(name, fontsize=8, fontweight="bold")
    a1.set_xlabel("Wavelength (nm)", fontsize=7); a1.set_ylabel("Intensity", fontsize=7)
    a2.plot(x, resid, "o", mfc="none", mec="b", ms=4, mew=0.5)
    a2.plot(x, np.zeros_like(x), "-k", lw=0.6)
    a2.set_xlim(550, 850)
    lo, hi = resid.min() * 1.1, resid.max() * 1.1
    if lo < hi:
        a2.set_ylim(lo, hi)
    a2.set_title(f"Residual Plot. Resnorm = {_num2str(float(resid @ resid))}", fontsize=8, fontweight="bold")
    a2.set_xlabel("Wavelength (nm)", fontsize=7); a2.set_ylabel("Residuals", fontsize=7)
    for ax in (a1, a2):
        ax.grid(True, lw=0.3, alpha=0.5)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    path = Path(path)
    for fmt in formats:
        if fmt == "tif":
            fig.savefig(path.with_suffix(".tif"), format="tiff", dpi=150, pil_kwargs={"compression": "packbits"})
        else:
            fig.savefig(path.with_suffix("." + fmt), format=fmt, dpi=150)
    plt.close(fig)


def _tif_task(args):
    write_tif(*args)


def write_tifs(dc_dir, ids, xs, ys, params, workers):
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(Path(dc_dir) / f"dc_{sid}.tif", f"{sid}.txt", x, y, p) for sid, x, y, p in zip(ids, xs, ys, params)]
    if workers <= 1:
        for t in tasks:
            _tif_task(t)
        return
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        list(ex.map(_tif_task, tasks, chunksize=8))
