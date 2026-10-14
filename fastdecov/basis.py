"""Data-driven discovery of the global spectral basis (the principal emission lines).

In the paper's NMF notation, a spectrum image is M = W x P, with P the principal spectral components
(emission line shapes) and W their per-pixel weights. The legacy workflow chose P by hand: 12 fixed
anchors for pillar 9, the ten Table S1 lines for pillar 6. This module learns it from the map:

1. census     per spectrum: mask cosmic rays, then add pseudo-Voigt lines greedily at the largest
              residual and refit all lines jointly (no anchors). A line is kept only if BIC improves.
2. clustering pool all census lines, weighted by their share of their spectrum's line area, and find
              the modes of the weighted kernel density of line centres. Each mode is a candidate
              basis line with a centre tolerance and width range from its members' spread.
3. selection  choose the number of lines K by 2-fold cross-validation over the map: candidates learnt
              on one half, held-out spectra of the other half fitted with the top-K basis, and K picked
              at the minimum held-out BIC.

Everything is driven by the data and by noise-relative thresholds; nothing is tuned per sample.
The defaults below were tuned on the pillar-9 training map only and are frozen for validation.
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import median_filter

from . import model

DEFAULTS = dict(
    k_census_max=16,         # lines per spectrum in the census
    census_snr=4.0,          # add a line only if the smoothed residual exceeds this many sigma ...
    census_rel=0.005,        # ... and this fraction of the spectrum's emission peak
    bic_gain=4.0,            # minimum BIC improvement to keep a census line
    emit_snr=20.0,           # spectra whose emission peak is below this many sigma are not used
    min_share=0.005,         # census lines below this share of their spectrum's line area are ignored
    kde_bw_nm=None,          # kernel bandwidth; default max(0.8 nm, 1.5 grid steps)
    tol_min_nm=0.6, tol_max_nm=3.0,
    k_max=20,
    cv_starts=(8.0, 3.0),    # width starts for the cross-validation fits
)


# --------------------------------------------------------------------------- preprocessing
def noise_sigma(y):
    d2 = np.diff(y, 2)
    return max(1.4826 * float(np.median(np.abs(d2 - np.median(d2)))) / np.sqrt(6.0), 1e-6)


COSMIC_SHOULDER = 0.1      # 2.5: a cosmic ray has no shoulders. Any real line is at least instrument-broadened, so with
#                            sampling step <= instrument FWHM its neighbouring pixels carry >= exp(-4 ln2) ~ 6 % (13 % at
#                            0.85 nm step and 1.0 nm FWHM) of its peak; a spike whose neighbours rise <= 10 % is not light.


def cosmic_keep(y, sigma, shoulder=COSMIC_SHOULDER):
    """True = keep. A run of <= 2 px far above a 7-px running median is a cosmic ray only if it is isolated: a cosmic ray
    is extremely intense with essentially no bandwidth, so the pixels just outside the run stay near the local baseline
    (10th percentile over +-15 px), at most `shoulder` x the run's height above it. A sharp emitter line or a line top has
    elevated neighbours and is kept. shoulder=None restores the 2.0-2.4 rule (every short hot run)."""
    from scipy.ndimage import percentile_filter
    med = median_filter(y, size=7, mode="nearest")
    hot = (y - med) > np.maximum(8.0 * sigma, 0.05 * max(float(med.max() - med.min()), 1.0))
    base = percentile_filter(y, 10, size=31, mode="nearest") if shoulder is not None else None
    keep = np.ones(y.size, bool)
    i = 0
    while i < y.size:
        if hot[i]:
            j = i
            while j < y.size and hot[j]:
                j += 1
            if j - i <= 2:
                if shoulder is None:
                    keep[i:j] = False
                else:
                    height = float(y[i:j].max() - base[i:j].max())
                    side = max(float(y[i - 1] - base[i - 1]) if i > 0 else 0.0, float(y[j] - base[j]) if j < y.size else 0.0)
                    if height > 0 and side <= shoulder * height:
                        keep[i:j] = False
            i = j
        else:
            i += 1
    return keep


def clip_keep(y, min_run=3, top=0.5):
    """True = keep. Clipped (saturated or capped) peaks: runs of >= `min_run` exactly equal values in the upper part of
    the spectrum (above median + `top` x (max - median)). Measured spectra never repeat a value over several pixels on
    a peak, so such a plateau is not data. The plateau and one pixel on each side (the clipping edge) are masked."""
    y = np.asarray(y)
    keep = np.ones(y.size, bool)
    med = float(np.median(y)); lo = med + top * (float(y.max()) - med)
    i = 0
    while i < y.size:
        j = i
        while j + 1 < y.size and y[j + 1] == y[i]:
            j += 1
        if j - i + 1 >= min_run and y[i] > lo:
            keep[max(i - 1, 0):min(j + 2, y.size)] = False
        i = j + 1
    return keep


def data_keep(y, sigma):
    """Pixels used by the v2/v3 fits: not a cosmic ray and not part of a clipped peak."""
    return cosmic_keep(y, sigma) & clip_keep(y)


def _fit_one(x, y, p0, lb, ub, nbg, K):
    from . import lm_native
    P, S, _, _ = lm_native.fit_problems_k(x[None], y[None], np.zeros(1, int), p0[None], lb[None], ub[None],
                                          K, nbg, threads=1)
    return P[0], float(S[0])


# --------------------------------------------------------------------------- 1. census
def census_spectrum(x, y, opts=None):
    o = {**DEFAULTS, **(opts or {})}
    sigma = noise_sigma(y)
    keep = data_keep(y, sigma)
    xm, ym = x[keep], y[keep]
    b0, b1 = model._edge_baseline(xm, ym)
    t = (xm - model.X_REF) / model.X_SCALE
    base = b0 + b1 * t
    peak = float((ym - base).max())
    out = dict(sigma=sigma, peak=peak, n_masked=int((~keep).sum()), n_clipped=int((~clip_keep(y)).sum()),
               lines=[], emitting=peak >= o["emit_snr"] * sigma)
    if not out["emitting"]:
        return out
    dx = float(np.median(np.diff(x)))
    wmin = max(0.8, 1.5 * dx)
    yr = max(float(ym.max() - ym.min()), 1.0)
    p = np.array([b0, b1]); lb = np.array([-yr, -2 * yr]); ub = np.array([float(ym.max()), 2 * yr])
    n = xm.size
    rss = float(np.sum((ym - base) ** 2)); bic = n * np.log(rss / n) + 2 * np.log(n)
    K = 0
    for _ in range(o["k_census_max"]):
        res = ym - model.evaluate_k(p, xm, K, 2)
        sm = np.convolve(res, np.ones(3) / 3, mode="same")
        i = int(np.argmax(sm))
        if sm[i] < max(o["census_snr"] * sigma, o["census_rel"] * peak):
            break
        half = sm[i] / 2
        l = i
        while l > 0 and sm[l] > half:
            l -= 1
        r = i
        while r < n - 1 and sm[r] > half:
            r += 1
        w0 = float(np.clip(xm[r] - xm[l], 2 * dx, 60.0))
        c = float(xm[i])
        q = np.r_[p, sm[i], c, 0.5, w0]
        ql = np.r_[lb, 0.0, c - max(2.0, 0.5 * w0), 0.0, wmin]
        qu = np.r_[ub, 3.0 * yr, c + max(2.0, 0.5 * w0), 1.0, 80.0]
        q, sse = _fit_one(xm, ym, q, ql, qu, 2, K + 1)
        bic_new = n * np.log(max(sse, 1e-300) / n) + (2 + 4 * (K + 1)) * np.log(n)
        if bic_new > bic - o["bic_gain"]:
            break
        p, lb, ub, bic, K = q, ql, qu, bic_new, K + 1
        # let earlier lines re-centre around their fitted positions on the next addition
        pk = p[2:].reshape(K, 4)
        lb[2:].reshape(K, 4)[:, 1] = pk[:, 1] - np.maximum(2.0, 0.5 * pk[:, 3])
        ub[2:].reshape(K, 4)[:, 1] = pk[:, 1] + np.maximum(2.0, 0.5 * pk[:, 3])
    if K:
        pk = p[2:].reshape(K, 4)
        ar = model.areas_k(p, K, 2)
        tot = max(float(ar.sum()), 1e-300)
        out["lines"] = [dict(center=float(pk[k, 1]), width=float(pk[k, 3]), nu=float(pk[k, 2]), height=float(pk[k, 0]),
                             area=float(ar[k]), share=float(ar[k] / tot)) for k in range(K)]
    return out


def _census_task(args):
    x, y, o = args
    os.environ["OMP_NUM_THREADS"] = "1"
    return census_spectrum(x, y, o)


def census(xs, ys, opts=None, workers=None):
    workers = workers or len(os.sched_getaffinity(0))
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        return list(ex.map(_census_task, [(x, y, opts) for x, y in zip(xs, ys)], chunksize=4))


# --------------------------------------------------------------------------- 2. clustering
def candidate_lines(cens, grid_step, opts=None):
    """Weighted-KDE modes of census line centres, ranked by mean area share per emitting spectrum."""
    o = {**DEFAULTS, **(opts or {})}
    lines = [(l["center"], l["width"], l["share"]) for c in cens if c["emitting"] for l in c["lines"] if l["share"] >= o["min_share"]]
    n_emit = max(sum(c["emitting"] for c in cens), 1)
    if not lines:
        return []
    C, Wd, Wt = map(np.array, zip(*lines))
    h = o["kde_bw_nm"] or max(0.8, 1.5 * grid_step)
    g = np.arange(C.min() - 3 * h, C.max() + 3 * h, 0.05)
    dens = np.zeros_like(g)
    for c, w in zip(C, Wt):
        dens += w * np.exp(-0.5 * ((g - c) / h) ** 2)
    peaks = [i for i in range(1, g.size - 1) if dens[i] >= dens[i - 1] and dens[i] > dens[i + 1] and dens[i] > 1e-6 * dens.max()]
    troughs = [0] + [int(np.argmin(dens[a:b])) + a for a, b in zip(peaks[:-1], peaks[1:])] + [g.size - 1]
    cands = []
    for k, pi in enumerate(peaks):
        lo, hi = g[troughs[k]], g[troughs[k + 1]]
        m = (C >= lo) & (C <= hi)
        if not m.any():
            continue
        w = Wt[m]; c = C[m]
        order = np.argsort(c, kind="stable"); cs = np.cumsum(w[order]) / w.sum()
        med = float(c[order][np.searchsorted(cs, 0.5)])
        q25, q75 = (float(c[order][np.searchsorted(cs, q)]) for q in (0.25, 0.75))
        tol = float(np.clip(1.5 * (q75 - q25), o["tol_min_nm"], o["tol_max_nm"]))
        ww = Wd[m]
        cands.append(dict(center=med, tol=tol, width_min=float(max(0.8, 1.5 * grid_step, 0.6 * np.percentile(ww, 10))),
                          width_max=float(min(100.0, 1.6 * np.percentile(ww, 90) + 2.0)),
                          support=float(w.sum() / n_emit), prevalence=float(m.sum() / n_emit), n_members=int(m.sum())))
    cands.sort(key=lambda d: -d["support"])
    return cands


def basis_from_candidates(cands, K, name=None):
    sel = sorted(cands[:K], key=lambda d: d["center"])
    c = np.array([d["center"] for d in sel]); tol = np.array([d["tol"] for d in sel])
    return model.Basis(c, c - tol, c + tol, width_min=[d["width_min"] for d in sel],
                       width_max=[d["width_max"] for d in sel], background="linear", height_rule="free",
                       name=name or f"discovered-{K}", meta=dict(candidates=sel))


# --------------------------------------------------------------------------- 3. selection
def heldout_bic(xs, ys, keep_masks, basis, opts=None, workers=None):
    """Sum over spectra of n*log(RSS/n) + p*log(n), fitting each spectrum's unmasked pixels."""
    from .pipeline import fit_spectra_basis
    o = {**DEFAULTS, **(opts or {})}
    xm = [x[k] for x, k in zip(xs, keep_masks)]; ym = [y[k] for y, k in zip(ys, keep_masks)]
    res = fit_spectra_basis(list(range(len(xs))), xm, ym, basis, workers=workers, starts=o["cv_starts"])
    n = np.array([x.size for x in xm], float)
    bic = n * np.log(np.maximum(res.sse, 1e-300) / n) + basis.npar * np.log(n)
    return float(bic.sum()), res


def discover_basis(xs, ys, opts=None, workers=None, verbose=True):
    """Returns (Basis, report). xs/ys: lists of 1-D arrays (one map; grids may differ in length)."""
    o = {**DEFAULTS, **(opts or {})}
    grid_step = float(np.median([np.median(np.diff(x)) for x in xs]))
    cens = census(xs, ys, o, workers)
    emit = np.array([c["emitting"] for c in cens])
    keeps = [data_keep(y, c["sigma"]) for y, c in zip(ys, cens)]
    idx = np.flatnonzero(emit)
    folds = [idx[0::2], idx[1::2]]                 # interleaved halves of the map
    cand_f = [candidate_lines([cens[i] for i in f], grid_step, o) for f in folds]
    kmax = min(o["k_max"], *(len(c) for c in cand_f)) if all(cand_f) else 0
    curve = []
    for K in range(1, kmax + 1):
        tot = 0.0
        for a, b in ((0, 1), (1, 0)):                # learn on fold a, score fold b
            bas = basis_from_candidates(cand_f[a], K)
            f = folds[b]
            tot += heldout_bic([xs[i] for i in f], [ys[i] for i in f], [keeps[i] for i in f], bas, o, workers)[0]
        curve.append((K, tot))
        if verbose:
            print(f"  K={K:2d}  held-out BIC {tot:14.1f}", flush=True)
    k_best = min(curve, key=lambda t: t[1])[0] if curve else 0
    cands = candidate_lines(cens, grid_step, o)
    basis = basis_from_candidates(cands, k_best, name=f"discovered-{k_best}")
    report = dict(grid_step_nm=grid_step, n_spectra=len(xs), n_emitting=int(emit.sum()),
                  census_lines_per_spectrum=float(np.mean([len(c["lines"]) for c in cens if c["emitting"]])) if emit.any() else 0.0,
                  cv_curve=[dict(K=k, heldout_bic=b) for k, b in curve], k_selected=k_best,
                  candidates=cands[:max(o["k_max"], k_best)], options={k: v for k, v in o.items()})
    basis.meta.update(report=dict(k_selected=k_best, n_emitting=int(emit.sum())))
    return basis, report


# --------------------------------------------------------------------------- 4. global growth (NMF-like)
GROW_DEFAULTS = dict(
    init_support=0.01,       # start from census candidates with mean area share >= 1 %
    max_lines=24,
    min_sep_nm=None,         # minimum spacing between basis centres; default max(1.5 nm, 2 grid steps)
    new_tol_nm=5.0,          # centre tolerance of a new line: lets one line follow strain shifts (chosen on training)
    new_width_max=60.0,
    accept_bic=10.0,         # total held-out BIC must drop by more than this ...
    spec_gain=10.0,          # ... and a line must improve per-spectrum BIC by > spec_gain ...
    min_prevalence=0.10,     # ... in at least this fraction of held-out spectra (a principal line, not a shape fix)
    min_line_share=0.02,     # a principal line must also carry >= this share of line area in >= min_prevalence of
                             # held-out spectra (NMF sense: a real emission component, not a residual-shape patch)
    tries=3,                 # residual peaks tried per growth round
    starts=(8.0, 3.0),
)


def _fit_set(xs, ys, keeps, basis, starts, workers):
    from .pipeline import fit_spectra_basis
    xm = [x[k] for x, k in zip(xs, keeps)]; ym = [y[k] for y, k in zip(ys, keeps)]
    res = fit_spectra_basis(list(range(len(xs))), xm, ym, basis, workers=workers, starts=starts)
    n = np.array([x.size for x in xm], float)
    per = n * np.log(np.maximum(res.sse, 1e-300) / n) + basis.npar * np.log(n)
    return res, per


def _residual_profile(xs, ys, keeps, sigmas, res):
    """Mean over spectra of the noise-normalised positive residual, on the common grid (NaN where masked)."""
    x0 = xs[0]
    acc = np.zeros(x0.size); cnt = np.zeros(x0.size)
    for s, (x, y, k, sg) in enumerate(zip(xs, ys, keeps, sigmas)):
        z = (y - model.evaluate_k(res.params[s], x, res.K, res.nbg)) / sg
        z = np.where(k, np.maximum(z, 0.0), 0.0)
        acc += np.interp(x0, x, z); cnt += np.interp(x0, x, k.astype(float))
    prof = acc / np.maximum(cnt, 1.0)
    return x0, np.convolve(prof, np.ones(3) / 3, mode="same")


def _with_line(basis, c, tol, wmin, wmax):
    cs = np.r_[basis.centers, c]; order = np.argsort(cs, kind="stable")
    lo = np.r_[basis.center_lo, c - tol][order]; hi = np.r_[basis.center_hi, c + tol][order]
    wl = np.r_[basis.width_min, wmin][order]; wh = np.r_[basis.width_max, wmax][order]
    return model.Basis(cs[order], lo, hi, wl, wh, basis.background, basis.height_rule, basis.name, dict(basis.meta))


def _without_line(basis, j):
    m = np.arange(basis.K) != j
    return model.Basis(basis.centers[m], basis.center_lo[m], basis.center_hi[m], basis.width_min[m], basis.width_max[m],
                       basis.background, basis.height_rule, basis.name, dict(basis.meta))


def grow_basis(xs, ys, opts=None, workers=None, verbose=True):
    """Global basis growth with 3 interleaved folds: propose on A, accept on B, report on C.

    Returns (Basis, report). All spectra must come from one map with a common grid.
    """
    o = {**DEFAULTS, **GROW_DEFAULTS, **(opts or {})}
    grid_step = float(np.median(np.diff(xs[0])))
    min_sep = o["min_sep_nm"] or max(1.5, 2 * grid_step)
    wmin = max(0.8, 1.5 * grid_step)
    cens = census(xs, ys, o, workers)
    emit = np.flatnonzero([c["emitting"] for c in cens])
    sig = [c["sigma"] for c in cens]
    keeps = [data_keep(y, s) for y, s in zip(ys, sig)]
    F = [emit[0::3], emit[1::3], emit[2::3]]
    sub = lambda f: ([xs[i] for i in f], [ys[i] for i in f], [keeps[i] for i in f], [sig[i] for i in f])
    A, Bf, C = sub(F[0]), sub(F[1]), sub(F[2])
    cands = [c for c in candidate_lines([cens[i] for i in F[0]], grid_step, o) if c["support"] >= o["init_support"]]
    basis = basis_from_candidates(cands, len(cands), name="grown")
    _, per_b = _fit_set(*Bf[:3], basis, o["starts"], workers)
    bic_b = float(per_b.sum())

    def better(per_new, per_old):
        """New basis wins: total held-out BIC drops and enough held-out spectra improve significantly."""
        d = per_new - per_old
        return d.sum() < -o["accept_bic"] and np.mean(d < -o["spec_gain"]) >= o["min_prevalence"]

    def principal(res, j):
        """Line j carries >= min_line_share of line area in >= min_prevalence of the fitted spectra."""
        ar = res.areas
        share = ar[:, j] / np.maximum(ar.sum(1), 1e-300)
        return np.mean(share >= o["min_line_share"]) >= o["min_prevalence"]

    history = [dict(step="init", K=basis.K, centers=basis.centers.round(2).tolist(), heldout_bic_B=bic_b)]
    if verbose:
        print(f"  init  K={basis.K:2d}  BIC_B {bic_b:12.1f}  {basis.centers.round(1).tolist()}", flush=True)
    while basis.K < o["max_lines"]:
        resA, _ = _fit_set(*A[:3], basis, o["starts"], workers)
        xg, prof = _residual_profile(*A, resA)
        order = [i for i in np.argsort(-prof, kind="stable") if 0 < i < xg.size - 1 and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]]
        accepted = False
        for i in order[: o["tries"]]:
            c = float(xg[i])
            if np.min(np.abs(basis.centers - c)) < min_sep:
                continue
            trial = _with_line(basis, c, o["new_tol_nm"], wmin, o["new_width_max"])
            res_t, per_t = _fit_set(*Bf[:3], trial, o["starts"], workers)
            j_new = int(np.argmin(np.abs(trial.centers - c)))
            if better(per_t, per_b) and principal(res_t, j_new):
                basis, per_b, accepted = trial, per_t, True
                bic_b = float(per_b.sum())
                history.append(dict(step="add", center=c, K=basis.K, heldout_bic_B=bic_b))
                if verbose:
                    print(f"  add {c:6.1f}  K={basis.K:2d}  BIC_B {bic_b:12.1f}", flush=True)
                break
        if not accepted:
            break
    # pruning: drop lines that are not principal in held-out fits, or whose removal costs no held-out BIC
    improved = True
    while improved and basis.K > 1:
        improved = False
        res_cur, _ = _fit_set(*Bf[:3], basis, o["starts"], workers)
        for j in range(basis.K):
            if not principal(res_cur, j):
                trial = _without_line(basis, j)
                _, per_t = _fit_set(*Bf[:3], trial, o["starts"], workers)
                if verbose:
                    print(f"  drop {basis.centers[j]:6.1f} (not principal)  K={trial.K:2d}", flush=True)
                history.append(dict(step="drop-not-principal", center=float(basis.centers[j]), K=trial.K))
                basis, per_b, bic_b, improved = trial, per_t, float(per_t.sum()), True
                break
        if improved:
            continue
        for j in range(basis.K):
            trial = _without_line(basis, j)
            _, per_t = _fit_set(*Bf[:3], trial, o["starts"], workers)
            # drop a line unless keeping it is "better" than removing it under the same principal-line rule
            if not better(per_b, per_t):
                bic_t = float(per_t.sum())
                if verbose:
                    print(f"  drop {basis.centers[j]:6.1f}  K={trial.K:2d}  BIC_B {bic_t:12.1f}", flush=True)
                history.append(dict(step="drop", center=float(basis.centers[j]), K=trial.K, heldout_bic_B=bic_t))
                basis, per_b, bic_b, improved = trial, per_t, bic_t, True
                break
    _, per_c = _fit_set(*C[:3], basis, o["starts"], workers)
    bic_c = float(per_c.sum())
    basis.name = f"grown-{basis.K}"
    report = dict(grid_step_nm=grid_step, n_spectra=len(xs), n_emitting=int(emit.size), folds=[len(f) for f in F],
                  history=history, heldout_bic_C=bic_c, k_selected=basis.K, options={k: v for k, v in o.items()})
    basis.meta.update(report=dict(k_selected=basis.K, heldout_bic_C=bic_c))
    return basis, report
