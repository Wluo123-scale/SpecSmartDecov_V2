# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""V4: a fixed set of emitters for the whole map, seeded from the brightest spectra.

Physical model: the map holds a fixed number K of emitters. Strain and quantum confinement change each emitter's colour
(centre, within +- strain_shift_halfrange_nm of its reference) and brightness (amplitude, free per spectrum, may be ~0)
from pixel to pixel, but never create or remove emitters. Broad bands are not a separate kind of component: the emitter
width is capped, so a band must be explained by the (unresolved-looking) emitters and exciton complexes it is made of.

1. seeds: spectra with peak SNR >= seed_snr (fallback: the brightest min_seeds spectra); their census lines (basis.census)
   are processed brightest first. A line within the strain window of an existing emitter that this spectrum has not
   used yet is that emitter; otherwise it founds a new emitter, if it is resolved (>= 0.85 x its FWHM) from every
   existing reference centre by >= min_sep_nm (default 2 x window: each emitter owns its colour window). Candidates that never share a spectrum and together span <= 2 x window are one emitter
   seen at both ends of its strain drift and are merged; the reference is the middle of the emitter's colour range.
2. support: an emitter needs lines (>= min_line_share of their spectrum's line area) in >= max(min_seed_spectra,
   min_seed_frac x seeds) seed spectra.
3. windows: centre_k in [ref_k - h_lo,k, ref_k + h_hi,k]. Each half-width starts from the colour range of the
   emitter's seed lines, clamped to [window_min_nm, window_max_nm] (default: the physics strain half-range .. 5 nm), and
   a side is widened by widen_step_nm (up to window_max_nm) while the fitted colour sits on that edge in > widen_edge_frac
   of the spectra where the emitter is present. Windows are clipped at the midpoint between neighbouring references, so
   two emitters can never swap identity; references are >= min_sep_nm apart. By default min_sep_nm is learned from the
   brightest spectra (learn_min_sep): the median spacing of neighbouring peaks and shoulders (maxima of -y'') within a
   band, in the top sep_top_frac of the map by peak SNR.
4. growth: resolved residual peaks of the seed fits outside every window, clustered brightest first, are tried as new
   emitters (a broad band that one capped emitter cannot fit leaves such residuals at its flanks).
5. necessity: an emitter is kept only if removing it worsens the per-spectrum BIC by > spec_gain in >= accept_frac of
   the seed spectra where it carries >= 2 % of the line area (and there are >= min_seed_spectra of them).
6. final fit: every spectrum of the map with the same K emitters (map-wide, no local patches).
"""
import numpy as np

from . import basis as B, basis3 as B3, physics

OPTS4 = dict(seed_snr=300.0, min_seeds=20, width_max_nm=None, min_seed_spectra=3, resolve=0.85, accept_frac=0.5,
             spec_gain=10.0, max_emitters=40, grow_tries=3, grow_rounds=6, edge_x_width=0.5, min_line_share=0.02,
             max_cooccur=0.05, min_sep_nm=None, min_seed_frac=0.01, window_min_nm=None, window_max_nm=5.0,
             window_margin_nm=0.5, widen_edge_frac=0.2, widen_step_nm=1.0, widen_rounds=3,
             sep_top_frac=0.10, sep_min_spectra=20, sep_smooth_x_width=0.5, sep_height_mad=3.0, sep_band_x_width=3.0,
             sep_min_pairs=10)


def components(x, y, smooth_nm, height_mad, edge_nm=5.0):
    """Peak and shoulder positions of one spectrum: maxima of -y'' (Savitzky-Golay, window smooth_nm) above
    height_mad x its robust spread. A shoulder of a broad band is a maximum of -y'' even where y has no maximum."""
    from scipy.signal import find_peaks, savgol_filter
    o = np.argsort(x); x, y = np.asarray(x)[o], np.asarray(y)[o]
    dx = float(np.median(np.diff(x))); w = max(5, int(round(smooth_nm / dx)) | 1)
    if w >= x.size:
        return np.array([])
    d2 = -savgol_filter(y, w, 3, deriv=2, delta=dx)
    mad = float(np.median(np.abs(d2 - np.median(d2)))) * 1.4826
    pk, _ = find_peaks(d2, height=height_mad * max(mad, 1e-300))
    c = x[pk]
    return c[(c > x[0] + edge_nm) & (c < x[-1] - edge_nm)]


def learn_min_sep(xs, ys, snr, wmax, o):
    """Typical spacing of neighbouring components (peaks and shoulders) within a band, in the brightest spectra.
    Spacings > sep_band_x_width x wmax are between bands and are left out. Returns (spacing or None, stats)."""
    top = max(o["sep_min_spectra"], int(np.ceil(o["sep_top_frac"] * len(xs))))
    idx = np.argsort(-snr)[:min(top, len(xs))]
    seps, per = [], []
    for i in idx:
        c = np.sort(components(xs[i], ys[i], o["sep_smooth_x_width"] * wmax, o["sep_height_mad"]))
        per.append(int(c.size)); seps += list(np.diff(c))
    seps = np.asarray(seps); within = seps[seps <= o["sep_band_x_width"] * wmax]
    stats = dict(spectra=int(idx.size), min_snr=round(float(snr[idx].min()), 1), components_per_spectrum=float(np.median(per) if per else 0),
                 pairs=int(seps.size), within_band_pairs=int(within.size))
    if within.size < o["sep_min_pairs"]:
        return None, stats
    stats.update(p10=round(float(np.percentile(within, 10)), 2), p25=round(float(np.percentile(within, 25)), 2),
                 median=round(float(np.median(within)), 2), p75=round(float(np.percentile(within, 75)), 2))
    return float(np.median(within)), stats


class Windows:
    """Emitter references (sorted) with their own unclipped half-widths on each side; basis() clips the windows at the
    midpoints between neighbouring references, so removing an emitter lets its neighbours' windows open up again."""

    def __init__(self, refs, hlo, hhi):
        refs = np.asarray(refs, float); o = np.argsort(refs, kind="stable")
        self.refs = refs[o]; self.hlo = np.asarray(hlo, float)[o]; self.hhi = np.asarray(hhi, float)[o]

    @property
    def K(self):
        return self.refs.size

    def bounds(self):
        lo, hi = self.refs - self.hlo, self.refs + self.hhi
        mid = 0.5 * (self.refs[1:] + self.refs[:-1])
        lo[1:] = np.maximum(lo[1:], mid); hi[:-1] = np.minimum(hi[:-1], mid)
        return lo, hi

    def basis(self, wmin, wmax):
        lo, hi = self.bounds()
        return B3.EmitterBasis(self.refs, lo, hi, wmin, wmax)

    def without(self, k):
        m = np.arange(self.K) != k
        return Windows(self.refs[m], self.hlo[m], self.hhi[m])

    def with_emitter(self, c, h):
        return Windows(np.r_[self.refs, c], np.r_[self.hlo, h], np.r_[self.hhi, h])

    def widened(self, k, side, step, hmax):
        w = Windows(self.refs, self.hlo.copy(), self.hhi.copy())
        arr = w.hlo if side == "lo" else w.hhi
        arr[k] = min(arr[k] + step, hmax)
        return w


def seed_emitters(cens, seeds, x_range, half, wmax, o):
    """Brightest-first assignment of the seed spectra's census lines to emitters.
    Returns [(ref, n_spectra, height_sum, half_width)]; `half` is the largest allowed colour half-range.

    The brightest line founds an emitter; a dimmer line within `half` of an emitter not yet used by its spectrum joins it;
    a line that is neither founds a new emitter only if it is >= min_sep_nm from every reference. The founding line may
    sit at one end of the emitter's strain drift, so two candidates are then merged when they are one emitter: they
    (almost) never occur in the same spectrum (an emitter appears once per spectrum) and their combined colour range fits
    in the window (p5-p95 spread <= 2 x half). The reference is the middle of that colour range, and the half-width its
    half-spread + window_margin_nm, clamped to [window_min_nm, half]."""
    lo_x, hi_x = x_range; edge = o["edge_x_width"] * wmax
    hmin = min(o.get("window_min_nm") or half, half)
    lines = [(l["height"], l["center"], l["width"], s) for s in seeds for l in cens[s]["lines"]
             if l["share"] >= o["min_line_share"] and lo_x + edge <= l["center"] <= hi_x - edge]
    lines.sort(key=lambda t: -t[0])
    sep = o["min_sep_nm"] or 2 * half
    em = []                                   # dict(ref, members, centers, height)
    for h, c, w, s in lines:
        near = [e for e in em if abs(c - e["ref"]) <= half]
        free = [e for e in near if s not in e["members"]]
        if free:
            e = min(free, key=lambda e: abs(c - e["ref"]))
            e["members"].add(s); e["centers"].append(c); e["height"] += h
            continue
        # A candidate between one and two colour half-windows from a reference
        # can be the far end of that same strained emitter.  Retain it
        # provisionally so the co-occurrence/span merge below can decide that
        # case; requiring it to already clear min_sep loses those endpoints when
        # min_sep is larger than their distance from the initial (brightest)
        # endpoint.  Beyond the mergeable 2*half span, preserve min_sep.
        d = [abs(c - e["ref"]) for e in em]
        provisional = any(half < z <= 2 * half for z in d)
        if provisional or all(z >= sep for z in d):
            em.append(dict(ref=c, members={s}, centers=[c], height=h))
    span = lambda cs: (float(np.percentile(cs, 5)), float(np.percentile(cs, 95)))
    merged = True
    while merged:
        merged = False
        em.sort(key=lambda e: e["ref"])
        for a, b in zip(em, em[1:]):
            if b["ref"] - a["ref"] > 2 * half:
                continue
            both = len(a["members"] & b["members"])
            if both > o["max_cooccur"] * min(len(a["members"]), len(b["members"])):
                continue                      # resolved in the same spectra: two emitters
            lo, hi = span(a["centers"] + b["centers"])
            if hi - lo > 2 * half:
                continue
            a["members"] |= b["members"]; a["centers"] += b["centers"]; a["height"] += b["height"]
            a["ref"] = 0.5 * (lo + hi); em.remove(b); merged = True
            break
    out = []
    for e in em:
        lo, hi = span(e["centers"]); e["ref"] = 0.5 * (lo + hi)
        hw = float(np.clip(0.5 * (hi - lo) + o["window_margin_nm"], hmin, half))
        out.append((e["ref"], len(e["members"]), e["height"], hw))
    return out


def _necessary(xm, ym, W, fit, k, o, wmin, wmax):
    """Is emitter k needed? Refit, without it (neighbours' windows may open up), the spectra where it carries >= 2 %
    of the line area."""
    core = np.flatnonzero(fit.share[:, k] >= 0.02)
    if core.size < o["min_seed_spectra"]:
        return False, core.size, 0.0
    trial = W.without(k).basis(wmin, wmax)
    fT = B3.fit_set([xm[i] for i in core], [ym[i] for i in core], trial, o["starts"], o, o["_workers"])
    n = fit.n[core]
    d = B3._bic(fT.sse, n, 0) - B3._bic(fit.sse[core], n, 4)          # > 0: worse without emitter k
    need = float(np.mean(d > o["spec_gain"]))
    return need >= o["accept_frac"], core.size, need


def _prune(xm, ym, W, fit, o, hist, wmin, wmax):
    while W.K > 1:
        order = np.argsort(fit.tab[..., 3].sum(0))                      # weakest total ZPL area first
        dropped = False
        for k in order:
            ok, n_core, need = _necessary(xm, ym, W, fit, k, o, wmin, wmax)
            if not ok:
                hist.append(dict(step="drop", center=round(float(W.refs[k]), 2), core_spectra=int(n_core), need=round(need, 2)))
                W = W.without(k); fit = B3.fit_set(xm, ym, W.basis(wmin, wmax), o["starts"], o, o["_workers"]); dropped = True
                break
        if not dropped:
            break
    return W, fit


def edge_fractions(fit, b, k, tol=0.05):
    """Fraction of the spectra where emitter k carries >= 2 % of the line area whose fitted colour is on the low / high
    edge of its window."""
    m = fit.share[:, k] >= 0.02
    if not m.any():
        return 0.0, 0.0, 0
    c = fit.tab[m, k, 0]
    return float(np.mean(c - b.lo[k] < tol)), float(np.mean(b.hi[k] - c < tol)), int(m.sum())


def _widen(xm, ym, W, fit, o, hist, wmin, wmax, hmax):
    """Open a window side (by widen_step_nm, up to hmax) while the fitted colour piles up on it."""
    for rnd in range(o["widen_rounds"]):
        b = W.basis(wmin, wmax); W2 = W; moved = []
        for k in range(W.K):
            flo, fhi, n = edge_fractions(fit, b, k)
            if n < o["min_seed_spectra"]:
                continue
            for side, frac in (("lo", flo), ("hi", fhi)):
                if frac > o["widen_edge_frac"]:
                    W3 = W2.widened(k, side, o["widen_step_nm"], hmax)
                    lo3, hi3 = W3.bounds(); lo2, hi2 = W2.bounds()
                    if (lo3[k] < lo2[k] - 1e-9) if side == "lo" else (hi3[k] > hi2[k] + 1e-9):
                        W2 = W3; moved.append(f"{W.refs[k]:.1f}{'-' if side == 'lo' else '+'}({frac:.0%})")
        if not moved:
            break
        hist.append(dict(step="widen", round=rnd + 1, windows=moved))
        W = W2; fit = B3.fit_set(xm, ym, W.basis(wmin, wmax), o["starts"], o, o["_workers"])
    return W, fit


def _grow(xs_s, ys_s, keeps_s, sig_s, xm, ym, W, fit, o, phys, hist, wmin, wmax, half, hmin):
    L = {**B3.LOCAL, **(o.get("local_opts") or {})}
    inst = lambda c: physics.instrument_fwhm_nm(c, phys)
    for rnd in range(o["grow_rounds"]):
        if W.K >= o["max_emitters"]:
            break
        b = W.basis(wmin, wmax)
        cands = []
        for s in range(len(xs_s)):
            x, y = np.asarray(xs_s[s]), np.asarray(ys_s[s])
            lim = lambda yy, s=s: np.maximum(L["noise_k"] * sig_s[s], 3.0 * np.sqrt(np.maximum(yy, 0.0)))
            for c, h, w in B3._residual_peaks(x, y, keeps_s[s], fit.params[s], b.psb, lim, inst, {**o, "new_width_max": 3 * wmax}, L):
                if np.all(np.abs(c - W.refs) >= o["min_sep_nm"]):
                    cands.append((h, c, s))
        if not cands:
            break
        cands.sort(key=lambda t: -t[0])
        groups = []                                                      # brightest first, within the strain window
        for h, c, s in cands:
            g = next((g for g in groups if abs(c - g["ref"]) <= half), None)
            if g is None:
                groups.append(dict(ref=c, members={s}, height=h))
            elif s not in g["members"]:
                g["members"].add(s); g["height"] += h
        groups = [g for g in groups if len(g["members"]) >= o["min_seed_spectra"]]
        groups.sort(key=lambda g: -g["height"])
        added = 0
        for g in groups[:o["grow_tries"]]:
            if not np.all(np.abs(g["ref"] - W.refs) >= o["min_sep_nm"]):
                continue
            Wt = W.with_emitter(g["ref"], hmin)
            fT = B3.fit_set(xm, ym, Wt.basis(wmin, wmax), o["starts"], o, o["_workers"])
            k = int(np.argmin(np.abs(Wt.refs - g["ref"])))
            ok, n_core, need = _necessary(xm, ym, Wt, fT, k, o, wmin, wmax)
            hist.append(dict(step="grow", center=round(float(g["ref"]), 2), spectra=len(g["members"]), accepted=bool(ok),
                             need=round(need, 2)))
            if ok:
                W, fit = Wt, fT; added += 1
                if W.K >= o["max_emitters"]:
                    break
        if not added:
            break
    return W, fit


def fit(xs, ys, opts=None, phys=None, workers=None, verbose=False):
    phys = phys or physics.load()
    o = {**B.DEFAULTS, **B.GROW_DEFAULTS, **B3.OPTS3, **OPTS4, **{k: v for k, v in (opts or {}).items() if v is not None}}
    o["_workers"] = workers
    hmin = float(o["window_min_nm"] if o["window_min_nm"] is not None else phys["strain_shift_halfrange_nm"])
    hmax = float(o["window_max_nm"])
    if not np.isfinite(hmin) or hmin <= 0:
        raise ValueError("v4 window_min_nm (or physics strain_shift_halfrange_nm) must be a positive finite number")
    if not np.isfinite(hmax) or hmax <= 0 or hmax < hmin:
        raise ValueError("v4 window_max_nm must be a positive finite number greater than or equal to window_min_nm")
    if o["width_max_nm"] is not None and (not np.isfinite(o["width_max_nm"]) or o["width_max_nm"] <= 0):
        raise ValueError("v4 width_max_nm must be a positive finite number")
    if o["min_sep_nm"] is not None and (not np.isfinite(o["min_sep_nm"]) or o["min_sep_nm"] <= 0):
        raise ValueError("v4 min_sep_nm must be a positive finite number")
    o["window_min_nm"] = hmin
    step = float(np.median(np.diff(xs[0]))); wmin = max(0.8, 1.5 * step)
    cens = B.census(xs, ys, o, workers)
    keeps = [B.data_keep(y, c["sigma"]) for y, c in zip(ys, cens)]
    xm = [x[k] for x, k in zip(xs, keeps)]; ym = [y[k] for y, k in zip(ys, keeps)]
    snr = np.array([c["peak"] / max(c["sigma"], 1e-300) for c in cens])
    seeds = np.flatnonzero(snr >= o["seed_snr"])
    if seeds.size < o["min_seeds"]:
        seeds = np.argsort(-snr)[:min(o["min_seeds"], len(xs))]
    seeds = np.sort(seeds)
    widths = [l["width"] for s in seeds for l in cens[s]["lines"] if l["share"] >= o["min_line_share"]]
    wmax = float(o["width_max_nm"] if o["width_max_nm"] is not None else (np.median(widths) if widths else 10.0))
    wmax = max(wmax, 2 * wmin)
    x_range = (min(float(np.min(x)) for x in xs), max(float(np.max(x)) for x in xs))
    sep_stats = None
    if o["min_sep_nm"] is None:
        learned, sep_stats = learn_min_sep(xs, ys, snr, wmax, o)
        o["min_sep_nm"] = learned if learned else 2 * hmax
        sep_stats["source"] = "learned" if learned else f"fallback 2 x window ({2 * hmax:g} nm): too few within-band pairs"
        if verbose:
            print(f"  v4: emitter spacing {o['min_sep_nm']:.2f} nm ({sep_stats['source']}; {sep_stats})", flush=True)
    em = seed_emitters(cens, seeds, x_range, hmax, wmax, o)
    need = max(o["min_seed_spectra"], int(np.ceil(o["min_seed_frac"] * seeds.size)))
    kept = [e for e in em if e[1] >= need]
    kept.sort(key=lambda e: -e[2]); kept = kept[:o["max_emitters"]]
    hist = [dict(step="seed", seeds=int(seeds.size), seed_snr=o["seed_snr"], width_max_nm=round(wmax, 3),
                 window_nm=[hmin, hmax], min_sep_nm=round(float(o["min_sep_nm"]), 3), min_sep_learned=sep_stats, candidates=len(em), supported=len(kept),
                 centers=sorted(round(e[0], 2) for e in kept))]
    if verbose:
        print(f"  v4: {seeds.size} seed spectra (SNR >= {o['seed_snr']:g}), width cap {wmax:.2f} nm, window +-{hmin:g}-{hmax:g} nm, "
              f"min separation {o['min_sep_nm']:g} nm, {len(em)} candidates -> {len(kept)} emitters with >= {need} seed spectra",
              flush=True)
    if not kept:
        raise RuntimeError("v4: no emitter found in the seed spectra")
    W = Windows([e[0] for e in kept], [e[3] for e in kept], [e[3] for e in kept])
    xs_s = [xs[i] for i in seeds]; ys_s = [ys[i] for i in seeds]; ks_s = [keeps[i] for i in seeds]
    sig_s = [cens[i]["sigma"] for i in seeds]
    xm_s = [xm[i] for i in seeds]; ym_s = [ym[i] for i in seeds]
    f_s = B3.fit_set(xm_s, ym_s, W.basis(wmin, wmax), o["starts"], o, workers)
    W, f_s = _prune(xm_s, ym_s, W, f_s, o, hist, wmin, wmax)
    W, f_s = _widen(xm_s, ym_s, W, f_s, o, hist, wmin, wmax, hmax)
    W, f_s = _grow(xs_s, ys_s, ks_s, sig_s, xm_s, ym_s, W, f_s, o, phys, hist, wmin, wmax, hmax, hmin)
    W, f_s = _prune(xm_s, ym_s, W, f_s, o, hist, wmin, wmax)
    W, f_s = _widen(xm_s, ym_s, W, f_s, o, hist, wmin, wmax, hmax)
    if verbose:
        for h in hist[1:]:
            print(f"  v4 {h}", flush=True)
    b = W.basis(wmin, wmax)
    final = B3.fit_set(xm, ym, b, o["final_starts"], o, workers)
    edges = [edge_fractions(final, b, k) for k in range(b.K)]
    rep = dict(history=hist, seeds=[int(i) for i in seeds], width_max_nm=wmax, window_nm=[hmin, hmax],
               min_sep_nm=o["min_sep_nm"], edge_fraction=[[round(a, 3), round(c, 3)] for a, c, _ in edges],
               n_emitting=int(sum(c["emitting"] for c in cens)), options={k: v for k, v in o.items() if not k.startswith("_")})
    return b, final, rep, keeps
