"""V3 basis discovery: emitters = zero-phonon line (ZPL) + optional tied red phonon sideband (PSB).

Differences from V2 (basis.grow_basis), motivated by single-photon-emitter physics at 4 K:
- one physical emitter is one ZPL; a red-side shoulder closer than R x FWHM to an emitter can only become that emitter's
  phonon sideband (fitting.emitter model), never a new Gaussian line;
- a new emitter must be resolved from every existing one: centre separation >= R x mean FWHM (checked again after fit);
- localised emitters are allowed: the prevalence rules use min_prev (a few % of the map), not V2's 10 %;
- pruning merges unresolved pairs and drops sidebands that held-out BIC does not support.
Census and the initial KDE candidates are V2's (basis.census, basis.candidate_lines).

OPTS3 were frozen on synthetic tuning maps (validation/v3_synthetic.py seeds 101-106) before any held-out map
(seeds 201-206) or real map was scored: resolve 0.85 (0.85 vs 1.0 tested), min_prev 0.03 (no effect 0.03 vs 0.10),
psb_first True (sideband recall 0.14 -> 0.62).
V3.2 (frozen on seeds 101-106 + 111-116, regenerated with the manuscript's instrument resolution and strain range):
width_mode 'fit', no distinct-map exception (it admitted strain-split pieces of one emitter on real data), dip_test on
(an unresolved pair is kept only if >= 3 % of spectra show two maxima with a significant dip), dip_prev 0.03 (= 0.10).
FINAL V3 = 'strict' (dip_test off): on pillar9/pillar6 V3.2 and strict give identical emitters, held-out V3.2 did not
improve on strict, and strict is simpler to specify and port. The V3.1/V3.2 options remain for experiments.
"""
import numpy as np

from . import basis as B, emitter, lm_native, model, physics

OPTS3 = dict(resolve=0.85, min_prev=0.03, min_share=0.02, accept_bic=10.0, spec_gain=10.0, tries=3, max_emitters=24,
             init_support=0.01, new_tol=physics.load()["strain_shift_halfrange_nm"], new_width_max=60.0, psb_r=(0.0, 1.0), psb_dd=(0.0, 6.0), psb_dw=(0.0, 15.0),
             starts=(8.0, 3.0), final_starts=("mid", 5.0, 2.0), psb_first=True,
             width_mode="fit", distinct_r=None, dip_test=False, dip_sigma=3.0, dip_rel=0.02, dip_prev=0.03,
             pair_rule="narrow", width_ratio=2.0, mimic_r=0.8, bg_floor_sigma=3.0, psb_width_gate=False, support_tries=0,
             local_psb_shield=True, no_psb_on_broad=False, broad_x_instrument=5.0)
# 2.5: local_psb_shield - in the local pass a peak much narrower than a fitted sideband (< 0.7 x its width) is a separate
# line, not that sideband's shape residual (a sideband is at least as broad as its ZPL). no_psb_on_broad - a broad
# component (FWHM > broad_x_instrument x instrument, e.g. band-edge emission) gets no tied sideband. Experimental, off:
# psb_width_gate (growth: a candidate narrower than an emitter is not tried as its sideband) and support_tries (when
# no candidate of the map-averaged residual profile is accepted, also try up to this many wavelengths where >= min_prev of
# the spectra exceed the residual limit max(100, 3 sqrt(y)) before growth stops: localized lines (a few % of the map)
# are diluted in the average. 0 / False restore 2.4.
# 2.2: pair_rule "narrow" - a narrow line next to a much broader one (width ratio >= width_ratio, e.g. GaSe band-edge
# emission) is resolved if it is >= resolve x ITS OWN FWHM from the broad centre and the broad component's spatial map
# does not just track it (share-map correlation < mimic_r; a pedestal that tracks a narrow line is how splitting fakes
# Lorentzian wings). Similar-width pairs keep the 2.0 rule (resolve x mean FWHM). "mean" restores the 2.0 rule.
# bg_floor_sigma: the linear background may not fall below -bg_floor_sigma x noise sigma anywhere in the window (None:
# free, as in 2.0).


class EmitterBasis:
    def __init__(self, centers, lo, hi, wmin, wmax, psb=None, allow=None):
        self.centers = np.asarray(centers, float); K = self.centers.size
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.wmin = np.broadcast_to(np.asarray(wmin, float), (K,)).copy()
        self.wmax = np.broadcast_to(np.asarray(wmax, float), (K,)).copy()
        self.psb = np.zeros(K, int) if psb is None else np.asarray(psb, int).copy()
        # per emitter: None (map-wide) or a boolean mask over the map's spectra where its amplitude may be non-zero
        # (local emitters, see local_emitters)
        self.allow = [None] * K if allow is None else list(allow)

    @property
    def K(self):
        return self.centers.size

    def copy(self):
        return EmitterBasis(self.centers, self.lo, self.hi, self.wmin, self.wmax, self.psb, self.allow)

    def without(self, j):
        m = np.arange(self.K) != j
        return EmitterBasis(self.centers[m], self.lo[m], self.hi[m], self.wmin[m], self.wmax[m], self.psb[m],
                            [a for i, a in enumerate(self.allow) if i != j])

    def with_emitter(self, c, tol, wmin, wmax, allow=None):
        cs = np.r_[self.centers, c]; o = np.argsort(cs, kind="stable")
        al = self.allow + [allow]
        return EmitterBasis(cs[o], np.r_[self.lo, c - tol][o], np.r_[self.hi, c + tol][o], np.r_[self.wmin, wmin][o],
                            np.r_[self.wmax, wmax][o], np.r_[self.psb, 0][o], [al[i] for i in o])

    @property
    def local(self):
        return np.array([a is not None for a in self.allow], bool)

    def with_psb(self, j, on=1):
        b = self.copy(); b.psb[j] = on
        return b

    def to_dict(self):
        return dict(K=self.K, centers=self.centers.tolist(), center_lo=self.lo.tolist(), center_hi=self.hi.tolist(),
                    width_min=self.wmin.tolist(), width_max=self.wmax.tolist(), psb=self.psb.tolist(),
                    local_spectra=[None if a is None else int(np.sum(a)) for a in self.allow])


def setup(x, y, b, ws, o, allowed=None):
    off, npar = emitter.offsets(b.psb)
    b0, b1 = model._edge_baseline(x, y)
    yr = max(float(y.max() - y.min()), 1.0); t = (x - model.X_REF) / model.X_SCALE
    p0 = np.empty(npar); lb = np.empty(npar); ub = np.empty(npar)
    p0[:2] = b0, b1; lb[:2] = -yr, -2 * yr; ub[:2] = float(y.max()), 2 * yr
    a0 = np.interp(b.centers, x, y - (b0 + b1 * t))
    for k in range(b.K):
        q = off[k]
        p0[q:q + 4] = max(a0[k], 0.0), b.centers[k], 0.5, (0.5 * (b.wmin[k] + b.wmax[k]) if ws == "mid" else min(ws, b.wmax[k]))
        lb[q:q + 4] = 0.0, b.lo[k], 0.0, b.wmin[k]
        ub[q:q + 4] = max(2 * yr, 1.0), b.hi[k], 1.0, b.wmax[k]
        if allowed is not None and not allowed[k]:          # local emitter outside its patch: amplitude pinned near 0
            p0[q] = 0.0; ub[q] = 1e-9 * yr
        if b.psb[k]:
            p0[q + 4:q + 7] = 0.1, 1.0, 2.0
            lb[q + 4:q + 7] = o["psb_r"][0], o["psb_dd"][0], o["psb_dw"][0]
            ub[q + 4:q + 7] = o["psb_r"][1], o["psb_dd"][1], o["psb_dw"][1]
    bad = (p0 <= lb) | (p0 >= ub); p0[bad] = 0.5 * (lb[bad] + ub[bad])
    if allowed is not None:
        off_k = off[~np.asarray(allowed, bool)]
        p0[off_k] = 0.5 * ub[off_k]
    return p0, lb, ub


class Fit:
    def __init__(self, b, params, sse, n, start=None):
        self.b, self.params, self.sse, self.n, self.start = b, params, sse, n, start
        self.tab = np.array([emitter.table(p, b.psb) for p in params])   # S, K, 8
        zl = self.tab[..., 3]; tot = self.tab[..., 3] + self.tab[..., 4]
        self.share = zl / np.maximum(tot.sum(1, keepdims=True), 1e-300)          # ZPL share of all line area
        self.psb_share = self.tab[..., 4] / np.maximum(tot.sum(1, keepdims=True), 1e-300)
        self.bic = n * np.log(np.maximum(sse, 1e-300) / n) + params.shape[1] * np.log(n)

    def med(self, col, j, strong=0.02):
        m = self.share[:, j] >= strong
        v = self.tab[m, j, col] if m.any() else self.tab[:, j, col]
        return float(np.median(v))


def fit_set(xm, ym, b, starts, o, workers=None, idx=None):
    """Fit spectra (xm, ym) with basis b. idx: their indices in the map, needed when b has local emitters."""
    S = len(xm); npar = emitter.offsets(b.psb)[1]
    has_local = any(a is not None for a in b.allow)
    if has_local and idx is None:
        raise ValueError("fit_set: a basis with local emitters needs the spectra's map indices (idx)")
    allowed = (lambda m: np.array([a is None or bool(a[idx[m]]) for a in b.allow])) if has_local else (lambda m: None)
    params = np.empty((S, npar)); sse = np.full(S, np.inf); start = [None] * S
    lengths = np.array([len(x) for x in xm])
    for L in np.unique(lengths):
        mem = np.flatnonzero(lengths == L)
        X = np.stack([xm[i] for i in mem]); Y = np.stack([ym[i] for i in mem])
        P0, LB, UB, SP = [], [], [], []
        for j in range(len(mem)):
            for ws in starts:
                a, l, u = setup(X[j], Y[j], b, ws, o, allowed(mem[j])); P0.append(a); LB.append(l); UB.append(u); SP.append(j)
        P, SS, _, _ = lm_native.fit_problems_zp(X, Y, np.array(SP), np.array(P0), np.array(LB), np.array(UB), b.psb,
                                                threads=workers or 0)
        SS = np.where(np.isfinite(SS), SS, np.inf); SP = np.array(SP)
        if o.get("bg_floor_sigma") is not None:
            P, SS = _bg_floor(X, Y, SP, P, SS, np.array(LB), np.array(UB), b, o, workers)
        for j, m in enumerate(mem):
            c = np.flatnonzero(SP == j); k = c[np.argmin(SS[c])]
            params[m], sse[m] = P[k], SS[k]; start[m] = starts[k - c[0]]
    return Fit(b, params, sse, lengths.astype(float), start)


def _bg_floor(X, Y, SP, P, SS, LB, UB, b, o, workers):
    """Refit problems whose linear background b0 + b1 t falls below -f x sigma inside the window, with bounds that keep
    it >= -f x sigma everywhere: b0 >= max(-f sigma, b0*) and b1 between the slopes that hit -f sigma at either end
    (any b0 >= b0* only raises the background, so the region is feasible and the floor holds)."""
    T = (X - model.X_REF) / model.X_SCALE
    f = o["bg_floor_sigma"]
    bad = []
    for q in range(len(SP)):
        t = T[SP[q]]; sg = B.noise_sigma(Y[SP[q]])
        if min(P[q, 0] + P[q, 1] * t.min(), P[q, 0] + P[q, 1] * t.max()) < -f * sg:
            bad.append((q, sg))
    if not bad:
        return P, SS
    q_idx = [q for q, _ in bad]
    P0 = P[q_idx].copy(); LB2 = LB[q_idx].copy(); UB2 = UB[q_idx].copy()
    for i, (q, sg) in enumerate(bad):
        t = T[SP[q]]; tlo, thi = float(t.min()), float(t.max()); floor = -f * sg
        b0 = max(floor, float(P[q, 0]))
        LB2[i, 0] = b0; UB2[i, 0] = max(UB2[i, 0], b0 + 1e-9)
        lo1 = (floor - b0) / thi if thi > 0 else -np.inf     # b0 + b1*thi >= floor
        hi1 = (floor - b0) / tlo if tlo < 0 else np.inf      # b0 + b1*tlo >= floor
        LB2[i, 1] = max(LB2[i, 1], lo1); UB2[i, 1] = min(UB2[i, 1], hi1)
        if LB2[i, 1] >= UB2[i, 1]:
            LB2[i, 1] = UB2[i, 1] = 0.0; UB2[i, 1] = 1e-12
        P0[i, 0] = b0 + 1e-9 * max(1.0, abs(b0)); P0[i, 1] = min(max(0.0, LB2[i, 1]), UB2[i, 1])
        bnd = (P0[i] <= LB2[i]) | (P0[i] >= UB2[i]); P0[i, bnd] = 0.5 * (LB2[i, bnd] + UB2[i, bnd])
    P2, S2, _, _ = lm_native.fit_problems_zp(X, Y, SP[q_idx], P0, LB2, UB2, b.psb, threads=workers or 0)
    P = P.copy(); SS = SS.copy()
    P[q_idx] = P2; SS[q_idx] = np.where(np.isfinite(S2), S2, np.inf)
    return P, SS


def share_corr(fit, i, j):
    """Pearson correlation of the two emitters' ZPL-share maps over spectra where either carries >= 2 %."""
    a, b = fit.share[:, i], fit.share[:, j]
    m = (a >= 0.02) | (b >= 0.02)
    if m.sum() < 5 or a[m].std() == 0 or b[m].std() == 0:
        return 1.0
    return float(np.corrcoef(a[m], b[m])[0, 1])


def resolved_in_data(xd, yd, sig, c1, c2, keep=None):
    """Fraction of spectra whose measured (3-px smoothed) spectrum shows two separate maxima near c1[s] and c2[s]
    with a dip between them of >= max(dip_sigma * sigma, dip_rel * peak) below the lower maximum (Rayleigh test on the
    data, not on the model). c1, c2: per-spectrum positions (follow strain shifts)."""
    o = OPTS_RUNTIME
    hits = 0
    for s, (x, y) in enumerate(zip(xd, yd)):
        a, b = sorted((float(c1[s]), float(c2[s])))
        if b - a <= 0:
            continue
        ys = np.convolve(y, np.ones(3) / 3, mode="same")
        h = 0.4 * (b - a)
        ia = np.flatnonzero(np.abs(x - a) <= h); ib = np.flatnonzero(np.abs(x - b) <= h)
        if ia.size == 0 or ib.size == 0:
            continue
        ma = ia[np.argmax(ys[ia])]; mb = ib[np.argmax(ys[ib])]
        if not (0 < ma < x.size - 1 and 0 < mb < x.size - 1 and ma < mb):
            continue
        if not (ys[ma] >= ys[ma - 1] and ys[ma] >= ys[ma + 1] and ys[mb] >= ys[mb - 1] and ys[mb] >= ys[mb + 1]):
            continue                                    # both must be true local maxima
        dip = min(ys[ma], ys[mb]) - ys[ma:mb + 1].min()
        peak = float(ys.max() - np.median(ys))
        if dip >= max(o["dip_sigma"] * sig[s], o["dip_rel"] * peak):
            hits += 1
    return hits / max(len(xd), 1)


OPTS_RUNTIME = dict(OPTS3)


def pair_unresolved(sep, w1, w2, corr, o):
    """True if two lines (centre separation sep, FWHMs w1, w2) count as one feature (see OPTS3 pair_rule)."""
    wn, wb = min(w1, w2), max(w1, w2)
    if o.get("pair_rule", "mean") == "narrow" and wb >= o["width_ratio"] * wn:
        return sep < o["resolve"] * wn or (sep < o["resolve"] * 0.5 * (w1 + w2) and corr() >= o["mimic_r"])
    return sep < o["resolve"] * 0.5 * (w1 + w2)


def unresolved(fit, R, wfun=None, distinct_r=None, data=None):
    """Adjacent emitter pairs closer than R x reference width that are not spatially distinct.

    Reference width: mean fitted median FWHM, or (wfun) the narrowest of the fitted and census widths of the pair, so
    a line broadened by absorbing its neighbours cannot widen its own exclusion zone. Pairs whose share maps correlate
    with |r| < distinct_r behave like different emitters and are not unresolved."""
    b = fit.b; out = []
    for j in range(b.K - 1):
        c1, c2 = fit.med(0, j), fit.med(0, j + 1)
        w1, w2 = fit.med(2, j), fit.med(2, j + 1)
        if wfun is not None:
            w1, w2 = min(w1, wfun(c1, w1)), min(w2, wfun(c2, w2))
        o = {**OPTS_RUNTIME, "resolve": R}
        if pair_unresolved(c2 - c1, w1, w2, lambda: share_corr(fit, j, j + 1), o) and \
                not (distinct_r is not None and abs(share_corr(fit, j, j + 1)) < distinct_r):
            if data is not None:                    # V3.2: separate if the data themselves resolve the pair
                xd, yd, sg = data
                if resolved_in_data(xd, yd, sg, fit.tab[:, j, 0], fit.tab[:, j + 1, 0]) >= OPTS_RUNTIME["dip_prev"]:
                    continue
            out.append(j)
    return out


def grow(xs, ys, opts=None, workers=None, verbose=False):
    o = {**B.DEFAULTS, **B.GROW_DEFAULTS, **OPTS3, **(opts or {})}
    step = float(np.median(np.diff(xs[0]))); wmin = max(0.8, 1.5 * step)
    cens = B.census(xs, ys, o, workers)
    emit = np.flatnonzero([c["emitting"] for c in cens])
    keeps = [B.data_keep(y, c["sigma"]) for y, c in zip(ys, cens)]
    xm = [x[k] for x, k in zip(xs, keeps)]; ym = [y[k] for y, k in zip(ys, keeps)]
    F = [emit[0::3], emit[1::3], emit[2::3]]
    pick = lambda f: ([xm[i] for i in f], [ym[i] for i in f])
    A, Bf, C = pick(F[0]), pick(F[1]), pick(F[2])
    sigA = [cens[i]["sigma"] for i in F[0]]; xA = [xs[i] for i in F[0]]; yA = [ys[i] for i in F[0]]; kA = [keeps[i] for i in F[0]]
    cands = [c for c in B.candidate_lines([cens[i] for i in F[0]], step, o) if c["support"] >= o["init_support"]]
    cl = np.array([(l["center"], l["width"]) for i in F[0] if cens[i]["emitting"] for l in cens[i]["lines"]] or [(0.0, 1.0)])

    def wfun(c, fallback):
        """Median census FWHM of fold-A lines within 1 nm of c (per-spectrum fits see sharp lines individually)."""
        m = np.abs(cl[:, 0] - c) <= 1.0
        return float(np.median(cl[m, 1])) if m.sum() >= 3 else fallback

    WF = wfun if o["width_mode"] == "census" else None
    DR = o["distinct_r"]
    OPTS_RUNTIME.clear(); OPTS_RUNTIME.update(o)
    sig_all = [c["sigma"] for c in cens]
    DATA_B = (Bf[0], Bf[1], [sig_all[i] for i in F[1]]) if o["dip_test"] else None
    DATA_A = (A[0], A[1], [sig_all[i] for i in F[0]]) if o["dip_test"] else None
    sel = sorted(cands, key=lambda d: d["center"])
    b = EmitterBasis([d["center"] for d in sel], [d["center"] - d["tol"] for d in sel], [d["center"] + d["tol"] for d in sel],
                     [max(d["width_min"], wmin) for d in sel], [d["width_max"] for d in sel])
    hist = [dict(step="init", K=b.K, centers=b.centers.round(2).tolist())]
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    def better(new, old):
        d = new - old
        return d.sum() < -o["accept_bic"] and np.mean(d < -o["spec_gain"]) >= o["min_prev"]

    def principal(fit, j):
        return np.mean(fit.share[:, j] >= o["min_share"]) >= o["min_prev"]

    fB = fit_set(*Bf, b, o["starts"], o, workers)
    # 0. merge unresolved initial candidates (keep the one with more ZPL area)
    while True:
        u = unresolved(fB, o["resolve"], WF, DR, DATA_B)
        if not u:
            break
        j = u[0]; jd = j if fB.tab[:, j, 3].sum() < fB.tab[:, j + 1, 3].sum() else j + 1
        log(f"  init merge: drop {b.centers[jd]:.1f}")
        hist.append(dict(step="drop-unresolved", center=float(b.centers[jd]), K=b.K - 1))
        b = b.without(jd); fB = fit_set(*Bf, b, o["starts"], o, workers)
    # 1. growth
    while b.K < o["max_emitters"]:
        fA = fit_set(*A, b, o["starts"], o, workers)
        x0 = xA[0]; acc = np.zeros(x0.size); cnt = np.zeros(x0.size)
        for s in range(len(xA)):
            z = (yA[s] - emitter.evaluate(fA.params[s], xA[s], b.psb)) / sigA[s]
            z = np.where(kA[s], np.maximum(z, 0.0), 0.0)
            acc += np.interp(x0, xA[s], z); cnt += np.interp(x0, xA[s], kA[s].astype(float))
        prof = np.convolve(acc / np.maximum(cnt, 1.0), np.ones(3) / 3, mode="same")
        order = [i for i in np.argsort(-prof, kind="stable") if 0 < i < x0.size - 1 and prof[i] >= prof[i - 1] and prof[i] >= prof[i + 1]]
        tried = order[:o["tries"]]
        if o.get("support_tries"):
            # support profile: fraction of spectra whose (3-px smoothed) residual exceeds the residual limit
            sup = np.zeros(x0.size)
            for s in range(len(xA)):
                r = np.where(kA[s], yA[s] - emitter.evaluate(fA.params[s], xA[s], b.psb), 0.0)
                r = np.convolve(r, np.ones(3) / 3, mode="same")
                lim = np.maximum(100.0, 3.0 * np.sqrt(np.maximum(yA[s], 0.0)))
                sup += np.interp(x0, xA[s], (r > lim).astype(float))
            sup = np.convolve(sup / max(len(xA), 1), np.ones(3) / 3, mode="same")
            near = lambda i: any(abs(x0[i] - x0[t]) < 1.0 for t in tried)
            extra = [i for i in np.argsort(-sup, kind="stable") if 0 < i < x0.size - 1 and sup[i] >= o["min_prev"]
                     and sup[i] >= sup[i - 1] and sup[i] >= sup[i + 1]]
            for i in extra:
                if len(tried) >= o["tries"] + o["support_tries"]:
                    break
                if not near(i):
                    tried.append(i)
        accepted = False
        for i in tried:
            c = float(x0[i])
            cen = np.array([fA.med(0, j) for j in range(b.K)]); wid = np.array([fA.med(2, j) for j in range(b.K)])
            j = int(np.argmin(np.abs(cen - c))) if b.K else -1
            moves = []
            if o["psb_first"] and b.K:
                # red-side peak within sideband reach of an emitter without one: try the sideband first
                red = [k for k in range(b.K) if not b.psb[k] and 0 < c - cen[k] <= 0.5 * wid[k] + o["psb_dd"][1] + 0.5 * wid[k]]
                if o.get("psb_width_gate"):          # a sideband is at least as broad as its ZPL: a narrower candidate is not one
                    red = [k for k in red if wfun(c, wmin) >= wid[k]]
                if o.get("no_psb_on_broad"):         # band-edge / broad components carry no tied sideband
                    red = [k for k in red if wid[k] <= o["broad_x_instrument"] * physics.instrument_fwhm_nm(cen[k])]
                if red:
                    moves.append(("psb", b.with_psb(min(red, key=lambda k: c - cen[k]))))
            wj = min(wid[j], WF(cen[j], wid[j]), WF(c, wid[j])) if (b.K and WF) else (wid[j] if b.K else 0.0)
            if b.K and o.get("pair_rule", "mean") == "narrow":
                wc = wfun(c, wmin)                   # census width of the candidate line
                if max(wc, wj) >= o["width_ratio"] * min(wc, wj):
                    wj = min(wc, wj)                 # very different widths: judged by the narrower (checked after fit)
            if b.K == 0 or abs(c - cen[j]) >= o["resolve"] * wj:
                moves.append(("add", b.with_emitter(c, o["new_tol"], wmin, o["new_width_max"])))
            else:
                if c > cen[j] and not b.psb[j] and not moves and not (o.get("psb_width_gate") and wfun(c, wmin) < wid[j]) \
                        and not (o.get("no_psb_on_broad") and wid[j] > o["broad_x_instrument"] * physics.instrument_fwhm_nm(cen[j])):
                    moves.append(("psb", b.with_psb(j)))
                if DR is not None:          # unresolved: allowed only if its map is distinct (checked after the fit)
                    moves.append(("add", b.with_emitter(c, o["new_tol"], wmin, o["new_width_max"])))
                elif DATA_A is not None:    # V3.2: allowed only if the data resolve it from its neighbour
                    cj = fA.tab[:, j, 0]
                    if resolved_in_data(*DATA_A, cj, cj + (c - cen[j])) >= o["dip_prev"]:
                        moves.append(("add", b.with_emitter(c, o["new_tol"], wmin, o["new_width_max"])))
            ok = False
            for kind, trial in moves:
                fT = fit_set(*Bf, trial, o["starts"], o, workers)
                if kind == "add":
                    jn = int(np.argmin(np.abs(trial.centers - c)))
                    ok = better(fT.bic, fB.bic) and principal(fT, jn) and not any(u in (jn - 1, jn) for u in unresolved(fT, o["resolve"], WF, DR, DATA_B))
                else:
                    ok = better(fT.bic, fB.bic)
                if ok:
                    break
            if not moves:
                continue
            if ok:
                b, fB, accepted = trial, fT, True
                hist.append(dict(step=kind, center=c if kind == "add" else float(b.centers[j]), K=b.K, psb=int(b.psb.sum())))
                log(f"  {kind} {c:.1f}  K={b.K} psb={int(b.psb.sum())}")
                break
        if not accepted:
            break
    # 2. pruning
    changed = True
    while changed and b.K > 1:
        changed = False
        for j in range(b.K):
            if not principal(fB, j):
                hist.append(dict(step="drop-not-principal", center=float(b.centers[j]), K=b.K - 1))
                log(f"  drop {b.centers[j]:.1f} (not principal)")
                b = b.without(j); fB = fit_set(*Bf, b, o["starts"], o, workers); changed = True
                break
        if changed:
            continue
        u = unresolved(fB, o["resolve"], WF, DR, DATA_B)
        if u:
            j = u[0]; jd = j if fB.tab[:, j, 3].sum() < fB.tab[:, j + 1, 3].sum() else j + 1
            hist.append(dict(step="drop-unresolved", center=float(b.centers[jd]), K=b.K - 1))
            log(f"  drop {b.centers[jd]:.1f} (unresolved)")
            b = b.without(jd); fB = fit_set(*Bf, b, o["starts"], o, workers); changed = True
            continue
        for j in range(b.K):
            if b.psb[j]:
                t = b.with_psb(j, 0); fT = fit_set(*Bf, t, o["starts"], o, workers)
                if not better(fB.bic, fT.bic):
                    hist.append(dict(step="drop-psb", center=float(b.centers[j]), K=b.K))
                    log(f"  drop PSB of {b.centers[j]:.1f}")
                    b, fB, changed = t, fT, True
                    break
        if changed:
            continue
        for j in range(b.K):
            t = b.without(j); fT = fit_set(*Bf, t, o["starts"], o, workers)
            if not better(fB.bic, fT.bic):
                hist.append(dict(step="drop", center=float(b.centers[j]), K=b.K - 1))
                log(f"  drop {b.centers[j]:.1f}")
                b, fB, changed = t, fT, True
                break
    fC = fit_set(*C, b, o["starts"], o, workers)
    final = fit_set(xm, ym, b, o["final_starts"], o, workers)
    rep = dict(history=hist, heldout_bic_C=float(fC.bic.sum()), n_emitting=int(emit.size), folds=[len(f) for f in F],
               census=cens, options={k: v for k, v in o.items()})
    return b, final, rep, keeps


# --------------------------------------------------------------------------- local emitters (2.1)
LOCAL = dict(ratio=2.0, min_spectra=2, width_x_instrument=0.6, group_x_strain=0.5, mask_gap_px=2, accept_frac=0.5,
             rounds=2, max_new=12)


def _neighbours(order, rows, cols):
    """For each spectrum (file order), the set of spectra in its 8-neighbourhood on the grid."""
    pos = {p: i for i, p in enumerate(order)}
    out = []
    for p in order:
        r, c = divmod(p, cols)
        out.append({pos[(r + dr) * cols + (c + dc)] for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dr or dc) and 0 <= r + dr < rows and 0 <= c + dc < cols and (r + dr) * cols + (c + dc) in pos})
    return out


def _components(members, nbr):
    """Connected components of `members` (spectrum indices) under the neighbour relation; one group if no grid."""
    if nbr is None:
        return [sorted(members)]
    left, comps = set(members), []
    while left:
        seed = left.pop(); comp = {seed}; stack = [seed]
        while stack:
            for n in nbr[stack.pop()] & left:
                left.discard(n); comp.add(n); stack.append(n)
        comps.append(sorted(comp))
    return comps


def _residual_peaks(x, y, keep, p, psb, limit_fn, inst_fn, o, L):
    """Residual peaks of one spectrum that could be an unmodelled emitter: >= L['ratio'] x the residual limit, at least
    L['width_x_instrument'] x the instrument FWHM wide, resolved from every fitted ZPL, outside every fitted sideband and
    away from masked pixels. Returns [(centre_nm, height, fwhm_nm)]."""
    from scipy.signal import find_peaks, peak_widths
    tot = emitter.evaluate(p, x, psb)
    o_ = np.argsort(x); x, y, keep, tot = x[o_], y[o_], keep[o_], tot[o_]
    r = np.where(keep, y - tot, 0.0); z = r / limit_fn(y)
    step = float(np.median(np.diff(x)))
    pk, _ = find_peaks(z, height=L["ratio"])
    if pk.size == 0:
        return []
    w = peak_widths(r, pk, rel_height=0.5)[0] * step
    tab = emitter.table(p, psb)
    masked = np.flatnonzero(~keep)
    out = []
    for i, wi in zip(pk, w):
        c = float(x[i])
        if wi < L["width_x_instrument"] * inst_fn(c) or wi > o["new_width_max"]:
            continue
        if masked.size and np.min(np.abs(masked - i)) <= L["mask_gap_px"]:
            continue
        clash = False
        for k in range(len(psb)):
            ck, sk = tab[k, 0], tab[k, 2]
            if tab[k, 3] <= 0:
                continue
            if abs(c - ck) < o["resolve"] * 0.5 * (sk + wi):
                clash = True; break
            if psb[k] and tab[k, 4] > 0:
                pc, pw = ck + tab[k, 5], tab[k, 6]
                # inside a fitted sideband: only a peak of comparable width can be that sideband's shape residual; a much
                # narrower peak is a separate line (a sideband is at least as broad as its ZPL)
                if abs(c - pc) <= 0.5 * pw and not (o.get("local_psb_shield", o.get("psb_width_gate")) and wi < 0.7 * pw):
                    clash = True; break
        if not clash:
            out.append((c, float(r[i]), float(wi)))
    return out


def local_emitters(xs, ys, keeps, b, final, o, phys, grid=None, workers=None, verbose=False):
    """Add emitters that exist only in a small, spatially connected patch of the map (V3's map-wide prevalence rule,
    min_prev, cannot see them). Each candidate is a resolved residual peak (see _residual_peaks) found at the same
    wavelength in >= L['min_spectra'] neighbouring spectra; its amplitude is allowed only in that patch and its
    8-neighbourhood, so it cannot split a line elsewhere. It is kept if the per-spectrum BIC improves by spec_gain in
    >= accept_frac of the patch and it stays resolved from the other emitters there."""
    from .report import residual_limit
    L = LOCAL; S = len(xs)
    xm = [x[k] for x, k in zip(xs, keeps)]; ym = [y[k] for y, k in zip(ys, keeps)]
    idx_all = np.arange(S)
    nbr = _neighbours(grid["order"], grid["rows"], grid["cols"]) if grid and grid.get("rows") else None
    inst = lambda c: physics.instrument_fwhm_nm(c, phys)
    added = []
    for rnd in range(L["rounds"]):
        cands = []
        for s in range(S):
            for c, h, w in _residual_peaks(np.asarray(xs[s]), np.asarray(ys[s]), keeps[s], final.params[s], b.psb,
                                           residual_limit, inst, o, L):
                cands.append((c, s, h, w))
        # a candidate within the strain tolerance of an existing local emitter is that emitter (already modelled there),
        # not a new one
        gtol = L["group_x_strain"] * phys["strain_shift_halfrange_nm"]
        loc_c = [float(cc) for cc, a in zip(b.centers, b.allow) if a is not None]
        cands = [t for t in cands if all(abs(t[0] - cc) > gtol for cc in loc_c)]
        if not cands:
            break
        cands.sort()
        groups, cur = [], [cands[0]]
        for t in cands[1:]:
            if t[0] - np.median([u[0] for u in cur]) <= gtol:
                cur.append(t)
            else:
                groups.append(cur); cur = [t]
        groups.append(cur)
        patches = []
        for g in groups:
            by_s = {}
            for c, s, h, w in g:
                if s not in by_s or h > by_s[s][1]:
                    by_s[s] = (c, h, w)
            # all connected patches at this wavelength (each >= min_spectra) form ONE emitter allowed in their union
            comp = sorted(s for cc in _components(list(by_s), nbr) if len(cc) >= L["min_spectra"] for s in cc)
            if comp:
                patches.append((sum(by_s[s][1] for s in comp), comp, float(np.median([by_s[s][0] for s in comp])),
                                float(np.median([by_s[s][2] for s in comp]))))
        patches.sort(key=lambda t: -t[0])
        new_round = 0
        for _, comp, c, w in patches[:L["max_new"]]:
            allow = np.zeros(S, bool); allow[comp] = True
            if nbr is not None:
                for s in comp:
                    allow[list(nbr[s])] = True
            tol = max(gtol, 1.0)
            wmin = max(0.8, 1.5 * float(np.median(np.diff(xs[0]))))
            trial = b.with_emitter(c, tol, wmin, max(3 * w, 2 * inst(c)), allow=allow)
            sub = np.flatnonzero(allow)
            fT = fit_set([xm[i] for i in sub], [ym[i] for i in sub], trial, o["final_starts"], o, workers, idx=sub)
            old_sse = final.sse[sub]; n = final.n[sub]
            d = (n * np.log(np.maximum(fT.sse, 1e-300) / n) + 4 * np.log(n)) - n * np.log(np.maximum(old_sse, 1e-300) / n)
            jn = int(np.argmin(np.abs(trial.centers - c)))
            core = np.isin(sub, comp)
            gain = float(np.mean(d[core] < -o["spec_gain"]))
            ok_res = True
            for q, s in enumerate(sub):
                if not core[q]:
                    continue
                tj = fT.tab[q, jn]
                for k in range(trial.K):
                    if k == jn or fT.tab[q, k, 3] <= 0.02 * max(fT.tab[q, :, 3].sum(), 1e-300):
                        continue
                    if abs(fT.tab[q, k, 0] - tj[0]) < o["resolve"] * 0.5 * (fT.tab[q, k, 2] + tj[2]):
                        ok_res = False
            dup = any(a is not None and abs(cc - c) <= gtol for cc, a in zip(b.centers, b.allow))
            accept = gain >= L["accept_frac"] and ok_res and not dup   # dup: an earlier patch already added this emitter
            if verbose:
                print(f"  local {c:.1f} nm, {len(comp)} spectra: BIC gain in {gain:.0%}, resolved {ok_res}"
                      f"{', duplicate' if dup else ''} -> {'add' if accept else 'reject'}", flush=True)
            if accept:
                b = trial
                final = fit_set(xm, ym, b, o["final_starts"], o, workers, idx=idx_all)
                added.append(dict(center=round(c, 2), spectra=len(comp), patch=int(allow.sum()), bic_gain_fraction=round(gain, 3),
                                  round=rnd + 1))
                new_round += 1
        if not new_round:
            break
    return b, final, added
