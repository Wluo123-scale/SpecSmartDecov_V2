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
             width_mode="fit", distinct_r=None, dip_test=False, dip_sigma=3.0, dip_rel=0.02, dip_prev=0.03)


class EmitterBasis:
    def __init__(self, centers, lo, hi, wmin, wmax, psb=None):
        self.centers = np.asarray(centers, float); K = self.centers.size
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.wmin = np.broadcast_to(np.asarray(wmin, float), (K,)).copy()
        self.wmax = np.broadcast_to(np.asarray(wmax, float), (K,)).copy()
        self.psb = np.zeros(K, int) if psb is None else np.asarray(psb, int).copy()

    @property
    def K(self):
        return self.centers.size

    def copy(self):
        return EmitterBasis(self.centers, self.lo, self.hi, self.wmin, self.wmax, self.psb)

    def without(self, j):
        m = np.arange(self.K) != j
        return EmitterBasis(self.centers[m], self.lo[m], self.hi[m], self.wmin[m], self.wmax[m], self.psb[m])

    def with_emitter(self, c, tol, wmin, wmax):
        cs = np.r_[self.centers, c]; o = np.argsort(cs, kind="stable")
        return EmitterBasis(cs[o], np.r_[self.lo, c - tol][o], np.r_[self.hi, c + tol][o], np.r_[self.wmin, wmin][o],
                            np.r_[self.wmax, wmax][o], np.r_[self.psb, 0][o])

    def with_psb(self, j, on=1):
        b = self.copy(); b.psb[j] = on
        return b

    def to_dict(self):
        return dict(K=self.K, centers=self.centers.tolist(), center_lo=self.lo.tolist(), center_hi=self.hi.tolist(),
                    width_min=self.wmin.tolist(), width_max=self.wmax.tolist(), psb=self.psb.tolist())


def setup(x, y, b, ws, o):
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
        if b.psb[k]:
            p0[q + 4:q + 7] = 0.1, 1.0, 2.0
            lb[q + 4:q + 7] = o["psb_r"][0], o["psb_dd"][0], o["psb_dw"][0]
            ub[q + 4:q + 7] = o["psb_r"][1], o["psb_dd"][1], o["psb_dw"][1]
    bad = (p0 <= lb) | (p0 >= ub); p0[bad] = 0.5 * (lb[bad] + ub[bad])
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


def fit_set(xm, ym, b, starts, o, workers=None):
    S = len(xm); npar = emitter.offsets(b.psb)[1]
    params = np.empty((S, npar)); sse = np.full(S, np.inf); start = [None] * S
    lengths = np.array([len(x) for x in xm])
    for L in np.unique(lengths):
        mem = np.flatnonzero(lengths == L)
        X = np.stack([xm[i] for i in mem]); Y = np.stack([ym[i] for i in mem])
        P0, LB, UB, SP = [], [], [], []
        for j in range(len(mem)):
            for ws in starts:
                a, l, u = setup(X[j], Y[j], b, ws, o); P0.append(a); LB.append(l); UB.append(u); SP.append(j)
        P, SS, _, _ = lm_native.fit_problems_zp(X, Y, np.array(SP), np.array(P0), np.array(LB), np.array(UB), b.psb,
                                                threads=workers or 0)
        SS = np.where(np.isfinite(SS), SS, np.inf); SP = np.array(SP)
        for j, m in enumerate(mem):
            c = np.flatnonzero(SP == j); k = c[np.argmin(SS[c])]
            params[m], sse[m] = P[k], SS[k]; start[m] = starts[k - c[0]]
    return Fit(b, params, sse, lengths.astype(float), start)


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


def unresolved(fit, R, wfun=None, distinct_r=None, data=None):
    """Adjacent emitter pairs closer than R x reference width that are not spatially distinct.

    Reference width: mean fitted median FWHM, or (wfun) the narrowest of the fitted and census widths of the pair, so
    a line broadened by absorbing its neighbours cannot widen its own exclusion zone. Pairs whose share maps correlate
    with |r| < distinct_r behave like different emitters and are not unresolved."""
    b = fit.b; out = []
    for j in range(b.K - 1):
        c1, c2 = fit.med(0, j), fit.med(0, j + 1)
        w = 0.5 * (fit.med(2, j) + fit.med(2, j + 1))
        if wfun is not None:
            w = min(w, wfun(c1, fit.med(2, j)), wfun(c2, fit.med(2, j + 1)))
        if c2 - c1 < R * w and not (distinct_r is not None and abs(share_corr(fit, j, j + 1)) < distinct_r):
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
    keeps = [B.cosmic_keep(y, c["sigma"]) for y, c in zip(ys, cens)]
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
        accepted = False
        for i in order[:o["tries"]]:
            c = float(x0[i])
            cen = np.array([fA.med(0, j) for j in range(b.K)]); wid = np.array([fA.med(2, j) for j in range(b.K)])
            j = int(np.argmin(np.abs(cen - c))) if b.K else -1
            moves = []
            if o["psb_first"] and b.K:
                # red-side peak within sideband reach of an emitter without one: try the sideband first
                red = [k for k in range(b.K) if not b.psb[k] and 0 < c - cen[k] <= 0.5 * wid[k] + o["psb_dd"][1] + 0.5 * wid[k]]
                if red:
                    moves.append(("psb", b.with_psb(min(red, key=lambda k: c - cen[k]))))
            wj = min(wid[j], WF(cen[j], wid[j]), WF(c, wid[j])) if (b.K and WF) else (wid[j] if b.K else 0.0)
            if b.K == 0 or abs(c - cen[j]) >= o["resolve"] * wj:
                moves.append(("add", b.with_emitter(c, o["new_tol"], wmin, o["new_width_max"])))
            else:
                if c > cen[j] and not b.psb[j] and not moves:
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
