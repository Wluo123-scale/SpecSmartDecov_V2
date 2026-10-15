# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Batched bound-constrained least squares with an analytic Jacobian (NumPy, CPU).

Each step is a Levenberg-Marquardt step in Coleman-Li affinely scaled variables (the same
scaling as MATLAB/SciPy trust-region-reflective): iterates stay strictly inside the box and
approach active bounds gradually, which avoids the early pinning of peak heights at zero that
a projected LM suffers on this 49-parameter problem.

All spectra in a batch share one grid length; every iteration operates on the whole set of
still-active spectra at once, so per-spectrum Python overhead disappears.
"""
import numpy as np

from .model import LN2, K_GAUSS, INV_PI, X_REF, X_SCALE

DEFAULTS = dict(max_iter=300, ftol=1e-8, xtol=1e-8, gtol=1e-10, lam0=1e-2, theta=0.9)


def residual_and_jacobian(p, x, y, want_jac=True, nbg=1):
    """p (B,P), x,y (B,n) -> r (B,n), J (B,n,P). P = nbg + 4K; nbg=1 constant, 2 linear background."""
    NPAR = p.shape[1]; NP = (NPAR - nbg) // 4
    pk = p[:, nbg:].reshape(-1, NP, 4)
    a, c, nu, s = (pk[:, :, i, None] for i in range(4))           # (B,NP,1)
    u = (x[:, None, :] - c) / s                                   # (B,NP,n)
    den = nu * K_GAUSS + (1 - nu) * INV_PI
    w = nu * K_GAUSS / den
    e = np.exp(-4 * LN2 * u * u)
    L = 1.0 / (1 + 4 * u * u)
    prof = w * e + (1 - w) * L
    t = (x - X_REF) / X_SCALE
    r = p[:, :1] + np.einsum("bkn,bk->bn", prof, pk[:, :, 0]) - y
    if nbg == 2:
        r = r + p[:, 1:2] * t
    if not want_jac:
        return r, None
    dpdu = -8 * u * (w * LN2 * e + (1 - w) * L * L)
    B, n = x.shape
    J = np.empty((B, n, NPAR))
    J[:, :, 0] = 1.0
    if nbg == 2:
        J[:, :, 1] = t
    Jp = J[:, :, nbg:].reshape(B, n, NP, 4)
    Jp[..., 0] = prof.transpose(0, 2, 1)
    Jp[..., 1] = (-a * dpdu / s).transpose(0, 2, 1)
    Jp[..., 2] = (a * (e - L) * (K_GAUSS * INV_PI / (den * den))).transpose(0, 2, 1)
    Jp[..., 3] = (-a * dpdu * u / s).transpose(0, 2, 1)
    return r, J


def coleman_li(p, lb, ub, g):
    """Scaling vector v and its derivative dv (SciPy trf CL_scaling_vector)."""
    v = np.ones_like(p)
    dv = np.zeros_like(p)
    m = g < 0
    v[m], dv[m] = (ub - p)[m], -1.0
    m = g > 0
    v[m], dv[m] = (p - lb)[m], 1.0
    return v, dv


def _solve(M, rhs):
    try:
        return np.linalg.solve(M, rhs[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.stack([np.linalg.lstsq(m, b, rcond=None)[0] for m, b in zip(M, rhs)])


def make_interior(p, lb, ub, rstep=1e-10):
    span = ub - lb
    return np.clip(p, lb + rstep * span, ub - rstep * span)


def fit_batch(p0, lb, ub, x, y, nbg=1, **opts):
    """Fit B spectra. Returns (p, sse, n_iter, status); status 0 = converged, 1 = max_iter."""
    o = {**DEFAULTS, **opts}
    p = make_interior(p0.astype(float, copy=True), lb, ub)
    B = p.shape[0]
    r, J = residual_and_jacobian(p, x, y, nbg=nbg)
    F = 0.5 * np.einsum("bn,bn->b", r, r)
    g = np.matmul(J.transpose(0, 2, 1), r[..., None])[..., 0]
    A = np.matmul(J.transpose(0, 2, 1), J)
    lam = np.full(B, float(o["lam0"]))
    vfac = np.full(B, 2.0)
    n_iter = np.zeros(B, dtype=int)
    status = np.ones(B, dtype=int)
    active = np.arange(B)
    idx = np.arange(p.shape[1])
    for _ in range(o["max_iter"]):
        if active.size == 0:
            break
        i = active
        pa, la, ua, ga, Aa = p[i], lb[i], ub[i], g[i], A[i]
        v, dv = coleman_li(pa, la, ua, ga)
        d = np.sqrt(v)
        gh = d * ga
        gconv = np.max(np.abs(gh), axis=1) <= o["gtol"] * np.maximum(1.0, np.sqrt(2 * F[i]))
        Ah = Aa * d[:, :, None] * d[:, None, :]
        Ah[:, idx, idx] += ga * dv                                  # C = diag(g*dv) >= 0
        diag = np.maximum(Ah[:, idx, idx], 1e-12 * np.max(Ah[:, idx, idx], axis=1, keepdims=True) + 1e-300)
        M = Ah.copy()
        M[:, idx, idx] += lam[i, None] * diag
        sh = _solve(M, -gh)
        step = d * sh
        # stay strictly feasible: components that would cross a bound move theta of the way to it
        pn = pa + step
        pn = np.where(pn >= ua, pa + o["theta"] * (ua - pa), pn)
        pn = np.where(pn <= la, pa + o["theta"] * (la - pa), pn)
        step = pn - pa
        sh = np.where(d > 0, step / np.where(d > 0, d, 1.0), 0.0)
        pred = -np.einsum("bp,bp->b", gh, sh) - 0.5 * np.einsum("bp,bpq,bq->b", sh, Ah, sh)
        rn, _ = residual_and_jacobian(pn, x[i], y[i], want_jac=False, nbg=nbg)
        Fn = 0.5 * np.einsum("bn,bn->b", rn, rn)
        rho = (F[i] - Fn) / np.where(pred > 0, pred, np.inf)
        ok = (Fn < F[i]) & (rho > 0)
        n_iter[i] += 1
        rel_dF = (F[i] - Fn) / np.maximum(F[i], 1e-300)
        small_step = np.max(np.abs(step) / (np.abs(pa) + 1e-12), axis=1) <= o["xtol"]
        conv = gconv | (ok & ((rel_dF <= o["ftol"]) | small_step)) | (lam[i] > 1e16)
        acc = i[ok]
        if acc.size:
            p[acc] = pn[ok]
            ra, Ja = residual_and_jacobian(p[acc], x[acc], y[acc], nbg=nbg)
            F[acc] = Fn[ok]
            Jt = Ja.transpose(0, 2, 1)
            g[acc] = np.matmul(Jt, ra[..., None])[..., 0]
            A[acc] = np.matmul(Jt, Ja)
            lam[acc] *= np.maximum(1 / 3, 1 - (2 * rho[ok] - 1) ** 3)
            vfac[acc] = 2.0
        rej = i[~ok]
        lam[rej] *= vfac[rej]
        vfac[rej] *= 2.0
        status[i[conv]] = 0
        active = i[~conv]
    return p, 2 * F, n_iter, status
