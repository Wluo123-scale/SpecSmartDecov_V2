# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Batched bound-constrained least squares on CUDA (CuPy RawKernel, NVRTC-compiled).

Same algorithm and stopping rules as lm_cpu.fit_batch: an LM step in Coleman-Li scaled
variables with component-wise strictly-feasible truncation. One thread block owns one
(spectrum, start) problem and runs the whole iteration loop on chip:

* Jacobian rows are produced in shared-memory tiles of TK points and folded straight into
  J^T J and J^T r, so the n x 49 Jacobian is never stored;
* the damped 49 x 49 system is solved by an in-block Cholesky factorisation.

Only the CUDA runtime and NVRTC are needed (no cuBLAS/cuSOLVER).
"""
import numpy as np

from .model import NP, NPAR, LN2, K_GAUSS, INV_PI

THREADS = 128
TK = 8  # Jacobian rows per shared-memory tile


def _pair_tables():
    """Work items for the J^T J / J^T r accumulation: (i, j) with j <= i, then (i, -1) for g."""
    pi, pj = [], []
    for i in range(NPAR):
        for j in range(i + 1):
            pi.append(i); pj.append(j)
    for i in range(NPAR):
        pi.append(i); pj.append(-1)
    return pi, pj


_PI, _PJ = _pair_tables()
NITEMS = len(_PI)
ITEMS_PER_THREAD = -(-NITEMS // THREADS)

KERNEL_SRC = r"""
#define NP %(NP)d
#define NPAR %(NPAR)d
#define THREADS %(THREADS)d
#define TK %(TK)d
#define NITEMS %(NITEMS)d
#define IPT %(IPT)d
#define LN2 %(LN2).17g
#define KG %(KG).17g
#define IPI %(IPI).17g

__constant__ short PAIR_I[NITEMS] = { %(PI)s };
__constant__ short PAIR_J[NITEMS] = { %(PJ)s };

// Profile and derivatives of one component at one point.
__device__ __forceinline__ void comp(const double* p, int k, double x, double* out_val, double* dj)
{
    const double a = p[1 + 4*k], c = p[2 + 4*k], nu = p[3 + 4*k], s = p[4 + 4*k];
    const double u = (x - c) / s;
    const double den = nu * KG + (1.0 - nu) * IPI;
    const double w = nu * KG / den;
    const double e = exp(-4.0 * LN2 * u * u);
    const double L = 1.0 / (1.0 + 4.0 * u * u);
    const double prof = w * e + (1.0 - w) * L;
    *out_val = a * prof;
    if (dj) {
        const double dpdu = -8.0 * u * (w * LN2 * e + (1.0 - w) * L * L);
        dj[0] = prof;
        dj[1] = -a * dpdu / s;
        dj[2] = a * (e - L) * (KG * IPI / (den * den));
        dj[3] = -a * dpdu * u / s;
    }
}

__device__ double block_sum(double v, double* red)
{
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    const double r = red[0];
    __syncthreads();
    return r;
}

// 0.5*sum(r^2) at parameters p (shared), residual only.
__device__ double cost_at(const double* p, const double* x, const double* y, int n, double* red)
{
    double acc = 0.0;
    for (int i = threadIdx.x; i < n; i += THREADS) {
        double m = p[0], v;
        for (int k = 0; k < NP; ++k) { comp(p, k, x[i], &v, 0); m += v; }
        const double r = m - y[i];
        acc += r * r;
    }
    return 0.5 * block_sum(acc, red);
}

// F, g = J^T r and A = J^T J (lower triangle mirrored) at parameters p.
__device__ double build_normal(const double* p, const double* x, const double* y, int n,
                               double* A, double* g, double* Jt, double* rt, double* mpart, double* red)
{
    const int t = threadIdx.x;
    double acc[IPT];
    for (int q = 0; q < IPT; ++q) acc[q] = 0.0;
    double sse = 0.0;
    for (int base = 0; base < n; base += TK) {
        // (point, component) work: Jacobian columns and partial model values
        for (int w = t; w < TK * NP; w += THREADS) {
            const int pt = w / NP, k = w %% NP, i = base + pt;
            double v = 0.0, dj[4] = {0.0, 0.0, 0.0, 0.0};
            if (i < n) comp(p, k, x[i], &v, dj);
            mpart[pt * NP + k] = v;
            for (int q = 0; q < 4; ++q) Jt[pt * NPAR + 1 + 4*k + q] = dj[q];
        }
        __syncthreads();
        if (t < TK) {
            const int i = base + t;
            double m = p[0];
            for (int k = 0; k < NP; ++k) m += mpart[t * NP + k];
            const double r = (i < n) ? m - y[i] : 0.0;
            rt[t] = r;
            Jt[t * NPAR] = (i < n) ? 1.0 : 0.0;
            sse += r * r;
        }
        __syncthreads();
        for (int q = 0; q < IPT; ++q) {
            const int it = t + q * THREADS;
            if (it < NITEMS) {
                const int ii = PAIR_I[it], jj = PAIR_J[it];
                double s = 0.0;
                if (jj >= 0) { for (int pt = 0; pt < TK; ++pt) s += Jt[pt*NPAR + ii] * Jt[pt*NPAR + jj]; }
                else         { for (int pt = 0; pt < TK; ++pt) s += Jt[pt*NPAR + ii] * rt[pt]; }
                acc[q] += s;
            }
        }
        __syncthreads();
    }
    for (int q = 0; q < IPT; ++q) {
        const int it = t + q * THREADS;
        if (it < NITEMS) {
            const int ii = PAIR_I[it], jj = PAIR_J[it];
            if (jj >= 0) { A[ii*NPAR + jj] = acc[q]; A[jj*NPAR + ii] = acc[q]; }
            else g[ii] = acc[q];
        }
    }
    const double F = 0.5 * block_sum(sse, red);   // block_sum syncs
    return F;
}

// In-place Cholesky of M (lower) and solve M z = b (b overwritten by z). Returns 0 on failure.
__device__ int chol_solve(double* M, double* b, int* flag)
{
    const int t = threadIdx.x;
    if (t == 0) *flag = 1;
    __syncthreads();
    for (int j = 0; j < NPAR; ++j) {
        if (t == 0) {
            const double d = M[j*NPAR + j];
            if (!(d > 0.0)) { *flag = 0; M[j*NPAR + j] = 1.0; }
            else M[j*NPAR + j] = sqrt(d);
        }
        __syncthreads();
        const double djj = M[j*NPAR + j];
        for (int i = j + 1 + t; i < NPAR; i += THREADS) M[i*NPAR + j] /= djj;
        __syncthreads();
        const int m = NPAR - j - 1;
        for (int w = t; w < m * m; w += THREADS) {
            const int i = j + 1 + w / m, k = j + 1 + w %% m;
            if (k <= i) M[i*NPAR + k] -= M[i*NPAR + j] * M[k*NPAR + j];
        }
        __syncthreads();
    }
    if (t == 0) {
        for (int i = 0; i < NPAR; ++i) {          // L z = b
            double s = b[i];
            for (int k = 0; k < i; ++k) s -= M[i*NPAR + k] * b[k];
            b[i] = s / M[i*NPAR + i];
        }
        for (int i = NPAR - 1; i >= 0; --i) {     // L^T x = z
            double s = b[i];
            for (int k = i + 1; k < NPAR; ++k) s -= M[k*NPAR + i] * b[k];
            b[i] = s / M[i*NPAR + i];
        }
    }
    __syncthreads();
    return *flag;
}

extern "C" __global__ void lm_fit(const double* __restrict__ X, const double* __restrict__ Y,
                                  const int* __restrict__ SPEC, double* P,
                                  const double* __restrict__ LB, const double* __restrict__ UB,
                                  double* SSE, int* NITER, int* STATUS,
                                  int n, int nprob, int max_iter,
                                  double ftol, double xtol, double gtol, double lam0, double theta)
{
    const int prob = blockIdx.x;
    if (prob >= nprob) return;
    const int t = threadIdx.x;
    const double* x = X + (size_t)SPEC[prob] * n;
    const double* y = Y + (size_t)SPEC[prob] * n;

    __shared__ double A[NPAR * NPAR];
    __shared__ double M[NPAR * NPAR];
    __shared__ double Jt[TK * NPAR];
    __shared__ double mpart[TK * NP];
    __shared__ double rt[TK];
    __shared__ double red[THREADS];
    __shared__ double p[NPAR], pn[NPAR], lb[NPAR], ub[NPAR], g[NPAR], d[NPAR], dv[NPAR], sh[NPAR], z[NPAR];
    __shared__ double F, lam, vfac, pred;
    __shared__ int flag, done, it;

    for (int i = t; i < NPAR; i += THREADS) {
        lb[i] = LB[prob*NPAR + i]; ub[i] = UB[prob*NPAR + i];
        const double span = ub[i] - lb[i];
        p[i] = fmin(fmax(P[prob*NPAR + i], lb[i] + 1e-10 * span), ub[i] - 1e-10 * span);
    }
    if (t == 0) { lam = lam0; vfac = 2.0; done = 0; it = 0; }
    __syncthreads();
    double Fl = build_normal(p, x, y, n, A, g, Jt, rt, mpart, red);
    if (t == 0) F = Fl;
    __syncthreads();
    int status = 1;

    while (it < max_iter) {
        // Coleman-Li scaling and scaled gradient
        for (int i = t; i < NPAR; i += THREADS) {
            double v = 1.0, dd = 0.0;
            if (g[i] < 0) { v = ub[i] - p[i]; dd = -1.0; }
            else if (g[i] > 0) { v = p[i] - lb[i]; dd = 1.0; }
            d[i] = sqrt(v); dv[i] = dd;
        }
        __syncthreads();
        if (t == 0) {
            double gmax = 0.0, dmax = 0.0;
            for (int i = 0; i < NPAR; ++i) {
                gmax = fmax(gmax, fabs(d[i] * g[i]));
                dmax = fmax(dmax, A[i*NPAR + i] * d[i] * d[i] + g[i] * dv[i]);
            }
            flag = gmax <= gtol * fmax(1.0, sqrt(2.0 * F));      // gradient convergence
            red[0] = 1e-12 * dmax + 1e-300;
        }
        __syncthreads();
        if (flag) { status = 0; break; }
        const double dfloor = red[0];
        // M = Ah + lam * diag(max(diag(Ah), floor)), rhs z = -gh
        for (int w = t; w < NPAR * NPAR; w += THREADS) {
            const int i = w / NPAR, j = w %% NPAR;
            double v = A[w] * d[i] * d[j];
            if (i == j) { v += g[i] * dv[i]; v += lam * fmax(v, dfloor); }
            M[w] = v;
        }
        for (int i = t; i < NPAR; i += THREADS) z[i] = -d[i] * g[i];
        __syncthreads();
        const int okchol = chol_solve(M, z, &flag);
        // strictly feasible component-wise truncation
        for (int i = t; i < NPAR; i += THREADS) {
            const double step = okchol ? d[i] * z[i] : 0.0;
            double q = p[i] + step;
            if (q >= ub[i]) q = p[i] + theta * (ub[i] - p[i]);
            if (q <= lb[i]) q = p[i] + theta * (lb[i] - p[i]);
            pn[i] = q;
            const double st = q - p[i];
            sh[i] = d[i] > 0.0 ? st / d[i] : 0.0;
        }
        __syncthreads();
        // predicted reduction: -gh.sh - 0.5 sh.Ah.sh
        if (t == 0) {
            double lin = 0.0, quad = 0.0;
            for (int i = 0; i < NPAR; ++i) {
                lin += d[i] * g[i] * sh[i];
                double row = 0.0;
                for (int j = 0; j < NPAR; ++j) row += A[i*NPAR + j] * d[j] * sh[j];
                quad += sh[i] * (d[i] * row + g[i] * dv[i] * sh[i]);
            }
            pred = -lin - 0.5 * quad;
        }
        __syncthreads();
        const double Fn = cost_at(pn, x, y, n, red);
        if (t == 0) {
            const double rho = (F - Fn) / (pred > 0.0 ? pred : 1e308);
            const int ok = okchol && (Fn < F) && (rho > 0.0);
            const double rel = (F - Fn) / fmax(F, 1e-300);
            double smax = 0.0;
            for (int i = 0; i < NPAR; ++i) smax = fmax(smax, fabs(pn[i] - p[i]) / (fabs(p[i]) + 1e-12));
            done = (ok && (rel <= ftol || smax <= xtol)) || (lam > 1e16);
            flag = ok;
            if (ok) {
                lam *= fmax(1.0 / 3.0, 1.0 - pow(2.0 * rho - 1.0, 3.0));
                vfac = 2.0;
                F = Fn;
            } else {
                lam *= vfac; vfac *= 2.0;
            }
            it += 1;
        }
        __syncthreads();
        if (flag) {
            for (int i = t; i < NPAR; i += THREADS) p[i] = pn[i];
            __syncthreads();
            Fl = build_normal(p, x, y, n, A, g, Jt, rt, mpart, red);
            if (t == 0) F = Fl;
            __syncthreads();
        }
        if (done) { status = 0; break; }
    }
    for (int i = t; i < NPAR; i += THREADS) P[prob*NPAR + i] = p[i];
    if (t == 0) { SSE[prob] = 2.0 * F; NITER[prob] = it; STATUS[prob] = status; }
}
""" % dict(NP=NP, NPAR=NPAR, THREADS=THREADS, TK=TK, NITEMS=NITEMS, IPT=ITEMS_PER_THREAD,
           LN2=LN2, KG=K_GAUSS, IPI=INV_PI, PI=", ".join(map(str, _PI)), PJ=", ".join(map(str, _PJ)))

_KERNELS = {}


def _kernel(cp):
    dev = cp.cuda.Device().id
    if dev not in _KERNELS:
        _KERNELS[dev] = cp.RawKernel(KERNEL_SRC, "lm_fit", options=("--std=c++11",))
    return _KERNELS[dev]


def fit_problems(x, y, spec, p0, lb, ub, device=0, **opts):
    """Fit problems on one GPU. x, y: (S, n) spectra; spec: (B,) spectrum index per problem."""
    from .lm_cpu import DEFAULTS
    from .pipeline import _preload_cuda_libs
    _preload_cuda_libs()
    import cupy as cp
    o = {**DEFAULTS, **opts}
    with cp.cuda.Device(device):
        X = cp.asarray(x, dtype=cp.float64); Y = cp.asarray(y, dtype=cp.float64)
        SPEC = cp.asarray(spec, dtype=cp.int32)
        P = cp.asarray(p0, dtype=cp.float64).copy()
        LB = cp.asarray(lb, dtype=cp.float64); UB = cp.asarray(ub, dtype=cp.float64)
        B = P.shape[0]
        SSE = cp.empty(B, dtype=cp.float64)
        NIT = cp.empty(B, dtype=cp.int32); ST = cp.empty(B, dtype=cp.int32)
        _kernel(cp)((B,), (THREADS,), (X, Y, SPEC, P, LB, UB, SSE, NIT, ST,
                                       np.int32(x.shape[1]), np.int32(B), np.int32(o["max_iter"]),
                                       np.float64(o["ftol"]), np.float64(o["xtol"]), np.float64(o["gtol"]),
                                       np.float64(o["lam0"]), np.float64(o["theta"])))
        cp.cuda.Device(device).synchronize()
        return cp.asnumpy(P), cp.asnumpy(SSE), cp.asnumpy(NIT), cp.asnumpy(ST)
