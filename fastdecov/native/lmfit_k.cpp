// Native CPU solver for a general basis (runtime number of lines K and background terms NBG).
// Same algorithm and stopping rules as lmfit.cpp / lm_cpu.fit_batch; lmfit.cpp stays the fixed
// 12-line engine used by the cross-language benchmark.
// One OpenMP thread per fit; analytic Jacobian stored column-major so J^T J is a set of
// contiguous dot products the compiler vectorises.
//
// Build: g++ -O3 -march=native -ffp-contract=fast -fopenmp -shared -fPIC lmfit.cpp -o liblmfit.so -lmvec
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <omp.h>

// Let the vectoriser use glibc's libmvec exp (_ZGVdN4v_exp, <= 4 ulp) without -ffast-math,
// so the rest of the solver keeps strict IEEE semantics (NaN checks, no reassociation).
extern "C" double exp(double) noexcept __attribute__((simd("notinbranch")));

namespace {
const double XREF = 700.0, XSCALE = 100.0;
const double LN2 = std::log(2.0);
const double KG = std::sqrt(std::log(2.0) / M_PI);
const double IPI = 1.0 / M_PI;

struct Work {
    int n, NP, NBG, NPAR;
    std::vector<double> J;     // NPAR x n, column-major per parameter
    std::vector<double> r, t;  // n
    std::vector<double> A, M, buf;
    Work(int n_, int np_, int nbg_) : n(n_), NP(np_), NBG(nbg_), NPAR(nbg_ + 4 * np_), J((size_t)(nbg_ + 4 * np_) * n_), r(n_), t(n_),
        A((size_t)NPAR * NPAR), M((size_t)NPAR * NPAR), buf(9 * (size_t)NPAR) {}
};

// residual (and Jacobian if want_j); returns 0.5*sum r^2
double residual(const double* p, const double* __restrict x, const double* __restrict y, Work& w, bool want_j) {
    const int n = w.n;
    double* __restrict r = w.r.data();
    const int NP = w.NP, NBG = w.NBG;
    for (int i = 0; i < n; ++i) { w.t[i] = (x[i] - XREF) / XSCALE; r[i] = p[0] - y[i] + (NBG == 2 ? p[1] * w.t[i] : 0.0); }
    if (want_j) {
        std::fill(w.J.begin(), w.J.begin() + n, 1.0);
        if (NBG == 2) std::copy(w.t.begin(), w.t.end(), w.J.begin() + n);
    }
    for (int k = 0; k < NP; ++k) {
        const double a = p[NBG + 4 * k], c = p[NBG + 1 + 4 * k], nu = p[NBG + 2 + 4 * k], s = p[NBG + 3 + 4 * k];
        const double den = nu * KG + (1.0 - nu) * IPI;
        const double wg = nu * KG / den, wl = 1.0 - wg, dw = KG * IPI / (den * den), inv_s = 1.0 / s;
        const double g4 = -4.0 * LN2;
        if (want_j) {
            double* __restrict Ja = w.J.data() + (size_t)(NBG + 4 * k) * n;
            double* __restrict Jc = Ja + n;
            double* __restrict Jn = Jc + n;
            double* __restrict Js = Jn + n;
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                const double u = (x[i] - c) * inv_s, u2 = u * u;
                const double e = ::exp(g4 * u2);
                const double L = 1.0 / (1.0 + 4.0 * u2);
                const double prof = wg * e + wl * L;
                const double dpdu = -8.0 * u * (wg * LN2 * e + wl * L * L);
                r[i] += a * prof;
                Ja[i] = prof;
                Jc[i] = -a * dpdu * inv_s;
                Jn[i] = a * (e - L) * dw;
                Js[i] = -a * dpdu * u * inv_s;
            }
        } else {
            #pragma omp simd
            for (int i = 0; i < n; ++i) {
                const double u = (x[i] - c) * inv_s, u2 = u * u;
                r[i] += a * (wg * ::exp(g4 * u2) + wl / (1.0 + 4.0 * u2));
            }
        }
    }
    double F = 0.0;
    #pragma omp simd reduction(+:F)
    for (int i = 0; i < n; ++i) F += r[i] * r[i];
    return 0.5 * F;
}

void normal_eq(const Work& w, double* A, double* g) {
    // J^T J in 4x4 register blocks (lower triangle, mirrored) + J^T r.
    const int n = w.n, NPAR = w.NPAR;
    const double* J = w.J.data();
    const double* r = w.r.data();
    for (int a = 0; a < NPAR; ++a) {
        const double* Ja = J + (size_t)a * n;
        double s = 0.0;
        #pragma omp simd reduction(+:s)
        for (int i = 0; i < n; ++i) s += Ja[i] * r[i];
        g[a] = s;
    }
    constexpr int BS = 4;
    for (int a0 = 0; a0 < NPAR; a0 += BS) {
        const int na = std::min(BS, NPAR - a0);
        for (int b0 = 0; b0 <= a0; b0 += BS) {
            const int nb = std::min(BS, NPAR - b0);
            const double* Ja[BS]; const double* Jb[BS];
            for (int q = 0; q < BS; ++q) {
                Ja[q] = J + (size_t)(a0 + std::min(q, na - 1)) * n;
                Jb[q] = J + (size_t)(b0 + std::min(q, nb - 1)) * n;
            }
            double s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0;
            double s20 = 0, s21 = 0, s22 = 0, s23 = 0, s30 = 0, s31 = 0, s32 = 0, s33 = 0;
            const double *A0 = Ja[0], *A1 = Ja[1], *A2 = Ja[2], *A3 = Ja[3];
            const double *B0 = Jb[0], *B1 = Jb[1], *B2 = Jb[2], *B3 = Jb[3];
            #pragma omp simd reduction(+:s00,s01,s02,s03,s10,s11,s12,s13,s20,s21,s22,s23,s30,s31,s32,s33)
            for (int i = 0; i < n; ++i) {
                const double x0 = A0[i], x1 = A1[i], x2 = A2[i], x3 = A3[i];
                const double y0 = B0[i], y1 = B1[i], y2 = B2[i], y3 = B3[i];
                s00 += x0 * y0; s01 += x0 * y1; s02 += x0 * y2; s03 += x0 * y3;
                s10 += x1 * y0; s11 += x1 * y1; s12 += x1 * y2; s13 += x1 * y3;
                s20 += x2 * y0; s21 += x2 * y1; s22 += x2 * y2; s23 += x2 * y3;
                s30 += x3 * y0; s31 += x3 * y1; s32 += x3 * y2; s33 += x3 * y3;
            }
            const double acc[BS][BS] = {{s00, s01, s02, s03}, {s10, s11, s12, s13}, {s20, s21, s22, s23}, {s30, s31, s32, s33}};
            for (int qa = 0; qa < na; ++qa)
                for (int qb = 0; qb < nb; ++qb) {
                    const int ia = a0 + qa, ib = b0 + qb;
                    if (ib <= ia) A[ia * NPAR + ib] = A[ib * NPAR + ia] = acc[qa][qb];
                }
        }
    }
}

// Cholesky solve of M z = b in place; returns false if not positive definite.
bool chol_solve(double* M, double* b, int NPAR) {
    for (int j = 0; j < NPAR; ++j) {
        double d = M[j * NPAR + j];
        for (int k = 0; k < j; ++k) d -= M[j * NPAR + k] * M[j * NPAR + k];
        if (!(d > 0.0)) return false;
        d = std::sqrt(d);
        M[j * NPAR + j] = d;
        for (int i = j + 1; i < NPAR; ++i) {
            double s = M[i * NPAR + j];
            for (int k = 0; k < j; ++k) s -= M[i * NPAR + k] * M[j * NPAR + k];
            M[i * NPAR + j] = s / d;
        }
    }
    for (int i = 0; i < NPAR; ++i) {
        double s = b[i];
        for (int k = 0; k < i; ++k) s -= M[i * NPAR + k] * b[k];
        b[i] = s / M[i * NPAR + i];
    }
    for (int i = NPAR - 1; i >= 0; --i) {
        double s = b[i];
        for (int k = i + 1; k < NPAR; ++k) s -= M[k * NPAR + i] * b[k];
        b[i] = s / M[i * NPAR + i];
    }
    return true;
}

void fit_one(const double* x, const double* y, int n, double* P, const double* LB, const double* UB,
             double* SSE, int* NIT, int* ST, int max_iter, double ftol, double xtol, double gtol,
             double lam0, double theta, Work& w) {
    const int NPAR = w.NPAR;
    double* const b9 = w.buf.data();
    double *p = b9, *pn = b9 + NPAR, *lb = b9 + 2 * NPAR, *ub = b9 + 3 * NPAR, *g = b9 + 4 * NPAR, *d = b9 + 5 * NPAR,
           *dv = b9 + 6 * NPAR, *sh = b9 + 7 * NPAR, *z = b9 + 8 * NPAR;
    double* A = w.A.data(); double* M = w.M.data();
    for (int i = 0; i < NPAR; ++i) {
        lb[i] = LB[i]; ub[i] = UB[i];
        const double span = ub[i] - lb[i];
        p[i] = std::min(std::max(P[i], lb[i] + 1e-10 * span), ub[i] - 1e-10 * span);
    }
    double F = residual(p, x, y, w, true);
    normal_eq(w, A, g);
    double lam = lam0, vfac = 2.0;
    int it = 0, status = 1;
    while (it < max_iter) {
        double gmax = 0.0, dmax = 0.0;
        for (int i = 0; i < NPAR; ++i) {
            double v = 1.0, dd = 0.0;
            if (g[i] < 0) { v = ub[i] - p[i]; dd = -1.0; }
            else if (g[i] > 0) { v = p[i] - lb[i]; dd = 1.0; }
            d[i] = std::sqrt(v); dv[i] = dd;
            gmax = std::max(gmax, std::fabs(d[i] * g[i]));
            dmax = std::max(dmax, A[i * NPAR + i] * d[i] * d[i] + g[i] * dd);
        }
        if (gmax <= gtol * std::max(1.0, std::sqrt(2.0 * F))) { status = 0; break; }
        const double dfloor = 1e-12 * dmax + 1e-300;
        for (int i = 0; i < NPAR; ++i) {
            for (int j = 0; j < NPAR; ++j) M[i * NPAR + j] = A[i * NPAR + j] * d[i] * d[j];
            double v = M[i * NPAR + i] + g[i] * dv[i];
            M[i * NPAR + i] = v + lam * std::max(v, dfloor);
            z[i] = -d[i] * g[i];
        }
        const bool okchol = chol_solve(M, z, NPAR);
        for (int i = 0; i < NPAR; ++i) {
            const double step = okchol ? d[i] * z[i] : 0.0;
            double q = p[i] + step;
            if (q >= ub[i]) q = p[i] + theta * (ub[i] - p[i]);
            if (q <= lb[i]) q = p[i] + theta * (lb[i] - p[i]);
            pn[i] = q;
            sh[i] = d[i] > 0.0 ? (q - p[i]) / d[i] : 0.0;
        }
        double lin = 0.0, quad = 0.0;
        for (int i = 0; i < NPAR; ++i) {
            lin += d[i] * g[i] * sh[i];
            double row = 0.0;
            for (int j = 0; j < NPAR; ++j) row += A[i * NPAR + j] * d[j] * sh[j];
            quad += sh[i] * (d[i] * row + g[i] * dv[i] * sh[i]);
        }
        const double pred = -lin - 0.5 * quad;
        const double Fn = residual(pn, x, y, w, false);
        const double rho = (F - Fn) / (pred > 0.0 ? pred : 1e308);
        const bool ok = okchol && (Fn < F) && (rho > 0.0);
        const double rel = (F - Fn) / std::max(F, 1e-300);
        double smax = 0.0;
        for (int i = 0; i < NPAR; ++i) smax = std::max(smax, std::fabs(pn[i] - p[i]) / (std::fabs(p[i]) + 1e-12));
        const bool done = (ok && (rel <= ftol || smax <= xtol)) || (lam > 1e16);
        ++it;
        if (ok) {
            lam *= std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3.0));
            vfac = 2.0;
            std::memcpy(p, pn, sizeof(double) * NPAR);
            F = residual(p, x, y, w, true);
            normal_eq(w, A, g);
        } else {
            lam *= vfac; vfac *= 2.0;
        }
        if (done) { status = 0; break; }
    }
    std::memcpy(P, p, sizeof(double) * NPAR);
    *SSE = 2.0 * F; *NIT = it; *ST = status;
}
}  // namespace

extern "C" void lmfit_k_batch(const double* X, const double* Y, const int* SPEC, double* P,
                              const double* LB, const double* UB, double* SSE, int* NIT, int* ST,
                              int n, int nprob, int K, int nbg, int max_iter, double ftol, double xtol, double gtol,
                              double lam0, double theta, int nthreads) {
    if (nthreads > 0) omp_set_num_threads(nthreads);
    const int NPAR = nbg + 4 * K;
    #pragma omp parallel
    {
        Work w(n, K, nbg);
        #pragma omp for schedule(dynamic, 1)
        for (int b = 0; b < nprob; ++b) {
            const size_t s = (size_t)SPEC[b] * n;
            fit_one(X + s, Y + s, n, P + (size_t)b * NPAR, LB + (size_t)b * NPAR, UB + (size_t)b * NPAR,
                    SSE + b, NIT + b, ST + b, max_iter, ftol, xtol, gtol, lam0, theta, w);
        }
    }
}
