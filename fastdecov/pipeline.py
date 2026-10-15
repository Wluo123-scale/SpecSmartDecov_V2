# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Batch driver: multi-start fitting of many spectra on all CPU cores or on one or more GPUs."""
import ctypes
import functools
import glob
import multiprocessing
import os
import site
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from . import model

# Default starting widths (nm): MATLAB-like box centre of the legacy width range, plus two
# physically typical widths. The fit with the lowest SSE is kept.
DEFAULT_STARTS = ("mid", 8.0, 5.0)
BLAS_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
CPU_CHUNK = 24  # problems per CPU task; small enough to balance, large enough to vectorise


@dataclass
class FitResults:
    ids: list
    params: np.ndarray          # (S, NPAR)
    sse: np.ndarray             # (S,)
    n_iter: np.ndarray          # (S,) iterations of the selected start
    converged: np.ndarray       # (S,) bool
    best_start: list
    device: str
    meta: dict = field(default_factory=dict)
    K: int = 12                 # number of basis lines
    nbg: int = 1                # background terms: 1 constant, 2 linear
    basis: object = None        # model.Basis (None = legacy 12-anchor basis)

    @property
    def areas(self):
        return model.areas_k(self.params, self.K, self.nbg)

    def table(self, s):
        """(K, 4) array [centre, nu, width, area] for spectrum index s."""
        pk = self.params[s, self.nbg:].reshape(self.K, 4)
        return np.column_stack([pk[:, 1], pk[:, 2], pk[:, 3], self.areas[s]])


# ----------------------------------------------------------------------------- devices
@functools.lru_cache(maxsize=None)
def _preload_cuda_libs():
    """Make pip-installed nvidia-* runtime/NVRTC libraries visible to CuPy (once per process)."""
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for pattern in ("nvidia/cuda_runtime/lib/libcudart.so.*", "nvidia/cuda_nvrtc/lib/libnvrtc.so.*",
                        "nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.*"):
            for lib in sorted(glob.glob(os.path.join(sp, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def probe_gpus():
    """Return [(device_id, name)] for GPUs that pass a CUDA JIT smoke test; raise on failure."""
    _preload_cuda_libs()
    import cupy as cp
    n = cp.cuda.runtime.getDeviceCount()
    if n < 1:
        raise RuntimeError("no CUDA devices reported")
    out = []
    src = 'extern "C" __global__ void probe(double* v) { v[0] = 7.0; }'
    for dev in range(n):
        with cp.cuda.Device(dev):
            v = cp.zeros(1, dtype=cp.float64)
            cp.RawKernel(src, "probe")((1,), (1,), (v,))
            cp.cuda.Device(dev).synchronize()
            if float(v.get()[0]) != 7.0:
                raise RuntimeError(f"CUDA JIT probe returned a wrong value on device {dev}")
            name = cp.cuda.runtime.getDeviceProperties(dev)["name"]
            out.append((dev, name.decode() if isinstance(name, bytes) else str(name)))
    return out


def resolve_device(requested, gpus=None):
    """cpu -> cpu; gpu -> verified CUDA or RuntimeError; auto -> gpu if verified else cpu."""
    if requested == "cpu":
        return "cpu", []
    try:
        found = probe_gpus()
        if gpus is not None:
            found = [g for g in found if g[0] in set(gpus)]
            if not found:
                raise RuntimeError(f"requested GPUs {gpus} not available")
    except Exception as err:
        if requested == "gpu":
            raise RuntimeError(f"--device gpu requested but no usable CUDA device: {err}") from err
        return "cpu", []
    return "gpu", found


# ----------------------------------------------------------------------------- problems
def build_problems(xs, ys, starts, seeds=None):
    """One problem per (spectrum, start), plus one per seeded spectrum (seeds[i] = (p0, lb, ub)).
    Returns per-problem arrays, spectrum index and start index (len(starts) marks a seed)."""
    p0, lb, ub, spec, sidx = [], [], [], [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        for k, ws in enumerate(starts):
            a, b, c = model.setup(x, y, width_start=ws)
            p0.append(a); lb.append(b); ub.append(c); spec.append(i); sidx.append(k)
        if seeds is not None and seeds[i] is not None:
            a, b, c = seeds[i]
            p0.append(a); lb.append(b); ub.append(c); spec.append(i); sidx.append(len(starts))
    return np.array(p0), np.array(lb), np.array(ub), np.array(spec), np.array(sidx)


def _cpu_task(args):
    from .lm_cpu import fit_batch
    x, y, p0, lb, ub, opts = args
    return fit_batch(p0, lb, ub, x, y, **opts)


def _run_cpu(x, y, spec, p0, lb, ub, workers, opts):
    B = len(p0)
    order = np.arange(B)
    chunks = [order[i:i + CPU_CHUNK] for i in range(0, B, CPU_CHUNK)]
    tasks = [(x[spec[c]], y[spec[c]], p0[c], lb[c], ub[c], opts) for c in chunks]
    P = np.empty_like(p0); S = np.empty(B); N = np.empty(B, int); ST = np.empty(B, int)
    if workers <= 1 or len(tasks) == 1:
        outs = list(map(_cpu_task, tasks))
    else:
        # spawned workers import NumPy fresh, so one BLAS thread each (no oversubscription)
        for var in BLAS_THREAD_VARS:
            os.environ[var] = "1"
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)),
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            outs = list(pool.map(_cpu_task, tasks))
    for c, (pp, ss, nn, st) in zip(chunks, outs):
        P[c], S[c], N[c], ST[c] = pp, ss, nn, st
    return P, S, N, ST


def _run_native(x, y, spec, p0, lb, ub, workers, opts):
    from .lm_native import fit_problems
    return fit_problems(x, y, spec, p0, lb, ub, threads=workers, **opts)


def resolve_engine(engine):
    """CPU engine: 'native' (C++/OpenMP, built with the system compiler) or 'numpy'.
    'auto' picks native when it builds and loads, otherwise NumPy."""
    if engine == "numpy":
        return "numpy"
    from . import lm_native
    if lm_native.available():
        return "native"
    if engine == "native":
        raise RuntimeError("--engine native requested but the C++ solver could not be built/loaded")
    return "numpy"


def _run_gpu(x, y, spec, p0, lb, ub, devices, opts):
    from .lm_cuda import fit_problems
    B = len(p0)
    parts = np.array_split(np.arange(B), len(devices))
    P = np.empty_like(p0); S = np.empty(B); N = np.empty(B, int); ST = np.empty(B, int)

    def job(dev, idx):
        return idx, fit_problems(x, y, spec[idx], p0[idx], lb[idx], ub[idx], device=dev, **opts)

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        for idx, (pp, ss, nn, st) in ex.map(lambda a: job(*a), [(d, i) for d, i in zip(devices, parts) if len(i)]):
            P[idx], S[idx], N[idx], ST[idx] = pp, ss, nn, st
    return P, S, N, ST


# ----------------------------------------------------------------------------- public API
def fit_spectra(ids, xs, ys, device="cpu", gpu_devices=(), workers=None, starts=DEFAULT_STARTS,
                seeds=None, engine="numpy", **opts):
    """Fit spectra (lists of 1-D arrays). Spectra are grouped by grid length for batching.

    seeds: optional list (aligned with ids) of (p0, lb, ub) or None -- an extra problem per
    spectrum, e.g. a legacy MATLAB solution to refine. With starts=() only seeds are fitted.
    """
    workers = workers or len(os.sched_getaffinity(0))
    S = len(ids)
    params = np.empty((S, model.NPAR)); sse = np.full(S, np.inf)
    n_iter = np.zeros(S, int); conv = np.zeros(S, bool); best = [None] * S
    lengths = np.array([len(x) for x in xs])
    for n in np.unique(lengths):
        members = np.flatnonzero(lengths == n)
        x = np.stack([xs[i] for i in members]); y = np.stack([ys[i] for i in members])
        p0, lb, ub, spec, sidx = build_problems(x, y, starts, None if seeds is None else [seeds[i] for i in members])
        if device == "gpu":
            P, SS, N, ST = _run_gpu(x, y, spec, p0, lb, ub, list(gpu_devices), opts)
        elif engine == "native":
            P, SS, N, ST = _run_native(x, y, spec, p0, lb, ub, workers, opts)
        else:
            P, SS, N, ST = _run_cpu(x, y, spec, p0, lb, ub, workers, opts)
        SS = np.where(np.isfinite(SS), SS, np.inf)
        for j, m in enumerate(members):
            cand = np.flatnonzero(spec == j)
            k = cand[np.argmin(SS[cand])]
            params[m], sse[m], n_iter[m], conv[m] = P[k], SS[k], N[k], ST[k] == 0
            best[m] = starts[sidx[k]] if sidx[k] < len(starts) else "seed"
    return FitResults(list(ids), params, sse, n_iter, conv, best, device,
                      meta=dict(starts=list(starts), workers=workers if device == "cpu" else None,
                                engine=engine if device == "cpu" else "cuda"))


def fit_spectra_basis(ids, xs, ys, basis, workers=None, starts=DEFAULT_STARTS, engine="auto", **opts):
    """Fit spectra with an arbitrary model.Basis (K lines, constant or linear background) on CPU.

    The legacy 12-anchor basis should go through fit_spectra(), which keeps past results bit-exact.
    """
    workers = workers or len(os.sched_getaffinity(0))
    use_native = engine != "numpy"
    if use_native:
        from . import lm_native
        try:
            lm_native.load_k()
        except Exception:
            if engine == "native":
                raise
            use_native = False
    S, K, nbg = len(ids), basis.K, basis.nbg
    params = np.empty((S, basis.npar)); sse = np.full(S, np.inf)
    n_iter = np.zeros(S, int); conv = np.zeros(S, bool); best = [None] * S
    lengths = np.array([len(x) for x in xs])
    for n in np.unique(lengths):
        members = np.flatnonzero(lengths == n)
        x = np.stack([xs[i] for i in members]); y = np.stack([ys[i] for i in members])
        p0, lb, ub, spec, sidx = [], [], [], [], []
        for j in range(len(members)):
            for k, ws in enumerate(starts):
                a, b, c = model.setup_basis(x[j], y[j], basis, width_start=ws)
                p0.append(a); lb.append(b); ub.append(c); spec.append(j); sidx.append(k)
        p0, lb, ub, spec, sidx = map(np.array, (p0, lb, ub, spec, sidx))
        if use_native:
            P, SS, N, ST = lm_native.fit_problems_k(x, y, spec, p0, lb, ub, K, nbg, threads=workers, **opts)
        else:
            P, SS, N, ST = _run_cpu(x, y, spec, p0, lb, ub, workers, dict(opts, nbg=nbg))
        SS = np.where(np.isfinite(SS), SS, np.inf)
        for j, m in enumerate(members):
            cand = np.flatnonzero(spec == j)
            k = cand[np.argmin(SS[cand])]
            params[m], sse[m], n_iter[m], conv[m] = P[k], SS[k], N[k], ST[k] == 0
            best[m] = starts[sidx[k]]
    return FitResults(list(ids), params, sse, n_iter, conv, best, "cpu",
                      meta=dict(starts=list(starts), workers=workers, engine="native" if use_native else "numpy",
                                basis=basis.name), K=K, nbg=nbg, basis=basis)
