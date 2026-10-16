# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""ctypes binding to the optional C++/OpenMP solver (fastdecov/native/lmfit.cpp).

Same algorithm and stopping rules as lm_cpu.fit_batch. The shared library is built on first use
with the system C++ compiler (hw.compiler, hw.compile_flags) and cached next to the source; if no compiler is
available, prebuilt libraries in the build directory are used as they are (as in the Docker image), and otherwise
available() returns False and callers fall back to the NumPy solver.
"""
import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

from . import hw
from .model import NPAR

SRC = Path(__file__).with_name("native") / "lmfit.cpp"


def _libdir():
    """Where compiled solvers go: $FASTDECOV_BUILD_DIR, else the package's native/ if writable, else ~/.cache/fastdecov."""
    d = os.environ.get("FASTDECOV_BUILD_DIR")
    if d:
        p = Path(d)
    elif os.access(SRC.parent, os.W_OK):
        p = SRC.parent
    else:
        p = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "fastdecov"
    p.mkdir(parents=True, exist_ok=True)
    return p


LIB = _libdir() / "liblmfit.so"
_lib = None


def _build(src, lib, force=False):
    """Compile `src` into `lib` unless an up-to-date `lib` exists. Without a compiler an existing (prebuilt) `lib` is
    used even if older than the source, so a compiler-free install (the Docker image) never tries to rebuild."""
    fresh = lib.exists() and lib.stat().st_mtime >= src.stat().st_mtime
    if fresh and not force:
        return lib
    cxx = hw.compiler()
    if not cxx:
        if lib.exists() and not force:
            return lib
        raise RuntimeError("no C++ compiler found: install g++ (macOS: brew install gcc), set CXX, or run the Docker "
                           "image (see README, 'Run anywhere with Docker')")
    flags, libs = hw.compile_flags()
    tmp = lib.with_suffix(f".{os.getpid()}.tmp")
    r = subprocess.run([cxx, *flags, str(src), "-o", str(tmp), *libs], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"building {src.name} with {cxx} failed: {r.stderr.strip()[-600:]}")
    os.replace(tmp, lib)
    return lib


def build(force=False):
    return _build(SRC, LIB, force)


def build_all(force=False):
    """Compile all three solvers without loading them (an image build may target a CPU it cannot run)."""
    return [_build(SRC, LIB, force), _build(SRC_K, LIB_K, force), _build(SRC_ZP, LIB_ZP, force)]


def load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(build()))
        d, i = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        lib.lmfit_batch.argtypes = [d, d, i, d, d, d, d, i, i] + [ctypes.c_int] * 3 + [ctypes.c_double] * 5 + [ctypes.c_int]
        lib.lmfit_batch.restype = None
        if lib.lmfit_npar() != NPAR:
            raise RuntimeError("native solver parameter count mismatch")
        _lib = lib
    return _lib


def available():
    try:
        load()
        return True
    except Exception:
        return False


def fit_problems(x, y, spec, p0, lb, ub, threads=0, **opts):
    """Same interface as lm_cuda.fit_problems. threads=0 uses all OpenMP threads."""
    from .lm_cpu import DEFAULTS
    o = {**DEFAULTS, **opts}
    lib = load()
    c = lambda a, t=np.float64: np.ascontiguousarray(a, dtype=t)
    X, Y, LB, UB = c(x), c(y), c(lb), c(ub)
    P = c(p0).copy()
    SPEC = c(spec, np.int32)
    B = P.shape[0]
    SSE = np.empty(B); NIT = np.empty(B, np.int32); ST = np.empty(B, np.int32)
    ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double if a.dtype == np.float64 else ctypes.c_int))
    lib.lmfit_batch(ptr(X), ptr(Y), ptr(SPEC), ptr(P), ptr(LB), ptr(UB), ptr(SSE), ptr(NIT), ptr(ST),
                    X.shape[1], B, int(o["max_iter"]), float(o["ftol"]), float(o["xtol"]), float(o["gtol"]),
                    float(o["lam0"]), float(o["theta"]), int(threads))
    return P, SSE, NIT, ST


# ---------------------------------------------------------------- general basis engine (lmfit_k.cpp)
SRC_K = SRC.with_name("lmfit_k.cpp")
LIB_K = LIB.with_name("liblmfit_k.so")
_lib_k = None


def load_k():
    global _lib_k
    if _lib_k is None:
        lib = ctypes.CDLL(str(_build(SRC_K, LIB_K)))
        d, i = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        lib.lmfit_k_batch.argtypes = [d, d, i, d, d, d, d, i, i] + [ctypes.c_int] * 5 + [ctypes.c_double] * 5 + [ctypes.c_int]
        lib.lmfit_k_batch.restype = None
        _lib_k = lib
    return _lib_k


def fit_problems_k(x, y, spec, p0, lb, ub, K, nbg, threads=0, **opts):
    """General-basis fits: parameter vectors of length nbg + 4K (see model.Basis)."""
    from .lm_cpu import DEFAULTS
    o = {**DEFAULTS, **opts}
    lib = load_k()
    c = lambda a, t=np.float64: np.ascontiguousarray(a, dtype=t)
    X, Y, LB, UB = c(x), c(y), c(lb), c(ub)
    P = c(p0).copy(); SPEC = c(spec, np.int32)
    assert P.shape[1] == nbg + 4 * K
    B = P.shape[0]
    SSE = np.empty(B); NIT = np.empty(B, np.int32); ST = np.empty(B, np.int32)
    ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double if a.dtype == np.float64 else ctypes.c_int))
    lib.lmfit_k_batch(ptr(X), ptr(Y), ptr(SPEC), ptr(P), ptr(LB), ptr(UB), ptr(SSE), ptr(NIT), ptr(ST),
                      X.shape[1], B, int(K), int(nbg), int(o["max_iter"]), float(o["ftol"]), float(o["xtol"]),
                      float(o["gtol"]), float(o["lam0"]), float(o["theta"]), int(threads))
    return P, SSE, NIT, ST


# ---------------------------------------------------------------- V3 emitter engine (lmfit_zp.cpp)
SRC_ZP = SRC.with_name("lmfit_zp.cpp")
LIB_ZP = LIB.with_name("liblmfit_zp.so")
_lib_zp = None


def load_zp():
    global _lib_zp
    if _lib_zp is None:
        lib = ctypes.CDLL(str(_build(SRC_ZP, LIB_ZP)))
        d, i = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)
        lib.lmfit_zp_batch.argtypes = [d, d, i, d, d, d, d, i, i] + [ctypes.c_int] * 3 + [i] + [ctypes.c_int] + \
            [ctypes.c_double] * 5 + [ctypes.c_int]
        lib.lmfit_zp_batch.restype = None
        _lib_zp = lib
    return _lib_zp


def fit_problems_zp(x, y, spec, p0, lb, ub, psb, threads=0, **opts):
    """V3 emitter fits. psb: per-emitter sideband flags (shared by all problems); NPAR = 2 + sum(4 + 3*psb)."""
    from .lm_cpu import DEFAULTS
    o = {**DEFAULTS, **opts}
    lib = load_zp()
    c = lambda a, t=np.float64: np.ascontiguousarray(a, dtype=t)
    X, Y, LB, UB = c(x), c(y), c(lb), c(ub)
    P = c(p0).copy(); SPEC = c(spec, np.int32); F = c(psb, np.int32)
    assert P.shape[1] == 2 + int(np.sum(4 + 3 * F))
    B = P.shape[0]
    SSE = np.empty(B); NIT = np.empty(B, np.int32); ST = np.empty(B, np.int32)
    ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_double if a.dtype == np.float64 else ctypes.c_int))
    lib.lmfit_zp_batch(ptr(X), ptr(Y), ptr(SPEC), ptr(P), ptr(LB), ptr(UB), ptr(SSE), ptr(NIT), ptr(ST),
                       X.shape[1], B, int(F.size), ptr(F), int(o["max_iter"]), float(o["ftol"]), float(o["xtol"]),
                       float(o["gtol"]), float(o["lam0"]), float(o["theta"]), int(threads))
    return P, SSE, NIT, ST
