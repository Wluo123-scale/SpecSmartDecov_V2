"""Python SDK.

    from fastdecov import api
    m = api.load_map("path/to/spectra")                      # directory of <id>.txt (wavelength nm, intensity)
    res = api.fit_map(m, method="v3", threads=8)             # "v3" (default), "v2" or "legacy"
    print(res.K, res.psb, [res.median(0, k) for k in range(res.K)])
    api.save(res, "out_dir", figures=True, grid="auto")      # dc tables, emitters.tsv, report, maps, .tif figures

Methods
  v3      physical emitters: zero-phonon line (pseudo-Voigt) + optional tied red phonon sideband; the number of
          emitters, their centres and the sidebands are discovered from the map (fastdecov.basis3); no line splitting
  v2      data-driven global basis of pseudo-Voigt lines (fastdecov.basis.grow_basis), best-fit oriented
  legacy  the original SpecSmartDecov model: 12 fixed anchors (611 ... 724 nm), constant background
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import basis as B, basis3, emitter, grid as G, model, physics, report

METHODS = ("v3", "v2", "legacy")


@dataclass
class SpectrumMap:
    ids: list
    xs: list
    ys: list

    def __len__(self):
        return len(self.ids)


def load_map(path, pattern="*.txt"):
    """Read every spectrum file in a directory (sorted). Each file: two columns, wavelength (nm) and intensity."""
    paths = sorted(Path(path).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no {pattern} spectra in {path}")
    xs, ys = map(list, zip(*[model.load_spectrum(p) for p in paths]))
    return SpectrumMap([p.stem for p in paths], xs, ys)


@dataclass
class Result:
    """A fitted map in the emitter layout: params[s] = [b0, b1, emitter_0, ...] (see fastdecov.emitter)."""
    method: str
    ids: list
    xs: list
    ys: list
    keeps: list                    # per spectrum: pixels used (False = masked cosmic ray)
    params: np.ndarray             # (S, NPAR)
    sse: np.ndarray
    psb: np.ndarray                # per emitter: phonon sideband flag
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.table = np.array([emitter.table(p, self.psb) for p in self.params])   # (S, K, 8)
        tot = self.table[..., 3] + self.table[..., 4]
        self.share = self.table[..., 3] / np.maximum(tot.sum(1, keepdims=True), 1e-300)
        off, _ = emitter.offsets(self.psb)
        self.heights = self.params[:, off]
        self.sigma = np.array([B.noise_sigma(y) for y in self.ys])

    @property
    def K(self):
        return int(len(self.psb))

    def median(self, col, k, strong=0.02):
        """Median of table column `col` for emitter k over spectra where its ZPL carries >= `strong` of line area."""
        m = self.share[:, k] >= strong
        return float(np.median(self.table[m, k, col] if m.any() else self.table[:, k, col]))

    def curve(self, s, parts=False):
        return emitter.evaluate(self.params[s], self.xs[s], self.psb, parts=parts)


def fit_map(m, method="v3", phys=None, threads=None, verbose=False, local=True, grid="auto", **opts):
    """Discover the emitters/basis of a map and fit every spectrum. `phys`: dict or path of a physics JSON.
    v3 only: local=True (default since 2.5) adds emitters confined to a small connected patch of the map
    (basis3.local_emitters), e.g. sharp lines on band-edge emission that map-wide discovery dilutes; `grid`
    ("auto", None, (rows, cols[, serpentine]) or a detect_grid dict) defines which spectra are neighbours."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    phys = phys if isinstance(phys, dict) else physics.load(phys)
    threads = threads or len(os.sched_getaffinity(0))
    xs, ys = m.xs, m.ys
    if method == "v3":
        o = dict(new_tol=phys["strain_shift_halfrange_nm"], **opts)
        b, fit, rep, keeps = basis3.grow(xs, ys, o, workers=threads, verbose=verbose)
        added = []
        if local:
            g = G.detect(m.ids, xs, ys) if grid == "auto" else (G.fixed(*grid) if isinstance(grid, (tuple, list)) else grid)
            oo = {**B.DEFAULTS, **B.GROW_DEFAULTS, **basis3.OPTS3, **o}
            b, fit, added = basis3.local_emitters(xs, ys, keeps, b, fit, oo, phys, g, workers=threads, verbose=verbose)
        hist = [{k: v for k, v in h.items() if k in ("step", "center", "K")} for h in rep["history"]]
        extra = dict(center_lo=b.lo.tolist(), center_hi=b.hi.tolist(), width_min=b.wmin.tolist(), width_max=b.wmax.tolist(),
                     history=hist, heldout_bic_C=rep["heldout_bic_C"], n_emitting=rep["n_emitting"],
                     best_start=[str(v) for v in fit.start], local_emitters=added,
                     local=[bool(v) for v in b.local])
        return Result("v3", list(m.ids), xs, ys, keeps, fit.params, fit.sse, b.psb.copy(), extra)
    if method == "v2":
        from . import pipeline
        b, rep = B.grow_basis(xs, ys, opts or None, workers=threads, verbose=verbose)
        keeps = [B.data_keep(y, B.noise_sigma(y)) for y in ys]
        r = pipeline.fit_spectra_basis(m.ids, [x[k] for x, k in zip(xs, keeps)], [y[k] for y, k in zip(ys, keeps)], b,
                                       workers=threads, starts=("mid", 8.0, 5.0))
        extra = dict(center_lo=b.center_lo.tolist(), center_hi=b.center_hi.tolist(), history=rep["history"],
                     heldout_bic_C=rep["heldout_bic_C"], best_start=[str(v) for v in r.best_start])
        return Result("v2", list(m.ids), xs, ys, keeps, r.params, r.sse, np.zeros(b.K, int), extra)
    from . import pipeline
    engine = pipeline.resolve_engine("auto")
    r = pipeline.fit_spectra(m.ids, xs, ys, device="cpu", workers=threads, engine=engine)
    params = np.insert(r.params, 1, 0.0, axis=1)               # constant background -> emitter layout (zero slope)
    keeps = [np.ones(len(y), bool) for y in ys]
    return Result("legacy", list(m.ids), xs, ys, keeps, params, r.sse, np.zeros(r.K, int),
                  dict(anchors_nm=model.ANCHORS_NM.tolist(), engine=engine))


def detect_grid(m):
    """Map shape from file names or spatial continuity (see fastdecov.grid); dict with rows=None if not detected."""
    return G.detect(m.ids, m.xs, m.ys)


def save(res, out, name=None, phys=None, figures=False, formats=("tif",), grid=None, threads=None, residual_limit=None,
         snr_classes=None, residual_features=True):
    """Write dc/dc_<id>.txt, emitters.tsv, spectra.tsv, basis.json, emitter_report.{md,json}, optional maps/figures.
    grid: None, "auto", (rows, cols[, serpentine]) or a dict from detect_grid. residual_limit: (floor, k), default (100, 3)."""
    phys = phys if isinstance(phys, dict) else physics.load(phys)
    if isinstance(grid, str) and grid == "auto":
        grid = G.detect(res.ids, res.xs, res.ys)
    return report.save(res, out, name, phys, figures, formats, grid, threads or len(os.sched_getaffinity(0)), residual_limit,
                       snr_classes, residual_features)
