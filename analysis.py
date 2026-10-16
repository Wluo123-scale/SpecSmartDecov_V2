#!/usr/bin/env python3
# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""V1 interface (kept for compatibility): fixed 12-anchor legacy fit. New code should use `fastdecov fit --method legacy`.

    python3 analysis.py --input INPUT_DIR --output OUTPUT_DIR --device cpu|gpu|auto
                        [--workers N] [--gpus 0,1] [--starts mid,8,5]
"""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

METHODS_TEMPLATE = """# Method

Each spectrum (wavelength nm, intensity) is fitted with the legacy SpecSmartDecov model:
a non-negative constant background plus twelve pseudo-Voigt components in the fixed
anchor order 611, 617, 627, 672, 680, 684, 689, 700, 707, 710, 717, 724 nm. Every
component is a*(w*G + (1-w)*L) with Gaussian fraction nu in [0, 1] and FWHM s, exactly as
legacy sumvoigt.m; reported area is writer.m's 0.5*a / (nu*sqrt(ln2/pi)/s + (1-nu)/(pi*s)).

Initial guess and bounds follow IGgen.m/BoundGen.m with IG='auto': rdnpeak.m resets
rng(0) before each draw, so each initial centre is the constant k + 0.5 - 0.8147236863931789;
centres may move +-0.5 %; heights lie in [0, 1.1 * interpolated intensity]; widths in
(0, (x_end - x_1)/4]; background in [0, max(y)]. Where interpolated intensity is
non-positive (legacy bounds inconsistent) the height upper bound is clamped to 1e-6.

Optimisation: bound-constrained Levenberg-Marquardt in Coleman-Li scaled variables
(the scaling of MATLAB/SciPy trust-region-reflective), analytic Jacobian, strictly
feasible iterates. Starting widths {starts}; the fit with the lowest sum of squared
residuals is kept. All spectra are batched: on CPU the fits run on {workers}; on GPU a CUDA C kernel (NVRTC via CuPy RawKernel) runs one
thread block per fit with an in-block Cholesky solve ({gpus}).

The legacy DC.m refit loop is dropped: it repeats an identical deterministic fit.
Plotting, pickpeaks suggestions and confidence intervals are not computed.
Dependencies: Python {python}, NumPy {numpy}, optional CuPy {cupy}.

Limitations: the 12-component model is degenerate for weak overlapping components
(roughly 700-724 nm), so their parameters depend on the optimiser path and are not
individually meaningful; strong components and the total fit are robust. Results are not
bit-identical to MATLAB lsqcurvefit. Per-spectrum SSE, iterations, convergence flag and
selected start are written to diagnostics.csv.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", choices=("cpu", "gpu", "auto"), default="auto")
    ap.add_argument("--workers", type=int, default=None, help="CPU workers/threads (default: all allowed cores)")
    ap.add_argument("--engine", choices=("auto", "native", "numpy"), default="auto",
                    help="CPU engine: native C++/OpenMP (built on first use) or NumPy; auto prefers native")
    ap.add_argument("--gpus", default=None, help="comma-separated CUDA device ids (default: all)")
    ap.add_argument("--starts", default="mid,8,5", help="starting widths: 'mid' and/or nm values")
    ap.add_argument("--max-iter", type=int, default=300, help="iteration cap per fit")
    ap.add_argument("--dc-dir", default=None, help="also write legacy writer.m tables dc_<id>.txt here")
    ap.add_argument("--dc-figures", action="store_true",
                    help="with --dc-dir, also write dc_<id>.tif fit/residual figures (plotter.m layout)")
    ap.add_argument("--refine-from", default=None,
                    help="directory of legacy dc_<id>.txt fits to restart from (parity/refinement mode)")
    ap.add_argument("--refine-mode", choices=("refine", "best"), default="refine",
                    help="refine: only restart from the legacy fit; best: also run the independent starts "
                         "and keep the lowest SSE")
    args = ap.parse_args(argv)

    started = time.perf_counter()
    import numpy as np
    from fastdecov import hw, io, pipeline

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    resolved, found = pipeline.resolve_device(args.device, gpus)
    starts = tuple(s if s == "mid" else float(s) for s in args.starts.split(","))
    workers = args.workers or hw.cpu_count()

    engine = pipeline.resolve_engine(args.engine) if resolved == "cpu" else "cuda"
    ids, xs, ys = io.read_directory(args.input)
    seeds, windows = None, {}
    if args.refine_from:
        from fastdecov import legacy
        seeds = []
        for sid, x, y in zip(ids, xs, ys):
            path = Path(args.refine_from) / f"dc_{sid}.txt"
            if path.is_file():
                p0, lb, ub, h = legacy.refine_problem(legacy.read_dc(path), x, y)
                seeds.append((p0, lb, ub)); windows[sid] = h
            else:
                seeds.append(None)
        if args.refine_mode == "refine":
            missing = [sid for sid, sd in zip(ids, seeds) if sd is None]
            if missing:
                raise SystemExit(f"--refine-mode refine: no legacy fit for {len(missing)} spectra, e.g. {missing[:5]}")
            starts = ()
    res = pipeline.fit_spectra(ids, xs, ys, device=resolved, gpu_devices=[g[0] for g in found],
                               workers=workers, starts=starts, seeds=seeds, engine=engine, max_iter=args.max_iter)
    io.write_results(res, args.output)
    if args.dc_dir:
        from fastdecov import legacy
        Path(args.dc_dir).mkdir(parents=True, exist_ok=True)
        areas = res.areas
        for s, sid in enumerate(ids):
            legacy.write_dc(Path(args.dc_dir) / f"dc_{sid}.txt", res.table(s))
        if args.dc_figures:
            legacy.write_tifs(args.dc_dir, ids, xs, ys, res.params, workers)

    cupy_version = None
    if resolved == "gpu":
        import cupy
        cupy_version = cupy.__version__
    elapsed = time.perf_counter() - started
    io.write_runtime(args.output, {
        "requested_device": args.device,
        "resolved_device": resolved,
        "elapsed_seconds": elapsed,
        "n_spectra": len(ids),
        "cpu_workers": workers if resolved == "cpu" else None,
        "engine": engine,
        "gpus": [{"id": d, "name": n} for d, n in found] if resolved == "gpu" else [],
        "starts": [str(s) for s in starts],
        "max_iter": args.max_iter,
        "refine_from": args.refine_from,
        "refine_mode": args.refine_mode if args.refine_from else None,
        "refine_windows_nm": windows or None,
        "converged_fraction": float(np.mean(res.converged)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cupy": cupy_version,
    })
    (Path(args.output) / "methods.md").write_text(METHODS_TEMPLATE.format(
        starts=", ".join(str(s) for s in starts),
        workers=(f"{workers} OpenMP threads, C++ engine" if engine == "native" else f"{workers} processes, NumPy engine")
                if resolved == "cpu" else "not used",
        gpus=", ".join(n for _, n in found) if resolved == "gpu" else "not used",
        python=platform.python_version(), numpy=np.__version__, cupy=cupy_version or "not used"))
    print(f"{len(ids)} spectra fitted on {resolved} in {elapsed:.2f} s -> {args.output}")


if __name__ == "__main__":
    main()
