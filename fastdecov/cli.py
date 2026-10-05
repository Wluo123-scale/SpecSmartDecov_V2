"""Command-line interface: `fastdecov <command> ...` (also `python -m fastdecov`).

  fastdecov fit INPUT_DIR OUTPUT_DIR [--method v3|v2|legacy] [--grid ROWSxCOLS] [--physics FILE]
                [--threads N] [--figures] [--png] [--name NAME] [--verbose]
  fastdecov report OUTPUT_DIR                print the emitter report of a finished run
  fastdecov physics [--template FILE]        show the default physics parameters, or write an editable copy
  fastdecov info                             version, compiler and solver status
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path


def _grid(s):
    try:
        r, c = (int(v) for v in s.lower().split("x"))
        return r, c
    except Exception:
        raise argparse.ArgumentTypeError("grid must look like 21x15 (rows x columns, row-major spectrum order)")


def cmd_fit(a):
    from . import __version__, api, physics
    t0 = time.perf_counter()
    phys = physics.load(a.physics)
    m = api.load_map(a.input, a.pattern)
    if a.grid and a.grid[0] * a.grid[1] != len(m):
        sys.exit(f"--grid {a.grid[0]}x{a.grid[1]} = {a.grid[0] * a.grid[1]} pixels, but {len(m)} spectra were found")
    threads = a.threads or len(os.sched_getaffinity(0))
    print(f"fastdecov {__version__}: {len(m)} spectra from {a.input}, method {a.method}, {threads} threads", flush=True)
    res = api.fit_map(m, a.method, phys, threads, verbose=a.verbose)
    t_fit = time.perf_counter()
    formats = ("tif", "png") if a.png else ("tif",)
    out = api.save(res, a.output, a.name, phys, a.figures, formats, a.grid, threads)
    t_end = time.perf_counter()
    q = out["quality"]
    (Path(a.output) / "run.json").write_text(json.dumps(dict(
        version=__version__, method=a.method, input=str(Path(a.input).resolve()), n_spectra=len(m), threads=threads,
        physics=phys, grid=a.grid, figures=a.figures, fit_s=t_fit - t0, total_s=t_end - t0, python=platform.python_version(),
        fit_quality=q), indent=1, default=float) + "\n")
    print((Path(a.output) / "emitter_report.md").read_text())
    print(f"{res.K} emitters ({int(sum(res.psb))} with phonon sideband); pass rate {q['pass_rate']:.2f}; "
          f"{t_end - t0:.1f} s -> {a.output}")


def cmd_report(a):
    p = Path(a.output) / "emitter_report.md"
    if not p.is_file():
        sys.exit(f"no emitter_report.md in {a.output}")
    print(p.read_text())


def cmd_physics(a):
    from . import physics
    if a.template:
        Path(a.template).write_text(physics.DEFAULT.read_text())
        print(f"wrote {a.template}; edit it and pass --physics {a.template}")
    else:
        print(physics.DEFAULT.read_text())


def cmd_info(a):
    from . import __version__, lm_native
    print(f"fastdecov {__version__}, Python {platform.python_version()}")
    for name, loader in (("legacy solver (lmfit.cpp)", lm_native.load), ("basis solver (lmfit_k.cpp)", lm_native.load_k),
                         ("emitter solver (lmfit_zp.cpp)", lm_native.load_zp)):
        try:
            loader(); print(f"  {name}: ok ({lm_native.LIB.parent})")
        except Exception as err:
            print(f"  {name}: NOT available: {err}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fastdecov", description="Fast emitter discovery and deconvolution of PL spectrum maps.")
    sub = ap.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit", help="discover emitters and fit every spectrum of a map")
    f.add_argument("input", help="directory of spectra (two columns: wavelength nm, intensity)")
    f.add_argument("output", help="output directory (created)")
    f.add_argument("--method", choices=("v3", "v2", "legacy"), default="v3",
                   help="v3: zero-phonon line + phonon sideband emitters (default); v2: best-fit line basis; "
                        "legacy: original 12 fixed anchors")
    f.add_argument("--grid", type=_grid, default=None, help="map shape ROWSxCOLS (row-major file order) for emitter maps")
    f.add_argument("--physics", default=None, help="physics JSON (instrument resolution, strain range); "
                                                   "default: GaSe pillars at 3.5 K, 150 g/mm (see `fastdecov physics`)")
    f.add_argument("--threads", type=int, default=None, help="CPU threads (default: all available)")
    f.add_argument("--figures", action="store_true", help="write dc/dc_<id>.tif fit figures")
    f.add_argument("--png", action="store_true", help="with --figures, also write .png")
    f.add_argument("--name", default=None, help="map name used in reports (default: output directory name)")
    f.add_argument("--pattern", default="*.txt", help="spectrum file pattern (default *.txt)")
    f.add_argument("--verbose", action="store_true", help="print discovery steps")
    f.set_defaults(func=cmd_fit)
    r = sub.add_parser("report", help="print the emitter report of a finished run"); r.add_argument("output")
    r.set_defaults(func=cmd_report)
    p = sub.add_parser("physics", help="show the default physics parameters or write an editable template")
    p.add_argument("--template", default=None, help="write the default physics JSON here")
    p.set_defaults(func=cmd_physics)
    i = sub.add_parser("info", help="version and solver status"); i.set_defaults(func=cmd_info)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
