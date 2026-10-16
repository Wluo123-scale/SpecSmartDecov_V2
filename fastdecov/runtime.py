# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Pick the best prebuilt solver build for this CPU, check it, then run fastdecov (the Docker image's entry point).

The image holds one build of the three solvers per instruction-set level in $FASTDECOV_SOLVERS/<level>/ (see
Dockerfile). On start-up:
1. detect the architecture, the CPU's instruction-set flags and the usable CPUs (hw.cpu_count: affinity and the
   container's CPU quota);
2. for each level the CPU supports, best first, fit a small synthetic map in a child process, so a build that crashes
   (SIGILL on a CPU without its instructions) cannot take the run down, and accept it only if it finds the two known
   lines;
3. cache the choice per CPU (flags hash) and image, and run `fastdecov ARGS` with that build and thread count.

  python -m fastdecov.runtime probe [--all]     hardware and per-level check (--all: also levels the CPU lacks)
  python -m fastdecov.runtime exec -- ARGS...   choose a level, then run `fastdecov ARGS`
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from . import hw

SOLVERS = Path(os.environ.get("FASTDECOV_SOLVERS", "/opt/fastdecov/solvers"))
CACHE = Path(os.environ.get("FASTDECOV_CACHE", "/tmp/fastdecov-solver.json"))

# Best first. Each level: the CPU flags (/proc/cpuinfo) it needs, and the -march it is built with (Dockerfile).
LEVELS = {
    "x86_64": [("x86-64-v4", {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"}),
               ("x86-64-v3", {"avx2", "fma", "bmi1", "bmi2", "f16c", "movbe"}),
               ("x86-64-v2", {"sse4_2", "ssse3", "popcnt"})],
    "aarch64": [("armv8.2-a", {"asimd", "fphp", "asimdhp", "atomics"}),
                ("armv8-a", {"asimd"})],
}


def cpu_flags():
    try:
        txt = Path("/proc/cpuinfo").read_text()
    except OSError:
        return set()
    for line in txt.splitlines():
        k, _, v = line.partition(":")
        if k.strip() in ("flags", "Features"):
            return set(v.split())
    return set()


def hardware():
    return dict(arch=platform.machine(), cpus=hw.cpu_count(), os_cpu_count=os.cpu_count(),
                cpu_quota=hw.cgroup_cpu_limit(), flags=len(cpu_flags()))


def selftest():
    """Synthetic 24-spectrum map with known lines at 650 and 700 nm (the generator of tests/test_cli.py)."""
    import numpy as np
    from . import api, emitter
    rng = np.random.default_rng(1)
    x = np.linspace(600, 760, 400)
    xs, ys = [], []
    for _ in range(24):
        p = np.array([10.0, 0.0, 800 * (1 + 0.3 * rng.random()), 650.0, 0.3, 4.0,
                      1500 * (1 + 0.3 * rng.random()), 700.0, 0.6, 3.0, 0.15, 1.5, 3.0])
        xs.append(x); ys.append(emitter.evaluate(p, x, [0, 1]) + rng.normal(0, 3, x.size))
    t = time.time()
    res = api.fit_map(api.SpectrumMap([f"{i + 1:04d}" for i in range(24)], xs, ys), "v3", threads=2, local=False, grid=None)
    centres = sorted(res.median(0, k) for k in range(res.K))
    ok = any(abs(c - 650) < 1 for c in centres) and any(abs(c - 700) < 1 for c in centres)
    print(json.dumps(dict(ok=bool(ok), centres=[round(c, 4) for c in centres], sse_total=float(res.sse.sum()),
                          seconds=round(time.time() - t, 2))))
    return 0 if ok else 1


def check(level):
    env = {**os.environ, "FASTDECOV_BUILD_DIR": str(SOLVERS / level)}
    try:
        r = subprocess.run([sys.executable, "-m", "fastdecov.runtime", "selftest"], env=env, capture_output=True,
                           text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return dict(level=level, ok=False, why="self-test timed out")
    if r.returncode < 0:
        return dict(level=level, ok=False, why=f"killed by signal {-r.returncode}"
                    + (" (SIGILL: the CPU lacks this level's instructions)" if -r.returncode == 4 else ""))
    try:
        out = json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        if "BrokenProcessPool" in r.stderr:
            return dict(level=level, ok=False, why="a solver worker process crashed (most likely SIGILL: the CPU lacks "
                                                   "this level's instructions)")
        return dict(level=level, ok=False, why=f"exit {r.returncode}: {r.stderr.strip()[-300:]}")
    return dict(level=level, **out)


def choose(all_levels=False, use_cache=True):
    """(hardware, chosen level or None, per-level results). Levels whose directory is missing are ignored."""
    hwi = hardware(); flags = cpu_flags()
    key = f"{hwi['arch']}:{hashlib.sha256(' '.join(sorted(flags)).encode()).hexdigest()[:16]}:{os.environ.get('FASTDECOV_IMAGE', '')}"
    if use_cache and CACHE.exists():
        try:
            c = json.loads(CACHE.read_text())
            if c.get("key") == key and c.get("chosen"):
                return hwi, c["chosen"], c["tried"]
        except ValueError:
            pass
    tried, chosen, ref = [], None, None
    for level, need in LEVELS.get(hwi["arch"], []):
        if not (SOLVERS / level).is_dir():
            continue
        if not all_levels and not need <= flags:
            tried.append(dict(level=level, ok=False, why=f"skipped: CPU lacks {sorted(need - flags)}"))
            continue
        v = check(level)
        if v.get("ok") and ref is not None:
            v["sse_rel_diff_vs_chosen"] = abs(v["sse_total"] - ref) / ref
        tried.append(v)
        if v.get("ok") and chosen is None:
            chosen, ref = level, v["sse_total"]
            if not all_levels:
                break
    if use_cache and chosen:
        try:
            CACHE.write_text(json.dumps(dict(key=key, chosen=chosen, tried=tried)))
        except OSError:
            pass
    return hwi, chosen, tried


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "probe"
    if cmd == "selftest":
        return selftest()
    if cmd == "probe":
        hwi, chosen, tried = choose(all_levels="--all" in argv, use_cache=False)
        print(json.dumps(dict(hardware=hwi, chosen=chosen, tried=tried), indent=2))
        return 0 if chosen else 1
    if cmd == "exec":
        args = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
        if not SOLVERS.is_dir():                         # not the image: plain fastdecov (builds its own solvers)
            os.execvp("fastdecov", ["fastdecov", *args])
        hwi, chosen, tried = choose()
        if not chosen:
            print("fastdecov: no prebuilt solver passed its self-test on this CPU:\n" + json.dumps(tried, indent=1),
                  file=sys.stderr)
            return 2
        os.environ.update(FASTDECOV_BUILD_DIR=str(SOLVERS / chosen), OMP_NUM_THREADS=str(hwi["cpus"]))
        has_threads = any(arg == "--threads" or arg.startswith("--threads=") for arg in args)
        if args[:1] == ["fit"] and not has_threads:
            args += ["--threads", str(hwi["cpus"])]
        print(f"[fastdecov] {hwi['arch']}, solver build {chosen}, {hwi['cpus']} threads", file=sys.stderr, flush=True)
        os.execvp("fastdecov", ["fastdecov", *args])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
