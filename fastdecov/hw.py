# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Hardware facts that size a run, on any OS: usable CPUs and the C++ toolchain for the native solvers."""
import math
import os
import platform
import re
import shutil
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")


def cgroup_cpu_limit(root=CGROUP):
    """CPUs allowed by a container's CPU quota (cgroup v2 cpu.max, or v1 cfs quota/period), or None if unlimited."""
    try:
        quota, period = (root / "cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, math.floor(int(quota) / int(period)))
        return None
    except (OSError, ValueError):
        pass
    try:
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        return max(1, math.floor(quota / period)) if quota > 0 and period > 0 else None
    except (OSError, ValueError):
        return None


def cpu_count():
    """CPUs this process may use: its affinity mask where the OS has one (Linux), else the CPU count, capped by a
    container CPU quota. Docker's --cpus sets a quota, not an affinity mask, so without the cap OpenMP would start
    more threads than the container may run."""
    if hasattr(os, "sched_getaffinity"):
        n = len(os.sched_getaffinity(0))
    else:
        n = (os.process_cpu_count() if hasattr(os, "process_cpu_count") else None) or os.cpu_count() or 1
    q = cgroup_cpu_limit()
    return max(1, min(n, q) if q else n)


def has_libmvec():
    """glibc's vector maths library (vectorised exp) exists on Linux with glibc only."""
    return platform.system() == "Linux" and platform.libc_ver()[0] == "glibc"


def _versioned_gcc():
    """Homebrew-style g++-NN on PATH, newest first (macOS: /usr/bin/g++ is Apple clang, which has no OpenMP)."""
    found = {}
    for d in os.environ.get("PATH", "").split(os.pathsep):
        try:
            for f in os.listdir(d):
                m = re.fullmatch(r"g\+\+-(\d+)", f)
                if m and os.access(os.path.join(d, f), os.X_OK):
                    found.setdefault(int(m.group(1)), os.path.join(d, f))
        except OSError:
            continue
    return [found[k] for k in sorted(found, reverse=True)]


def compiler():
    """The C++ compiler for the solvers: $CXX, else (macOS) a Homebrew g++-NN, else g++, c++ or clang++; None if none."""
    if os.environ.get("CXX"):
        return os.environ["CXX"]
    cands = (_versioned_gcc() if platform.system() == "Darwin" else []) + ["g++", "c++", "clang++"]
    for c in cands:
        p = c if os.path.isabs(c) else shutil.which(c)
        if p:
            return p
    return None


def compile_flags():
    """Flags for the solver libraries. -march defaults to native (a build for this machine); set FASTDECOV_MARCH (for
    example x86-64-v3, armv8-a) to build for other machines, and FASTDECOV_CXXFLAGS to add flags."""
    flags = ["-O3", f"-march={os.environ.get('FASTDECOV_MARCH') or 'native'}", "-ffp-contract=fast", "-fopenmp",
             "-shared", "-fPIC"]
    libs = []
    if has_libmvec():
        flags.append("-DFASTDECOV_LIBMVEC"); libs.append("-lmvec")
    flags += os.environ.get("FASTDECOV_CXXFLAGS", "").split()
    return flags, libs
