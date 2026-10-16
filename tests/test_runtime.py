# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Portability: CPU count under container quotas, compiler and flag choice per OS, prebuilt solvers without a
compiler, and the Docker entry point's choice of solver build (best supported level, fallback, cache)."""
import os

import pytest

from fastdecov import hw, lm_native, runtime


@pytest.mark.parametrize("cpu_max, expect", [("max 100000", None), ("200000 100000", 2), ("150000 100000", 1),
                                             ("50000 100000", 1)])
def test_cgroup_v2_quota(tmp_path, cpu_max, expect):
    (tmp_path / "cpu.max").write_text(cpu_max + "\n")
    assert hw.cgroup_cpu_limit(tmp_path) == expect


def test_cgroup_v1_quota_and_none(tmp_path):
    assert hw.cgroup_cpu_limit(tmp_path) is None                         # no cgroup files: unlimited
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("300000")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000")
    assert hw.cgroup_cpu_limit(tmp_path) == 3
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1")
    assert hw.cgroup_cpu_limit(tmp_path) is None


def test_cpu_count_is_capped_by_quota(monkeypatch):
    monkeypatch.setattr(hw, "cgroup_cpu_limit", lambda root=None: 1)
    assert hw.cpu_count() == 1
    monkeypatch.setattr(hw, "cgroup_cpu_limit", lambda root=None: None)
    assert hw.cpu_count() >= 1


def test_cpu_count_without_affinity(monkeypatch):
    """macOS and Windows have no os.sched_getaffinity."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(hw, "cgroup_cpu_limit", lambda root=None: None)
    assert hw.cpu_count() == (getattr(os, "process_cpu_count", os.cpu_count)() or 1)


def test_compile_flags_follow_libmvec_and_env(monkeypatch):
    monkeypatch.delenv("FASTDECOV_MARCH", raising=False); monkeypatch.delenv("FASTDECOV_CXXFLAGS", raising=False)
    monkeypatch.setattr(hw, "has_libmvec", lambda: True)
    flags, libs = hw.compile_flags()
    assert "-march=native" in flags and "-DFASTDECOV_LIBMVEC" in flags and libs == ["-lmvec"] and "-fopenmp" in flags
    monkeypatch.setattr(hw, "has_libmvec", lambda: False)                # macOS, musl: scalar exp, no -lmvec
    flags, libs = hw.compile_flags()
    assert "-DFASTDECOV_LIBMVEC" not in flags and libs == []
    monkeypatch.setenv("FASTDECOV_MARCH", "x86-64-v3"); monkeypatch.setenv("FASTDECOV_CXXFLAGS", "-g -DX=1")
    flags, _ = hw.compile_flags()
    assert "-march=x86-64-v3" in flags and "-march=native" not in flags and flags[-2:] == ["-g", "-DX=1"]


def _fake_exe(d, name):
    p = d / name; p.write_text("#!/bin/sh\n"); p.chmod(0o755)
    return str(p)


def test_compiler_prefers_cxx_then_homebrew_gcc_on_macos(tmp_path, monkeypatch):
    _fake_exe(tmp_path, "g++-13"); newest = _fake_exe(tmp_path, "g++-16"); _fake_exe(tmp_path, "g++")
    monkeypatch.setenv("PATH", str(tmp_path)); monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    assert hw.compiler() == newest                                       # not Apple clang's g++
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    assert hw.compiler() == str(tmp_path / "g++")
    monkeypatch.setenv("CXX", "/opt/my/c++")
    assert hw.compiler() == "/opt/my/c++"
    monkeypatch.delenv("CXX"); monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert hw.compiler() is None


def test_prebuilt_library_is_used_without_a_compiler(tmp_path, monkeypatch):
    src = tmp_path / "s.cpp"; lib = tmp_path / "libs.so"
    lib.write_text("prebuilt"); src.write_text("// newer source")
    os.utime(lib, (1, 1))                                                # library older than its source
    monkeypatch.setattr(hw, "compiler", lambda: None)
    assert lm_native._build(src, lib) == lib and lib.read_text() == "prebuilt"
    lib.unlink()
    with pytest.raises(RuntimeError, match="Docker"):
        lm_native._build(src, lib)


def _levels(tmp_path, monkeypatch, arch, flags, results):
    for name in results:
        (tmp_path / name).mkdir()
    monkeypatch.setattr(runtime, "SOLVERS", tmp_path)
    monkeypatch.setattr(runtime, "CACHE", tmp_path / "cache.json")
    monkeypatch.setattr(runtime.platform, "machine", lambda: arch)
    monkeypatch.setattr(runtime, "cpu_flags", lambda: set(flags))
    calls = []

    def check(level):
        calls.append(level)
        ok = results[level]
        return dict(level=level, ok=ok, sse_total=1.0) if ok else dict(level=level, ok=False, why="killed by signal 4")
    monkeypatch.setattr(runtime, "check", check)
    return calls


def test_choose_best_level_the_cpu_supports(tmp_path, monkeypatch):
    avx2 = {"sse4_2", "ssse3", "popcnt", "avx2", "fma", "bmi1", "bmi2", "f16c", "movbe"}
    calls = _levels(tmp_path, monkeypatch, "x86_64", avx2, {"x86-64-v4": True, "x86-64-v3": True, "x86-64-v2": True})
    _, chosen, tried = runtime.choose()
    assert chosen == "x86-64-v3" and calls == ["x86-64-v3"]              # v4 skipped: no AVX-512
    assert tried[0]["level"] == "x86-64-v4" and "skipped" in tried[0]["why"]
    calls.clear()
    assert runtime.choose()[1] == "x86-64-v3" and calls == []            # cached


def test_choose_falls_back_when_a_build_fails_its_self_test(tmp_path, monkeypatch):
    calls = _levels(tmp_path, monkeypatch, "aarch64", {"asimd", "fphp", "asimdhp", "atomics"},
                    {"armv8.2-a": False, "armv8-a": True})
    _, chosen, tried = runtime.choose(use_cache=False)
    assert chosen == "armv8-a" and calls == ["armv8.2-a", "armv8-a"] and "signal 4" in tried[0]["why"]


def test_choose_none_when_nothing_passes(tmp_path, monkeypatch):
    _levels(tmp_path, monkeypatch, "aarch64", {"asimd"}, {"armv8-a": False})
    assert runtime.choose(use_cache=False)[1] is None
    assert not (tmp_path / "cache.json").exists()                        # a failure is not cached


def test_exec_outside_the_image_runs_plain_fastdecov(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "SOLVERS", tmp_path / "missing")
    seen = []
    monkeypatch.setattr(runtime.os, "execvp", lambda f, a: seen.append(a))
    runtime.main(["exec", "--", "info"])
    assert seen == [["fastdecov", "info"]]


@pytest.mark.parametrize("args, expected", [
    (["fit", "in", "out"], ["fit", "in", "out", "--threads", "3"]),
    (["fit", "in", "out", "--threads", "1"], ["fit", "in", "out", "--threads", "1"]),
    (["fit", "in", "out", "--threads=1"], ["fit", "in", "out", "--threads=1"]),
])
def test_exec_sets_build_dir_and_threads(tmp_path, monkeypatch, args, expected):
    _levels(tmp_path, monkeypatch, "aarch64", {"asimd"}, {"armv8-a": True})
    monkeypatch.setattr(runtime.hw, "cpu_count", lambda: 3)
    seen = []
    monkeypatch.setattr(runtime.os, "execvp", lambda f, a: seen.append(a))
    monkeypatch.delenv("FASTDECOV_BUILD_DIR", raising=False); monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    runtime.main(["exec", "--", *args])                                # exec sets both; monkeypatch restores them
    assert seen == [["fastdecov", *expected]]
    assert os.environ["FASTDECOV_BUILD_DIR"] == str(tmp_path / "armv8-a") and os.environ["OMP_NUM_THREADS"] == "3"
