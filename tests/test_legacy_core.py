# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastdecov import lm_cpu, model, pipeline  # noqa: E402

X = np.linspace(549.601, 849.883, 354)


def legacy_sumvoigt(parveci, x):
    """Literal transcription of legacy/sumvoigt.m (mode 1)."""
    a, x0, nu, sig = parveci
    nu = min(max(nu, 0.0), 1.0)
    s = abs(sig)
    f1 = nu * np.sqrt(np.log(2) / np.pi) * np.exp(-4 * np.log(2) * ((x - x0) / s) ** 2) / s
    f2 = (1 - nu) / (np.pi * s * (1 + 4 * ((x - x0) / s) ** 2))
    f3 = nu * np.sqrt(np.log(2) / np.pi) / s + (1 - nu) / (np.pi * s)
    return a * (f1 + f2) / f3


def random_params(seed=0):
    rng = np.random.default_rng(seed)
    y = 1000 + 500 * np.exp(-((X - 680) / 20) ** 2)
    p0, lb, ub = model.setup(X, y)
    return lb + (ub - lb) * rng.uniform(0.2, 0.8, model.NPAR), y


def test_initial_centres_are_deterministic_rdnpeak_values():
    assert np.allclose(model.initial_centers(), model.ANCHORS_NM - 0.3147236863931789)


def test_model_matches_legacy_sumvoigt_multiv():
    p, _ = random_params()
    ref = p[0] + sum(legacy_sumvoigt(p[1 + 4 * k: 5 + 4 * k], X) for k in range(model.NP))
    assert np.allclose(model.evaluate(p, X), ref, rtol=1e-12, atol=1e-9)


def test_area_is_integral_of_component():
    p, _ = random_params(1)
    xf = np.linspace(-20000, 20000, 4_000_001) + 680
    comp = model.components(p, xf)
    num = np.trapezoid(comp, xf, axis=-1)
    # Lorentzian tails beyond +-20000 nm are negligible at this tolerance
    assert np.allclose(num, model.areas(p), rtol=2e-3)


def test_analytic_jacobian_matches_finite_differences():
    p, y = random_params(2)
    _, J = lm_cpu.residual_and_jacobian(p[None], X[None], y[None])
    Jfd = np.empty_like(J[0])
    for k in range(model.NPAR):
        h = 1e-6 * max(1.0, abs(p[k]))
        e = np.zeros_like(p); e[k] = h
        rp, _ = lm_cpu.residual_and_jacobian((p + e)[None], X[None], y[None], want_jac=False)
        rm, _ = lm_cpu.residual_and_jacobian((p - e)[None], X[None], y[None], want_jac=False)
        Jfd[:, k] = (rp[0] - rm[0]) / (2 * h)
    assert np.abs(J[0] - Jfd).max() <= 1e-6 * np.abs(Jfd).max()


def test_noise_free_recovery_of_strong_components():
    truth, _ = random_params(3)
    pk = truth[1:].reshape(model.NP, 4)
    pk[:, 3] = np.clip(pk[:, 3], 3, 15)            # physically typical widths
    y = model.evaluate(truth, X)
    res = pipeline.fit_spectra(["s"], [X], [y], device="cpu", workers=1)
    assert res.sse[0] / np.sum((y - y.mean()) ** 2) < 1e-4
    a = model.areas(truth)
    strong = a / a.sum() >= 0.1
    got = res.params[0, 1:].reshape(model.NP, 4)
    assert np.all(np.abs(got[strong, 1] - pk[strong, 1]) < 0.25)


def test_bounds_respected_and_outputs_physical():
    rng = np.random.default_rng(4)
    xs = [X] * 4
    ys = [model.evaluate(random_params(10 + i)[0], X) + rng.normal(0, 5, X.size) for i in range(4)]
    res = pipeline.fit_spectra(list("abcd"), xs, ys, device="cpu", workers=1)
    for s in range(4):
        _, lb, ub = model.setup(xs[s], ys[s])
        assert np.all(res.params[s] >= lb) and np.all(res.params[s] <= ub)
    assert np.all(res.areas > 0)


def gpu_available():
    try:
        pipeline.probe_gpus()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not gpu_available(), reason="no usable CUDA device")
def test_gpu_matches_cpu_on_first_iterations():
    from fastdecov import lm_cuda
    p, y = random_params(5)
    y = model.evaluate(p, X) + np.random.default_rng(5).normal(0, 5, X.size)
    p0, lb, ub = model.setup(X, y)
    for it in (1, 3, 10):
        pc, sc, _, _ = lm_cpu.fit_batch(p0[None], lb[None], ub[None], X[None], y[None], max_iter=it)
        pg, sg, _, _ = lm_cuda.fit_problems(X[None], y[None], np.zeros(1, int), p0[None], lb[None], ub[None], max_iter=it)
        assert np.allclose(pg, pc, rtol=1e-8, atol=1e-10)


def test_cli_cpu_contract(tmp_path):
    rng = np.random.default_rng(6)
    inp = tmp_path / "in"; inp.mkdir()
    for i in range(3):
        y = model.evaluate(random_params(20 + i)[0], X) + rng.normal(0, 5, X.size)
        np.savetxt(inp / f"{i:04d}.txt", np.column_stack([X, y]), delimiter="\t")
    out = tmp_path / "out"
    subprocess.run([sys.executable, str(ROOT / "analysis.py"), "--input", str(inp), "--output", str(out),
                    "--device", "cpu", "--workers", "1"], check=True)
    rt = json.loads((out / "runtime.json").read_text())
    assert rt["requested_device"] == rt["resolved_device"] == "cpu" and rt["n_spectra"] == 3
    rows = (out / "fit_results.csv").read_text().splitlines()
    assert rows[0] == "spectrum_id,component_id,center_nm,nu,width_nm,area,fit_status"
    assert len(rows) == 1 + 3 * 12
    assert len((out / "methods.md").read_text()) >= 250


def test_gpu_request_without_cuda_fails_clearly(monkeypatch):
    def broken():
        raise RuntimeError("no CUDA")
    monkeypatch.setattr(pipeline, "probe_gpus", broken)
    with pytest.raises(RuntimeError, match="--device gpu"):
        pipeline.resolve_device("gpu")
    assert pipeline.resolve_device("auto") == ("cpu", [])


def test_legacy_dc_roundtrip_and_refine_never_worse(tmp_path):
    from fastdecov import legacy
    p, y = random_params(30)
    y = model.evaluate(p, X) + np.random.default_rng(30).normal(0, 5, X.size)
    res = pipeline.fit_spectra(["s"], [X], [y], device="cpu", workers=1, starts=("mid",), max_iter=15)
    pk = res.params[0, 1:].reshape(model.NP, 4)
    table = np.column_stack([pk[:, 1], pk[:, 2], pk[:, 3], res.areas[0]])
    legacy.write_dc(tmp_path / "dc_s.txt", table)
    raw = (tmp_path / "dc_s.txt").read_bytes()
    assert raw.startswith(b"Peak #\t\tPeak Center") and b"\r\n" in raw
    back = legacy.read_dc(tmp_path / "dc_s.txt")
    assert np.allclose(back, table, rtol=1e-5)
    p0, lb, ub, h = legacy.refine_problem(back, X, y)
    start_sse = float(np.sum((model.evaluate(p0, X) - y) ** 2))
    ref = pipeline.fit_spectra(["s"], [X], [y], device="cpu", workers=1, starts=(), seeds=[(p0, lb, ub)])
    assert h == 0.5 and ref.sse[0] <= start_sse * (1 + 1e-9)


@pytest.mark.skipif(not __import__("fastdecov.lm_native", fromlist=["x"]).available(), reason="no C++ compiler")
def test_native_engine_matches_numpy():
    from fastdecov import lm_native
    p, y = random_params(40)
    y = model.evaluate(p, X) + np.random.default_rng(40).normal(0, 5, X.size)
    p0, lb, ub = model.setup(X, y)
    for it in (1, 3, 10, 300):
        pc, sc, _, _ = lm_cpu.fit_batch(p0[None], lb[None], ub[None], X[None], y[None], max_iter=it)
        pn, sn, _, _ = lm_native.fit_problems(X[None], y[None], np.zeros(1, int), p0[None], lb[None], ub[None],
                                              threads=1, max_iter=it)
        if it <= 10:
            assert np.allclose(pn, pc, rtol=1e-8, atol=1e-10)
        assert sn[0] <= sc[0] * 1.05
