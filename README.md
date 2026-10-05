# SpecSmartDecov V2 (`fastdecov`)

Fast emitter discovery and deconvolution of photoluminescence (PL) spectrum maps, for example strained GaSe single-photon emitters.

This is the successor of the MATLAB [SpecSmartDecov](https://github.com/arenasluo/SpecSmartDecov). It runs directly on raw spectra:
- **One command line:** `fastdecov fit`.
- **A Python SDK:** `fastdecov.api`.
- **No preset peaks:** the number of emitters, their centres and their phonon sidebands are discovered from each map.

```bash
pip install .                                    # or: pip install -e .   (needs a C++ compiler, see Requirements)
fastdecov fit path/to/spectra out/ --grid 11x11 --figures
```

## Methods

| `--method` | Model | Use it for |
|---|---|---|
| **`v3`** (default) | **Physical emitters:** each one is a zero-phonon line (pseudo-Voigt), optionally with a phonon sideband tied to it on the red side. A shoulder within one linewidth of an emitter can only become its sideband, never a new line. Lines closer than 0.85 × FWHM are not allowed. | Single-photon-emitter work: one emitter stays one line, so ν (0 = Lorentzian, 1 = Gaussian) and the linewidth keep their meaning. |
| `v2` | Data-driven basis of pseudo-Voigt lines, grown by held-out BIC (no anchors). | The best phenomenological fit. It can split one emitter into several unresolved lines. |
| `legacy` | The original SpecSmartDecov model: 12 fixed anchors (611 … 724 nm), constant background. | Compatibility with old `dc` results. |

All three use the same fast solver: bound-constrained Levenberg–Marquardt with an analytic Jacobian, in C++ with OpenMP, parallel over spectra. It is compiled on first use.

## Command line

```text
fastdecov fit INPUT_DIR OUTPUT_DIR [--method v3|v2|legacy] [--grid ROWSxCOLS] [--physics FILE]
              [--threads N] [--figures] [--png] [--name NAME] [--pattern '*.txt'] [--verbose]
fastdecov report OUTPUT_DIR            # print the emitter report of a finished run
fastdecov physics [--template FILE]    # show / copy the physics parameters
fastdecov info                         # version and solver status
python -m fastdecov ...                # same as `fastdecov`
```

- **`INPUT_DIR`** is a directory of spectra: one file per map pixel, with two columns (wavelength in nm, intensity). Files are read in sorted order, and descending wavelength is handled.
- **`--grid 11x11`** gives the map shape (row-major file order). It adds `emitter_maps.png`.
- **`--figures`** writes `dc/dc_<id>.tif`, one per spectrum: data, fit, background, each zero-phonon line, and the sidebands (dashed), plus a residual panel. Add **`--png`** for PNG copies.

Example: the 121-spectrum (11 × 11) tent map from the reference study, on 12 CPU threads, in about 6 s:

```bash
fastdecov fit pillar6_validation/output3-good out/pillar6 --grid 11x11 --figures --name pillar6
```

## Python SDK

```python
from fastdecov import api

m = api.load_map("pillar6_validation/output3-good")      # SpectrumMap(ids, xs, ys)
res = api.fit_map(m, method="v3", threads=8)              # discovery + fit of every spectrum
print(res.K, res.psb)                                     # emitters, sideband flags
print([res.median(0, k) for k in range(res.K)])           # emitter centres (nm)
res.table      # (spectra, emitters, 8): ZPL centre, ν, FWHM, ZPL area, PSB area, PSB offset, PSB FWHM, ZPL fraction
res.params     # raw fit parameters per spectrum (layout in fastdecov/emitter.py)
out = api.save(res, "out/pillar6", grid=(11, 11), figures=True)
out["ranking"], out["quality"]                            # emitter ranking and fit quality
```

## Outputs

| File | Content |
|---|---|
| `dc/dc_<id>.txt` | MATLAB `writer.m` format, one row per emitter: zero-phonon centre, ν, FWHM, area |
| `dc/dc_<id>.tif` | fit figure (with `--figures`) |
| `emitters.tsv` | per spectrum and emitter: ZPL centre/ν/FWHM/area, sideband area/offset/FWHM, ZPL fraction, peak SNR |
| `spectra.tsv` | per spectrum: SSE, background, number of masked (cosmic-ray) pixels |
| `basis.json` | discovered emitters, sideband flags, centre windows, discovery history, held-out BIC |
| `emitter_report.md` / `.json` | emitter ranking (below) and fit quality |
| `emitter_maps.png` | per-emitter area maps (with `--grid`) |
| `run.json` | version, settings, physics, timings |

### Emitter ranking

Each emitter is flagged as follows:
- **strong:** peak ≥ 50 σ;
- **isolated:** no other emitter within 1.5 × FWHM;
- **sharp:** FWHM ≤ 3 × the instrument FWHM at its wavelength.

Emitters that are all three are flagged as **g² priority**: the best candidates for photon-correlation measurements.

ν and the sideband's zero-phonon fraction are reported but not used as gates. At low spectral resolution, a true single-photon emitter can look Gaussian-dominated: in the reference study, the antibunched ~692 nm emitter (g²(0) = 0.43) has ν ≈ 0.8 at 150 g/mm, because unresolved exciton/biexciton structure and spectral diffusion broaden it. Confirmation needs g²(0) < 0.5 or a high-resolution spectrum.

## Physics parameters

The instrument resolution and the strain-shift range come from the experiment, not from the code. The default, `fastdecov/physics/gase_pillar_4K.json`, comes from the reference manuscript (arXiv:2305.03230), with provenance in the file:

| parameter | value | meaning |
|---|---|---|
| `instrument_resolution_meV` | 3.25 | 150 g/mm grating, about 1.0 nm at 620 nm and 1.3 nm at 700 nm; sets the "sharp" flag |
| `strain_shift_halfrange_nm` | 3.0 | how far one emitter's line may move across the map (strain); sets each emitter's centre window |
| `temperature_K` | 3.5 | documentation |

For another sample or spectrometer, run `fastdecov physics --template my.json`, edit the values from your own setup or manuscript, and pass `--physics my.json`.

## Requirements and performance

- **Python and libraries:** Python ≥ 3.10, NumPy, SciPy, Matplotlib.
- **A C++ compiler:** g++ (or clang++) with OpenMP, on Linux with glibc, because the vectorised `exp` comes from libmvec. Solvers are compiled on first use into the package's `native/`, or into `$FASTDECOV_BUILD_DIR` or `~/.cache/fastdecov` when that is read-only.
- **Optional GPU:** a CUDA path for the legacy method (`pip install .[gpu]`, see `analysis.py`).

**Timing.** On a 20-core desktop, V3 discovery plus fitting takes about 2–8 s for a 315-spectrum map. The original MATLAB code under Octave takes about 325 s for the same map with fixed peaks, and fits it much worse.

**Other languages.** The algorithm has also been ported to C++, Rust, Fortran, Julia, Java, Go, Numba and CUDA. All ports give byte-identical results, and Go was the fastest. Those ports and their benchmark are kept in the development repository; this package ships the Python + C++ implementation.

## Validation and limits

- **Synthetic validation.** V3's rules were tuned and then frozen on synthetic maps with known emitters. On held-out maps, V3 compared with V2:
  - splits per map: 0.33 against 2.08;
  - precision: 0.91 against 0.73;
  - correct ν class: 0.83 against 0.77.
- **Dense clusters.** Emitters closer together than their linewidth cannot be resolved at 150 g/mm. V3 then reports fewer, broader emitters and leaves structured residuals: the 700–730 nm band of the reference pillar-9 map has a pass rate of 0.74. V2 fits these clusters better, but its extra lines are not physical emitters.
- **Sideband shape.** The sideband is modelled as a Gaussian, while real acoustic sidebands are skewed.

## Backward compatibility

`analysis.py` keeps the V1 interface (`--input/--output/--device`) for the fixed 12-anchor legacy fit with CPU or GPU solvers.

## Tests

```bash
pip install .[test] && pytest -q tests
```

## License

MIT; see `LICENSE`.

## Manuscript and benchmark results

- `manuscript/`: the methodology manuscript (`.md`/`.docx`/`.pdf`), figures and the scripts that generate every number from verified metrics.
- `results/`: outputs of the three fastest implementations (Go, Fortran, C++), the Python + C++ reference and the Octave/MATLAB baseline on Pillar 1 and Pillar 2: `dc` tables, `.tif` figures (Git LFS; run `git lfs pull`), emitter tables and parity reports. See `results/README.md` for the V3 metrics.
