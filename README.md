# SpecSmartDecov V2 (`fastdecov`)

Fast emitter discovery and deconvolution of photoluminescence (PL) spectrum maps, for example strained GaSe single-photon emitters.

This is the successor of the MATLAB [SpecSmartDecov](https://github.com/arenasluo/SpecSmartDecov). It runs directly on raw spectra:
- **One command line:** `fastdecov fit`.
- **A Python SDK:** `fastdecov.api`.
- **No preset peaks:** the number of emitters, their centres and their phonon sidebands are discovered from each map.

```bash
pip install .                                    # or: pip install -e .   (needs a C++ compiler, see Requirements)
fastdecov fit path/to/spectra out/ --figures            # map shape is detected automatically
```

## Methods

| `--method` | Model | Use it for |
|---|---|---|
| **`v3`** (default) | **Physical emitters:** each one is a zero-phonon line (pseudo-Voigt), optionally with a phonon sideband tied to it on the red side. A shoulder within one linewidth of an emitter can only become its sideband, never a new line. Lines of similar width closer than 0.85 × their mean FWHM are not allowed. A narrow line on top of a much broader component (for example GaSe band-edge emission) is allowed when it is 0.85 × its own FWHM away; see [Broad components](#broad-components-and-the-background). | Single-photon-emitter work: one emitter stays one line, so ν (0 = Lorentzian, 1 = Gaussian) and the linewidth keep their meaning. |
| `v2` | Data-driven basis of pseudo-Voigt lines, grown by held-out BIC (no anchors). | The best phenomenological fit. It can split one emitter into several unresolved lines. |
| `legacy` | The original SpecSmartDecov model: 12 fixed anchors (611 … 724 nm), constant background. | Compatibility with old `dc` results. |

All three use the same fast solver: bound-constrained Levenberg–Marquardt with an analytic Jacobian, in C++ with OpenMP, parallel over spectra. It is compiled on first use.

## Command line

```text
fastdecov fit INPUT_DIR OUTPUT_DIR [--method v3|v2|legacy] [--grid auto|none|ROWSxCOLS[s]] [--physics FILE]
              [--residual-limit FLOOR,K] [--snr-classes LOW,HIGH] [--local-emitters] [--threads N] [--figures] [--png]
              [--name NAME] [--pattern '*.txt'] [--verbose]
fastdecov report OUTPUT_DIR            # print the emitter report of a finished run
fastdecov physics [--template FILE]    # show / copy the physics parameters
fastdecov info                         # version and solver status
python -m fastdecov ...                # same as `fastdecov`
```

- **`INPUT_DIR`** is a directory of spectra: one file per map pixel, with two columns (wavelength in nm, intensity). Files are read in sorted order, and descending wavelength is handled.
- **`--grid`** sets the map shape, used for `emitter_maps.png`. The default, `auto`, first reads pixel coordinates from the file names (for example `x12_y07`), then infers the shape from the spectra (see [Map shape detection](#map-shape-detection)). `21x15` gives it explicitly (rows × columns in file order), `21x15s` declares a serpentine scan, and `none` skips the maps.
- **`--residual-limit 100,3`** sets the per-pixel residual check |r| ≤ max(100, 3√y); see [Residual limit](#residual-limit).
- **`--snr-classes 20,300`** sets the peak-SNR bounds of the substrate / weak / emitter classes; see [Spectrum classes](#spectrum-classes).
- **`--local-emitters`** (V3, experimental) also adds emitters that exist only in a small connected patch of the map; see [Local emitters](#local-emitters-experimental).
- **`--figures`** writes `dc/dc_<id>.tif`, one per spectrum: data, fit, background, each zero-phonon line, and the sidebands (dashed), plus a residual panel. Add **`--png`** for PNG copies.

Example: the 121-spectrum tent map from the reference study, on 12 CPU threads, in about 6 s. The 11 × 11 shape is detected automatically:

```bash
fastdecov fit pillar6_validation/output3-good out/pillar6 --figures --name pillar6
```

Paths are relative to the current directory. If `fastdecov` is not found, activate the environment it was installed in (for example `conda activate fastdecov`).

## Python SDK

```python
from fastdecov import api

m = api.load_map("pillar6_validation/output3-good")      # SpectrumMap(ids, xs, ys)
res = api.fit_map(m, method="v3", threads=8)              # discovery + fit of every spectrum
print(res.K, res.psb)                                     # emitters, sideband flags
print([res.median(0, k) for k in range(res.K)])           # emitter centres (nm)
res.table      # (spectra, emitters, 8): ZPL centre, ν, FWHM, ZPL area, PSB area, PSB offset, PSB FWHM, ZPL fraction
res.params     # raw fit parameters per spectrum (layout in fastdecov/emitter.py)
out = api.save(res, "out/pillar6", grid="auto", figures=True)   # or grid=(11, 11); api.detect_grid(m) to inspect
out["ranking"], out["quality"]                            # emitter ranking and fit quality
```

## Outputs

| File | Content |
|---|---|
| `dc/dc_<id>.txt` | MATLAB `writer.m` format, one row per emitter: zero-phonon centre, ν, FWHM, area |
| `dc/dc_<id>.tif` | fit figure (with `--figures`) |
| `emitters.tsv` | per spectrum and emitter: ZPL centre/ν/FWHM/area, sideband area/offset/FWHM, ZPL fraction, peak SNR, implied Huang–Rhys S |
| `spectra.tsv` | per spectrum: SSE, background, masked pixels (cosmic rays and clipped peaks), clipped pixels, max \|residual\|, residual / limit, residual-limit pass flag, peak counts, peak SNR, spectrum class |
| `basis.json` | discovered emitters, sideband flags, centre windows, discovery history, held-out BIC |
| `emitter_report.md` / `.json` | emitter ranking (below), fit quality and the residual-limit summary |
| `emitter_maps.png` | per-emitter area maps (when a grid is known) |
| `snr_map.png` | peak-SNR map and the substrate / weak / emitter classes (when a grid is known) |
| `run.json` | version, settings, physics, detected grid (with its confidence and candidates), timings |

### Emitter ranking

Each emitter is flagged as follows:
- **strong:** peak ≥ 50 σ;
- **isolated:** no other emitter within 1.5 × FWHM;
- **sharp:** FWHM ≤ 3 × the instrument FWHM at its wavelength.

Emitters that are all three are flagged as **g² priority**: the best candidates for photon-correlation measurements.

**Implied Huang–Rhys factor.** For emitters with a sideband, the report and `emitters.tsv` give S = −ln(ZPL fraction). The ZPL fraction is the Debye–Waller factor, which equals e^(−S) for linear coupling to a single phonon mode at low temperature. This is a model-dependent estimate, not a fit of the phonon structure:
- the Gaussian sideband's area is approximate;
- an unresolved neighbouring emitter inside the sideband would raise it.

It is not used in any decision. Examples: the tent map's 695.33 nm g²-priority emitter has a ZPL fraction of 0.948, so S ≈ 0.05; pillar 9's 709.5 nm emitter has 0.765, so S ≈ 0.27. The sideband is deliberately kept a simple Gaussian. A physical sideband shape follows from Franck–Condon factors and the phonon spectrum, and a free skew parameter would also hide unresolved neighbouring emitters.

ν and the sideband's zero-phonon fraction are reported but not used as gates. At low spectral resolution, a true single-photon emitter can look Gaussian-dominated: in the reference study, the antibunched ~692 nm emitter (g²(0) = 0.43) has ν ≈ 0.8 at 150 g/mm, because unresolved exciton/biexciton structure and spectral diffusion broaden it. Confirmation needs g²(0) < 0.5 or a high-resolution spectrum.

## Map shape detection

`--grid auto` (the default) works in two steps:
1. **File names.** If every file name holds two integers that form a complete rows × columns grid, those are the pixel coordinates.
2. **Spatial continuity.** Otherwise the shape comes from the spectra in file order. For each candidate width W and scan order (row by row, or serpentine), it compares the spectral distance from each pixel to the pixel "below" it with the distance between neighbours in the same row, using the spectra's first 8 principal components. The true shape gives the smallest ratio. The shape is accepted only if the runner-up is at least 10 % worse; otherwise no grid is used and a message says so.

On the three measured maps (11 × 11, 21 × 15 and 30 × 30), including their serpentine and column-by-column reorderings, the correct shape won in all 12 cases, by at least 15 %. Row-by-row and column-by-column order cannot be told apart, since that only transposes the map image. `run.json` records the method, the confidence and the top candidates.

## Residual limit

Every fitted pixel is checked against |residual| ≤ max(FLOOR, K·√y), by default max(100, 3√y):
- **The floor** covers the baseline and weak lines.
- **The √y term** is a shot-noise-like allowance that grows with the signal. When counts are close to detected photons, √y is the Poisson noise, so a fixed absolute limit would demand fitting below the noise on strong peaks.

`spectra.tsv` gives each spectrum's max |r|, its worst |r| / limit and a pass flag. The report gives the fraction of spectra that pass and the worst ones, and the `.tif` figures draw the limit band. The check is a report; it does not change the fit.

**Clipped peaks.** Measured spectra never repeat exactly the same value over several pixels on a peak, so runs of ≥ 3 identical values in the upper half of a spectrum are treated as clipping (saturation, or a capped export). They are masked, together with one pixel on each side, in the same way as cosmic rays, and counted in `clipped_pixels`. This applies to V3 and V2; the legacy method is unchanged. On the reference tent map, 19 of 121 spectra have a flat-topped 695 nm peak. Masking them raises its fit pass rate from 0.72 to 0.81 and keeps the same 4 emitters and the same g²-priority emitter. Maps without clipping give byte-identical results to 2.0.0.

## Spectrum classes

A map mixes spectra taken on the substrate, which are weak, with spectra taken on the emitters, which have high S/N. Each spectrum is classed by its peak SNR (peak above the edge baseline, divided by the noise sigma):
- **substrate:** below 20σ, the census "emitting" threshold;
- **weak:** 20σ up to 300σ;
- **emitter:** 300σ and above.

The bounds can be set with `--snr-classes LOW,HIGH`. Fit quality and the residual limit are reported per class, because map-wide fractions are dominated by whichever class has more pixels. Substrate spectra meet an absolute residual limit almost trivially, and a relative one is strict for weak spectra.

**Default V3** (fit pass rate / residual limit max(100, 3√y) met):

| Map | substrate | weak | emitter |
|---|---|---|---|
| Tent map (pillar 6, 121) | – | 39 spectra: 51 % / 56 % | 82 spectra: 95 % / 41 % |
| Pillar 9 crop (315) | – | 24: 62 % / 42 % | 291: 75 % / 1 % |
| Pillar 9 full (900) | 93: 94 % / 100 % | 446: 63 % / 77 % | 361: 96 % / 4 % |

On the emitter spectra of pillar 9, the typical max |r| is about 550 counts on peaks of 4,000–13,000 counts, a 4–6 % line-shape error at the peak. That, and not the substrate, is what the residual limit exposes.

## Broad components and the background

**Broad components.** Some features are much broader than the instrument (FWHM 9–30 nm against about 1 nm), for example GaSe band-edge emission around 600–650 nm. They are real and are fitted.
- **How they are reported.** A component wider than 5 × the instrument FWHM is labelled **broad** in the emitter table, with the note "e.g. GaSe band-edge emission or an unresolved ensemble". It is never a g² candidate.
- **The 2.0/2.1 problem.** Resolvability used 0.85 × the *mean* FWHM of a pair, so a 26 nm band created a ±12 nm exclusion zone. Narrow emitters on top of it could not be added, and V3 covered whole bands with a few broad blobs.
- **The 2.2 rule.** For a pair whose widths differ by at least 2×, the narrow line is resolved if it is at least 0.85 × *its own* FWHM from the broad centre. One guard remains: the broad component's spatial map must not simply track the narrow line (share-map correlation < 0.8). A pedestal that tracks a narrow line is how splitting fakes Lorentzian wings, while a real band has its own spatial distribution.
- **Similar-width pairs** keep the 0.85 × mean-FWHM rule.

**Background floor.** The linear background may not drop below −3 noise σ anywhere in the window, so the tails of strong lines must be explained by the lines and not by a tilted background. Fits that violate the floor are refitted with bounds that guarantee it.

`basis3.OPTS3`: `pair_rule="mean"` and `bg_floor_sigma=None` restore the 2.1 behaviour.

**Held-out synthetic maps** (12 maps with known emitters, the set used to freeze V3):

| | 2.1 rules | 2.2 rules |
|---|---|---|
| splits per map (lower is better) | 0.33 | **0.17** |
| precision | 0.914 | **0.924** |
| recall | 0.852 | **0.886** |
| close-cluster recall | 0.49 | **0.56** |
| ν class accuracy | 0.826 | **0.854** |
| fit pass rate | 0.865 | **0.931** |
| sideband recall | 0.58 | **0.67** |
| false-sideband rate (lower is better) | **0.29** | 0.33 |

**Real maps:**
- **Full pillar-9 map.** A narrow 683.6 nm line is now resolved on top of a broad 680 nm feature, and the broad filler component under 710/718/725 nm is gone. The emitter-class fit pass rate rises from 72 % to 96 %.
- **Tent map.** The 600–650 nm band gains a resolved 628/635 nm structure on top of the band-edge components. 695.35 nm stays the only g²-priority emitter.
- **Pillar-9 crop.** Unchanged.

**Still unmodelled:** a weak bump at about 706–710 nm on the red tail of the tent map's 695 nm line, about 30 meV below it. It may be an optical-phonon replica or a weak separate emitter, and it is left as residual on purpose.

## Local emitters (experimental)

V3 keeps an emitter only if it matters in at least 3 % of the map, so an emitter present in a handful of pixels can be missed, although such localized lines are prime single-photon candidates. `--local-emitters` adds a pass after discovery:
- **Candidates.** It looks for residual peaks that are ≥ 2× the residual limit, at least 0.6× the instrument width, and resolved from every fitted emitter in that spectrum.
- **Patches.** Candidates at the same wavelength (within half the strain range) that form spatially connected patches of ≥ 2 spectra become one emitter. Its amplitude is allowed only in those patches and their 8-neighbourhood, so it cannot split a line elsewhere.
- **Acceptance.** It is kept if per-spectrum BIC improves in at least half of the patch and it stays resolved there.

| Map | Emitters added | Fit pass rate (default → local) | Residual-limit pass | Time |
|---|---|---|---|---|
| Tent map (pillar 6) | 0 | 0.81 → 0.81 | 46 % | 6 s |
| Pillar 9 crop (315) | 4 (e.g. 672.5 nm in 28 spectra, 627.3 nm in 2) | 0.74 → 0.83 | 4 % → 5 % | 7 → 16 s |
| Pillar 9 full (900) | 10 | 0.66 → 0.72 | 46 % → 48 % | 10 → 45 s |

It finds real localized lines, but it barely changes the residual-limit pass fraction, and nearby patches of one strain-shifted emitter can still become two emitters (672.5 and 674.2 nm on the full map). It therefore stays opt-in.

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

- **Synthetic validation.** V3's rules were tuned and then frozen on synthetic maps with known emitters. On held-out maps, V3 2.0 compared with V2:
  - splits per map: 0.33 against 2.08;
  - precision: 0.91 against 0.73;
  - correct ν class: 0.83 against 0.77.

  The 2.2 resolvability rule improves these further, to 0.17 splits per map, 0.92 precision and 0.85 correct ν class; see [Broad components](#broad-components-and-the-background).
- **Dense clusters.** Emitters closer together than their linewidth cannot be resolved at 150 g/mm. V3 then reports fewer, broader emitters and leaves structured residuals: the 700–730 nm band of the reference pillar-9 map has a pass rate of 0.74. V2 fits these clusters better, but its extra lines are not physical emitters.
- **Sideband shape.** The sideband is modelled as a Gaussian, while real acoustic sidebands are skewed.
- **Residual limit.** Most spectra that fail max(100, 3√y) fail on line shape or dense clusters, not on missing emitters (see [Residual limit](#residual-limit)).
- **Other languages.** The C++, Rust, Fortran, Julia, Java, Go, Numba and CUDA ports implement 2.0.0. They match this package exactly on maps without clipped peaks and without `--local-emitters`.

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
