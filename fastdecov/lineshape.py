"""Voigt decomposition of each emitter line (report only; the fit is unchanged).

An ideal zero-phonon line is Lorentzian (homogeneous, lifetime-limited). What the spectrometer records is that line
convolved with Gaussian broadening: the instrument response (physics JSON: instrument_resolution_meV) plus any spectral
diffusion or inhomogeneity, and unresolved structure looks the same way. The pseudo-Voigt nu of the fit mixes all of
these. Here every emitter line is isolated in each spectrum where it is strong (data minus background and every other
fitted component, clipped spectra skipped) and refitted locally as a true Voigt, Lorentzian (x) Gaussian, with the
Gaussian FWHM >= the instrument FWHM. Reported per emitter (medians over those spectra):
  lorentz_fwhm       homogeneous width (poorly constrained below ~0.5 x the instrument FWHM; flagged)
  gauss_fwhm         total Gaussian width, and gauss_extra_meV beyond the instrument
  nu_intrinsic       0 = Lorentzian ... 1 = Gaussian, of the line with the instrument response removed: the direction
                     towards an ideal ZPL (nu -> 0). A preference in the ranking, never a gate.
  broadened          extra Gaussian > the instrument resolution: spectral diffusion or unresolved structure; confirm
                     with a high-resolution spectrum.
"""
import numpy as np

from . import basis as B, physics

HC = 1239.84193
LN2 = np.log(2.0)
MIN_SNR = 50.0          # spectra where the line's peak is >= 50 sigma
MIN_SHARE = 0.10        # ... and it carries >= 10 % of the line area
MIN_SPECTRA = 5


def _voigt(x, a, c, fl, fg, b0, b1):
    from scipy.special import voigt_profile
    s = fg / (2.0 * np.sqrt(2.0 * LN2)); g = 0.5 * fl
    return a * voigt_profile(x - c, s, g) / voigt_profile(0.0, s, g) + b0 + b1 * (x - c)


def voigt_fwhm(fl, fg):
    """Olivero-Longbothum approximation (0.02 % accuracy)."""
    return 0.5346 * fl + np.sqrt(0.2166 * fl ** 2 + fg ** 2)


def nu_from_widths(fl, fg):
    """Gaussian fraction (fastdecov nu convention: 0 Lorentzian, 1 Gaussian) of the pseudo-Voigt equivalent to a
    Voigt with Lorentzian FWHM fl and Gaussian FWHM fg (Thompson-Cox-Hastings)."""
    if fl <= 0 and fg <= 0:
        return float("nan")
    f = (fg ** 5 + 2.69269 * fg ** 4 * fl + 2.42843 * fg ** 3 * fl ** 2 + 4.47163 * fg ** 2 * fl ** 3
         + 0.07842 * fg * fl ** 4 + fl ** 5) ** 0.2
    r = fl / f
    eta = 1.36603 * r - 0.47719 * r ** 2 + 0.11116 * r ** 3          # Lorentzian fraction
    return float(min(max(1.0 - eta, 0.0), 1.0))


def analyse(res, phys=None, kinds=None):
    """Per emitter line: dict with the Voigt decomposition, or None if too few strong, unclipped spectra."""
    from scipy.optimize import curve_fit
    phys = phys or physics.load()
    snr = res.heights / res.sigma[:, None]
    clipped = np.array([not B.clip_keep(np.asarray(y)).all() for y in res.ys])
    out = []
    for k in range(res.K):
        if kinds is not None and kinds[k] != "emitter line":
            out.append(None); continue
        S = np.flatnonzero((snr[:, k] >= MIN_SNR) & (res.share[:, k] >= MIN_SHARE) & ~clipped)
        rows = []
        for s in S:
            x = np.asarray(res.xs[s], float); y = np.asarray(res.ys[s], float); kp = res.keeps[s]
            tot, bg, zpl, sb = res.curve(s, parts=True)
            line = y - (tot - zpl[k])                    # data minus background, other lines and every sideband
            c, w = res.table[s, k, 0], res.table[s, k, 2]
            inst = physics.instrument_fwhm_nm(c, phys)
            sel = kp & (np.abs(x - c) <= max(2.5 * w, 3.0 * inst))
            if sel.sum() < 9:
                continue
            xs, ys = x[sel], line[sel]
            wgt = 1.0 / np.sqrt(res.sigma[s] ** 2 + np.maximum(y[sel], 0.0))   # read + shot noise
            a0 = float(ys.max())
            try:
                p, _ = curve_fit(_voigt, xs, ys, p0=[a0, c, 0.3 * w, max(0.8 * w, inst * 1.01), 0.0, 0.0],
                                 bounds=([0.0, c - w, 1e-4, inst, -np.inf, -np.inf], [np.inf, c + w, 3 * w + 1, 3 * w + 1, np.inf, np.inf]),
                                 sigma=1.0 / wgt, maxfev=5000)
            except Exception:
                continue
            fl, fg = float(p[2]), float(p[3])
            extra = float(np.sqrt(max(fg ** 2 - inst ** 2, 0.0)))
            rows.append((fl, fg, extra, nu_from_widths(fl, extra), inst, float(p[1])))
        if len(rows) < MIN_SPECTRA:
            out.append(None); continue
        r = np.array(rows)
        c = float(np.median(r[:, 5])); inst = float(np.median(r[:, 4]))
        to_meV = lambda nm: float(nm / c ** 2 * HC * 1000.0)
        fl, fg, ex, nu = (float(np.median(r[:, i])) for i in range(4))
        out.append(dict(spectra=int(len(r)), lorentz_fwhm_nm=round(fl, 3), lorentz_fwhm_meV=round(to_meV(fl), 2),
                        gauss_fwhm_nm=round(fg, 3), gauss_extra_nm=round(ex, 3), gauss_extra_meV=round(to_meV(ex), 2),
                        instrument_fwhm_nm=round(inst, 3), nu_intrinsic=round(nu, 3),
                        lorentz_below_resolution=bool(fl < 0.5 * inst),
                        broadened=bool(to_meV(ex) > phys["instrument_resolution_meV"])))
    return out


def markdown(rows, ranking_rows):
    have = [(r, v) for r, v in zip(ranking_rows, rows) if v]
    if not have:
        return ""
    L = ["", "Line shape (Voigt decomposition, report only): Lorentzian (homogeneous) FWHM, Gaussian FWHM >= instrument, "
         "extra Gaussian beyond the instrument, and nu_intrinsic = nu with the instrument removed (0 Lorentzian = ideal "
         "ZPL ... 1 Gaussian). Lower nu_intrinsic is preferred among g2 candidates; it is a direction, not a gate. "
         "'broadened': extra Gaussian > instrument resolution (spectral diffusion or unresolved structure; confirm with a "
         "high-resolution spectrum).", "",
         "| # | centre (nm) | spectra | Lorentz FWHM (meV) | Gaussian FWHM (nm) | extra Gaussian (meV) | nu (fit) | nu_intrinsic | notes |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r, v in have:
        notes = ", ".join(n for n, f in (("broadened", v["broadened"]), ("Lorentz width below resolution", v["lorentz_below_resolution"])) if f)
        L.append(f"| {r['emitter']} | {r['center_nm']} | {v['spectra']} | {v['lorentz_fwhm_meV']} | {v['gauss_fwhm_nm']} | "
                 f"{v['gauss_extra_meV']} | {r['nu_median']} | {v['nu_intrinsic']} | {notes or '–'} |")
    return "\n".join(L) + "\n"
