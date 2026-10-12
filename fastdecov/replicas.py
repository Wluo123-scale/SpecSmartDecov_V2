"""Residual-feature diagnostic: is a recurring residual bump a phonon replica of a fitted emitter, or a separate
emitter? Nothing is fitted away: the fit is unchanged, the bumps are only classified and reported.

1. Bumps. In every spectrum, positive residual peaks of >= MIN_SNR noise sigma, at least 0.6 x the instrument FWHM
   wide, away from masked pixels and from the core of every fitted line, are fitted locally with a Gaussian on a
   linear residual baseline (centre, height, FWHM, area).
2. Features. Bumps at the same wavelength (within half the strain-shift range) in >= MIN_SPECTRA spectra form one
   feature.
3. Tests against every emitter line on its blue side (the bump at lower energy), over the spectra where the bump is
   detected. A replica must pass all four map criteria:
     presence  - the emitter is present (>= 5 % of its maximum area) wherever the bump is detected;
     offset    - the energy offset emitter - bump is constant: interquartile range <= 0.5 x instrument resolution;
     tracking  - the bump follows the emitter's strain shift: slope of bump energy on emitter energy within
                 [0.7, 1.3] (only tested when the emitter's position varies by >= 0.25 x instrument resolution);
     width     - bump FWHM / emitter FWHM within [0.7, 2.0].
   Two physics gates make the four specific when the emitter is present everywhere or barely shifts:
     energy    - the offset lies between 2 x the instrument resolution (a discrete line, not the acoustic sideband
                 tied to the ZPL) and the maximum optical-phonon energy (physics JSON: optical_phonon_max_meV);
     intensity - the bump area rank-correlates with the emitter's area over the map (Spearman >= 0.5): a replica's
                 intensity is proportional to its parent's. The area ratio itself is reported, not gated (a weak
                 replica is near the detection limit, so the ratio is noisy).
   The presence test only means something if the emitter is absent somewhere: if it is absent from < 10 % of the map,
   two independent map-wide emitters sharing the strain field can pass every test. Such a feature is reported as a
   "possible replica (presence test uninformative)", never as a replica.
   A discrete line at a constant offset that passes all four is reported as an optical-phonon replica of that
   emitter, with the offset in meV and the replica/ZPL area ratio. A feature detected where no candidate emitter is
   present in >= 20 % of its spectra is reported as a possible separate emitter; anything else as unclassified.
"""
import numpy as np

from . import basis as B, physics

HC = 1239.84193                      # eV nm
MIN_SNR = 5.0
MIN_SPECTRA = 5
TEST = dict(presence=0.95, offset_iqr_x_res=0.5, slope=(0.7, 1.3), track_min_x_res=0.25, width_ratio=(0.7, 2.0),
            offset_min_x_res=2.0, intensity_rank_corr=0.5, parent_absent_min=0.10,
            present_area=0.05, core_x_fwhm=1.5, mask_gap_px=2, width_x_instrument=0.6)


def _gauss(x, h, c, w, a0, a1):
    return h * np.exp(-4.0 * np.log(2.0) * ((x - c) / w) ** 2) + a0 + a1 * (x - c)


def _bumps(res, s, phys):
    """Positive residual bumps of spectrum s: list of (centre_nm, height, fwhm_nm, area, snr)."""
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks, peak_widths
    x = np.asarray(res.xs[s], float); y = np.asarray(res.ys[s], float); k = res.keeps[s]
    o = np.argsort(x); x, y, k = x[o], y[o], k[o]
    r = np.where(k, y - res.curve(s)[o], 0.0); sg = res.sigma[s]
    step = float(np.median(np.diff(x)))
    sm = np.convolve(r, np.ones(3) / 3, mode="same")
    pk, _ = find_peaks(sm, height=MIN_SNR * sg)
    if pk.size == 0:
        return []
    widths = peak_widths(sm, pk, rel_height=0.5)[0] * step
    masked = np.flatnonzero(~k)
    tab = res.table[s]
    out = []
    for i, w0 in zip(pk, widths):
        c0 = float(x[i]); inst = physics.instrument_fwhm_nm(c0, PHYS)
        if w0 < TEST["width_x_instrument"] * inst:
            continue
        if masked.size and np.min(np.abs(masked - i)) <= TEST["mask_gap_px"]:
            continue
        if any(tab[q, 3] > 0 and abs(c0 - tab[q, 0]) < TEST["core_x_fwhm"] * tab[q, 2] for q in range(len(tab))):
            continue                                    # inside a fitted line's core: shape residual, not a new line
        sel = k & (np.abs(x - c0) <= max(3.0 * w0, 3.0 * inst))
        if sel.sum() < 7:
            continue
        try:
            p, _ = curve_fit(_gauss, x[sel], r[sel], p0=[max(r[i], sg), c0, max(w0, inst), 0.0, 0.0],
                             bounds=([0.0, c0 - 2 * inst, 0.5 * inst, -np.inf, -np.inf],
                                     [np.inf, c0 + 2 * inst, 4.0 * max(w0, inst), np.inf, np.inf]), maxfev=5000)
        except Exception:
            continue
        h, c, w = p[:3]
        if h < MIN_SNR * sg:
            continue
        out.append((float(c), float(h), float(w), float(h * w * 1.0645), float(h / sg)))
    return out


PHYS = physics.load()


def _test_parent(res, S, k, bc, bw, ba, present, area, res_meV):
    """The six replica tests of a feature (per-spectrum centres bc, FWHMs bw, areas ba over spectra S) against
    emitter k as its parent."""
    ec = res.table[S, k, 0]; ew = res.table[S, k, 2]; ea = area[S, k]
    pres = float(np.mean(present[S, k]))
    on = present[S, k]
    if on.sum() < MIN_SPECTRA:
        return dict(emitter=k + 1, emitter_center_nm=round(float(np.median(ec)), 2), presence=round(pres, 3),
                    passed=False, reason="emitter absent in most spectra of the feature")
    Ee, Eb = HC / ec[on], HC / bc[on]
    dE = (Ee - Eb) * 1000.0
    iqr = float(np.subtract(*np.percentile(dE, [75, 25])))
    spread = float(np.subtract(*np.percentile(Ee * 1000.0, [75, 25])))
    testable = spread >= TEST["track_min_x_res"] * res_meV
    slope = float(np.polyfit(Ee, Eb, 1)[0]) if testable else None
    wr = float(np.median(bw[on] / ew[on]))
    rank = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rank(ba[on]), rank(ea[on]))[0, 1]) if on.sum() >= 3 else None
    ratio = ba[on] / np.maximum(ea[on], 1e-300)
    off = float(np.median(dE)); emax = PHYS.get("optical_phonon_max_meV")
    t = dict(emitter=k + 1, emitter_center_nm=round(float(np.median(ec)), 2), presence=round(pres, 3),
             offset_meV=round(off, 2), offset_iqr_meV=round(iqr, 2), emitter_shift_iqr_meV=round(spread, 2),
             tracking_slope=None if slope is None else round(slope, 3), tracking_tested=bool(testable),
             width_ratio=round(wr, 3), area_ratio_median=round(float(np.median(ratio)), 5),
             area_ratio_cv=round(float(np.std(ratio) / max(np.mean(ratio), 1e-300)), 3),
             area_rank_corr=None if rho is None else round(rho, 3))
    checks = dict(presence=pres >= TEST["presence"], offset=iqr <= TEST["offset_iqr_x_res"] * res_meV,
                  tracking=(not testable) or (TEST["slope"][0] <= slope <= TEST["slope"][1]),
                  width=TEST["width_ratio"][0] <= wr <= TEST["width_ratio"][1],
                  energy=off >= TEST["offset_min_x_res"] * res_meV and (emax is None or off <= emax),
                  intensity=rho is not None and rho >= TEST["intensity_rank_corr"])
    t["checks"] = {k2: bool(v) for k2, v in checks.items()}
    absent = float(np.mean(~present[:, k]))
    t["emitter_absent_fraction"] = round(absent, 3)
    t["presence_informative"] = bool(absent >= TEST["parent_absent_min"])
    t["passed"] = all(checks.values()) and t["presence_informative"]
    t["possible"] = all(checks.values()) and not t["presence_informative"]
    return t


def fitted_line_replicas(res, phys=None, kinds=None):
    """Fitted emitter lines that are optical-phonon replicas of a bluer fitted line (same six tests, on the fitted
    centres, widths and areas). Returns {replica emitter index (0-based): test dict of its parent}."""
    global PHYS
    PHYS = phys or physics.load()
    res_meV = PHYS["instrument_resolution_meV"]
    area = res.table[..., 3]
    present = area >= TEST["present_area"] * np.maximum(area.max(0, keepdims=True), 1e-300)
    out = {}
    lines = [k for k in range(res.K) if kinds is None or kinds[k] == "emitter line"]
    for b in lines:
        S = np.flatnonzero(present[:, b])
        if S.size < MIN_SPECTRA:
            continue
        bc, bw, ba = res.table[S, b, 0], res.table[S, b, 2], area[S, b]
        ok = []
        for a in lines:
            if a == b or np.median(res.table[S, a, 0]) >= np.median(bc):
                continue
            t = _test_parent(res, S, a, bc, bw, ba, present, area, res_meV)
            if t.get("passed"):
                ok.append(t)
        if ok:
            out[b] = min(ok, key=lambda t: t["offset_iqr_meV"])
    return out


def analyse(res, phys=None, kinds=None):
    """Classify recurring residual features. kinds: per emitter 'emitter line' / 'broad' (from report.ranking); only
    emitter lines are tested as replica parents. Returns a list of feature dicts."""
    global PHYS
    PHYS = phys or physics.load()
    res_meV = PHYS["instrument_resolution_meV"]
    gtol = 0.5 * PHYS["strain_shift_halfrange_nm"]
    det = []                                             # (centre, spectrum, height, fwhm, area, snr)
    for s in range(len(res.ids)):
        det += [(c, s, h, w, a, z) for c, h, w, a, z in _bumps(res, s, PHYS)]
    if not det:
        return []
    det.sort()
    groups, cur = [], [det[0]]
    for d in det[1:]:
        if d[0] - np.median([u[0] for u in cur]) <= gtol:
            cur.append(d)
        else:
            groups.append(cur); cur = [d]
    groups.append(cur)
    area = res.table[..., 3]
    present = area >= TEST["present_area"] * np.maximum(area.max(0, keepdims=True), 1e-300)
    feats = []
    for g in groups:
        best = {}
        for c, s, h, w, a, z in g:                      # one detection per spectrum (the strongest)
            if s not in best or z > best[s][4]:
                best[s] = (c, h, w, a, z)
        if len(best) < MIN_SPECTRA:
            continue
        S = np.array(sorted(best)); bc = np.array([best[s][0] for s in S]); bw = np.array([best[s][2] for s in S])
        ba = np.array([best[s][3] for s in S])
        f = dict(center_nm=round(float(np.median(bc)), 2), spectra=int(S.size), fwhm_nm=round(float(np.median(bw)), 2),
                 snr_median=round(float(np.median([best[s][4] for s in S])), 1), spectrum_ids=[res.ids[s] for s in S],
                 classification="unclassified", tests=[])
        cands = []
        for k in range(res.K):
            if kinds is not None and kinds[k] != "emitter line":
                continue
            if np.median(res.table[S, k, 0]) >= np.median(bc):
                continue                                 # replica must be on the red (low-energy) side
            cands.append(_test_parent(res, S, k, bc, bw, ba, present, area, res_meV))
        f["tests"] = cands
        ok = [t for t in cands if t.get("passed")]
        maybe = [t for t in cands if t.get("possible")]
        if not ok and maybe:
            t = min(maybe, key=lambda t: t["offset_iqr_meV"])
            f["classification"] = "possible replica (presence test uninformative)"
            f["parent_emitter"] = t["emitter"]; f["parent_center_nm"] = t["emitter_center_nm"]
            f["offset_meV"] = t["offset_meV"]; f["area_ratio"] = t["area_ratio_median"]
        elif ok:
            t = min(ok, key=lambda t: t["offset_iqr_meV"])
            f["classification"] = "optical-phonon replica"
            f["parent_emitter"] = t["emitter"]; f["parent_center_nm"] = t["emitter_center_nm"]
            f["offset_meV"] = t["offset_meV"]; f["area_ratio"] = t["area_ratio_median"]
        else:
            parents = [k for k in range(res.K) if (kinds is None or kinds[k] == "emitter line")
                       and np.median(res.table[S, k, 0]) < np.median(bc)]
            alone = float(np.mean(~present[S][:, parents].any(axis=1))) if parents else 1.0
            f["fraction_without_parent"] = round(alone, 3)
            if alone >= 0.2:
                f["classification"] = "possible separate emitter"
        feats.append(f)
    return feats


def markdown(feats):
    if not feats:
        return "\nResidual features: none recurring (>= %d spectra at >= %g sigma).\n" % (MIN_SPECTRA, MIN_SNR)
    L = ["", "Residual features (recurring bumps the fit leaves; classified against the map, not fitted). A phonon replica "
         "must: be present only where its emitter is, keep a constant energy offset (IQR <= %g x instrument resolution), "
         "follow the emitter's strain shift (slope 0.7-1.3), and match its width (0.7-2x); its offset must be an "
         "optical-phonon energy (2 x instrument resolution to optical_phonon_max_meV) and its intensity must track the "
         "emitter's over the map (Spearman >= 0.5)."
         % TEST["offset_iqr_x_res"], "",
         "| centre (nm) | spectra | FWHM (nm) | classification | parent emitter | offset (meV) | offset IQR (meV) | slope | width ratio | area ratio |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for f in feats:
        t = next((t for t in f["tests"] if t.get("emitter") == f.get("parent_emitter")), None)
        if t is None and f["tests"]:
            t = min((t for t in f["tests"] if "offset_iqr_meV" in t), key=lambda t: t["offset_iqr_meV"], default=None)
        g = (lambda k, d="–": d if t is None or t.get(k) is None else t[k])
        L.append(f"| {f['center_nm']} | {f['spectra']} | {f['fwhm_nm']} | {f['classification']} | "
                 f"{('#%d (%s nm)' % (f['parent_emitter'], f['parent_center_nm'])) if 'parent_emitter' in f else '–'} | "
                 f"{g('offset_meV')} | {g('offset_iqr_meV')} | {g('tracking_slope')} | {g('width_ratio')} | {g('area_ratio_median')} |")
    return "\n".join(L) + "\n"
