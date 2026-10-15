# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Follow-up shortlist: the screening result of a 150 g/mm map, ranked for high-resolution measurements.

fastdecov is a preliminary screen. At 150 g/mm the instrument (3.25 meV) hides the true line shape, so the shortlist
says which lines to re-measure with a higher grating, where on the map, over which window, and what that measurement
should answer.

Tiers (broad components and phonon replicas are never listed):
  1  g2 priority (strong, isolated, sharp), ordered by nu_intrinsic: the more Lorentzian after removing the instrument,
     the closer to an ideal zero-phonon line (a direction, not a gate);
  2  other strong emitter lines (peak >= 50 sigma): crowded, broadened or not sharp; the higher grating should resolve
     whether they are single emitters;
  3  possible separate emitters found in the residual (residual-feature diagnostic), not in the fit.
Grating: the coarsest listed grating whose resolution is <= half the line's estimated Lorentzian width (enough to
resolve the homogeneous line), else the finest one.
"""
import json

import numpy as np

from . import physics

TOP_PIXELS = 3


def _pixels(res, k, grid):
    snr = res.heights[:, k] / res.sigma
    score = snr * np.minimum(res.share[:, k] / 0.5, 1.0)        # strong and dominant in that spectrum
    best = np.argsort(-score)[:TOP_PIXELS]
    out = []
    for s in best:
        d = dict(spectrum_id=res.ids[s], line_snr=round(float(snr[s]), 1), share=round(float(res.share[s, k]), 2))
        if grid and grid.get("rows"):
            pos = grid["order"][s]; d["row"], d["col"] = divmod(int(pos), grid["cols"])
        out.append(d)
    return out


def _grating(lorentz_meV, phys):
    g = sorted((phys.get("follow_up_gratings_meV") or {}).items(), key=lambda t: -t[1])
    if not g:
        return None
    for name, r in g:
        if lorentz_meV is not None and r <= 0.5 * lorentz_meV:
            return name
    return g[-1][0]


def build(res, rows, feats, phys, grid=None):
    phys = phys or physics.load()
    items = []
    for k, r in enumerate(rows):
        if r["kind"] != "emitter line" or not r["strong"]:
            continue
        ls = r.get("line_shape") or {}
        tier = 1 if r["g2_priority"] else 2
        why, ask = [], []
        if tier == 1:
            why.append("strong, isolated and sharp")
        else:
            why += [w for w, f in (("not isolated", not r["isolated"]), ("not sharp", not r["sharp"])) if f]
        if ls.get("broadened"):
            why.append(f"extra Gaussian {ls['gauss_extra_meV']} meV")
            ask.append("is the Gaussian broadening spectral diffusion or unresolved lines?")
        if ls.get("lorentz_below_resolution"):
            ask.append("homogeneous (Lorentzian) width, below the 150 g/mm resolution")
        if not r["isolated"]:
            ask.append("separate it from its neighbours")
        if r["phonon_sideband"]:
            ask.append("sideband shape and ZPL fraction (Debye-Waller factor)")
        lw = ls.get("lorentz_fwhm_meV")
        c = r["center_nm"]; w = r["fwhm_nm"]
        items.append(dict(tier=tier, emitter=r["emitter"], center_nm=c, center_eV=round(1239.84193 / c, 4), fwhm_nm=w,
                          nu_fit=r["nu_median"], nu_intrinsic=ls.get("nu_intrinsic"), lorentz_fwhm_meV=lw,
                          gauss_extra_meV=ls.get("gauss_extra_meV"), max_snr=r["max_snr"],
                          window_nm=[round(c - max(4 * w, 5.0), 1), round(c + max(4 * w, 5.0) + (8.0 if r["phonon_sideband"] else 0.0), 1)],
                          grating=_grating(lw if lw else None, phys), best_pixels=_pixels(res, k, grid),
                          why=", ".join(why), questions=ask))
    for f in feats or []:
        if f["classification"] != "possible separate emitter":
            continue
        items.append(dict(tier=3, emitter=None, center_nm=f["center_nm"], center_eV=round(1239.84193 / f["center_nm"], 4),
                          fwhm_nm=f["fwhm_nm"], nu_fit=None, nu_intrinsic=None, lorentz_fwhm_meV=None, gauss_extra_meV=None,
                          max_snr=f["snr_median"], window_nm=[round(f["center_nm"] - 5, 1), round(f["center_nm"] + 5, 1)],
                          grating=_grating(None, phys),
                          best_pixels=[dict(spectrum_id=s) for s in f["spectrum_ids"][:TOP_PIXELS]],
                          why=f"residual bump in {f['spectra']} spectra with no parent line", questions=["is it a real emitter?"]))
    nu_key = lambda it: 1.0 if it["nu_intrinsic"] is None else it["nu_intrinsic"]
    items.sort(key=lambda it: (it["tier"], nu_key(it) if it["tier"] == 1 else 0.0, -(it["max_snr"] or 0)))
    for i, it in enumerate(items):
        it["rank"] = i + 1
    return items


def markdown(items, name):
    L = [f"# {name}: follow-up shortlist (screening at 150 g/mm)", "",
         "Lines to re-measure with a higher grating. Tier 1: g2 priority, ordered by nu_intrinsic (more Lorentzian after "
         "removing the instrument = closer to an ideal ZPL). Tier 2: other strong emitter lines. Tier 3: possible separate "
         "emitters left in the residual. Broad components and phonon replicas are not listed.", "",
         "| rank | tier | centre (nm / eV) | FWHM (nm) | nu fit → intrinsic | Lorentz (meV) | extra Gauss (meV) | window (nm) | grating | best pixels (row, col) | why | what to answer |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for it in items:
        px = "; ".join(f"{p['spectrum_id']}" + (f" ({p['row']}, {p['col']})" if "row" in p else "") for p in it["best_pixels"])
        nu = "–" if it["nu_fit"] is None else f"{it['nu_fit']} → {it['nu_intrinsic'] if it['nu_intrinsic'] is not None else '–'}"
        L.append(f"| {it['rank']} | {it['tier']} | {it['center_nm']} / {it['center_eV']} | {it['fwhm_nm']} | {nu} | "
                 f"{it['lorentz_fwhm_meV'] if it['lorentz_fwhm_meV'] is not None else '–'} | "
                 f"{it['gauss_extra_meV'] if it['gauss_extra_meV'] is not None else '–'} | {it['window_nm'][0]}–{it['window_nm'][1]} | "
                 f"{it['grating'] or '–'} | {px} | {it['why']} | {'; '.join(it['questions']) or '–'} |")
    return "\n".join(L) + "\n"


def save(out, items, name):
    from pathlib import Path
    (Path(out) / "shortlist.json").write_text(json.dumps(dict(map=name, items=items), indent=1, default=float) + "\n")
    (Path(out) / "shortlist.md").write_text(markdown(items, name))
