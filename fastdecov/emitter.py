# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""V3 emitter model: linear background + K emitters, each a zero-phonon line (ZPL) with an optional tied phonon sideband.

ZPL (pseudo-Voigt, as V1/V2): a, c, nu, s.  Phonon sideband (PSB, red side, tied to its ZPL):
    PSB(x) = r * a * exp(-4 ln2 v^2),  v = (x - c - 0.5*s - delta) / (s + dw)
so it has no amplitude without its ZPL, shares its spatial map, sits outside the ZPL half-width on the red side
(0.5*s + delta) and is at least as broad as the ZPL. Per-emitter parameters: [a, c, nu, s] or, with a sideband, [a, c, nu, s, r, delta, dw].
Background p[0] + p[1] * (x - 700) / 100.
"""
import numpy as np

from . import model

LN2 = np.log(2.0)
PSB_AREA = np.sqrt(np.pi / (4.0 * LN2))      # area of a unit-height Gaussian per unit FWHM


def offsets(psb):
    off, q = [], 2
    for f in psb:
        off.append(q); q += 7 if f else 4
    return np.array(off, int), q


def evaluate(p, x, psb, parts=False):
    off, _ = offsets(psb)
    t = (x - model.X_REF) / model.X_SCALE
    bg = p[0] + p[1] * t
    zpl, sb = [], []
    for k, f in enumerate(psb):
        a, c, nu, s = p[off[k]:off[k] + 4]
        zpl.append(model.components_k(np.r_[0.0, 0.0, a, c, nu, s], x, 1, 2)[0])
        if f:
            r, de, dw = p[off[k] + 4:off[k] + 7]
            v = (x - c - 0.5 * s - de) / (s + dw)
            sb.append(r * a * np.exp(-4 * LN2 * v * v))
        else:
            sb.append(np.zeros_like(x))
    tot = bg + np.sum(zpl, axis=0) + np.sum(sb, axis=0)
    return (tot, bg, np.array(zpl), np.array(sb)) if parts else tot


def table(p, psb):
    """Per emitter: ZPL centre, nu, FWHM, ZPL area, PSB area, PSB offset from ZPL centre, PSB FWHM, ZPL fraction."""
    off, _ = offsets(psb)
    rows = []
    for k, f in enumerate(psb):
        a, c, nu, s = p[off[k]:off[k] + 4]
        az = float(model.areas_k(np.r_[0.0, 0.0, a, c, nu, s], 1, 2)[0])
        if f:
            r, de, dw = p[off[k] + 4:off[k] + 7]
            ap = float(r * a * (s + dw) * PSB_AREA)
            rows.append((c, nu, s, az, ap, 0.5 * s + de, s + dw))
        else:
            rows.append((c, nu, s, az, 0.0, np.nan, np.nan))
    out = np.array(rows, float)
    zf = out[:, 3] / np.maximum(out[:, 3] + out[:, 4], 1e-300)
    return np.column_stack([out, zf])
