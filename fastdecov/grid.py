# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Map shape (grid) detection for `--grid auto`.

1. File names. If every spectrum id holds two integers that form a complete rows x columns grid (e.g. `x12_y07`,
   `r3c5`, `0012_0007`), those are the pixel coordinates.
2. Spatial continuity. Otherwise the shape is inferred from the spectra in file order. For each candidate width W
   (W divides N, both sides >= 3) and scan order (row by row, or serpentine), the denoised spectral distance between
   each pixel and the pixel "below" it is compared with the distance between neighbours in the same row:
       ratio(W) = median d(i, below(i)) / median d(i, i + 1 within a row).
   Neighbours in a measured map are alike, so the true shape gives the smallest ratio; a wrong width puts pixels that
   are several steps apart "below" each other. The result is accepted only if the runner-up ratio is >= `margin`
   times the best one; otherwise the map is reported as ambiguous and no grid is used.

Row-by-row and column-by-column file order cannot be told apart (the map is only transposed); the shape is reported
as rows x columns in file order.
"""
import re

import numpy as np

MARGIN = 1.10
N_COMPONENTS = 8


def _from_names(ids):
    nums = [re.findall(r"\d+", str(s)) for s in ids]
    if not nums or any(len(n) < 2 for n in nums):
        return None
    pairs = [(int(n[-2]), int(n[-1])) for n in nums]
    rows, cols = sorted({p[0] for p in pairs}), sorted({p[1] for p in pairs})
    if len(rows) < 2 or len(cols) < 2 or len(set(pairs)) != len(pairs) or len(rows) * len(cols) != len(pairs):
        return None
    ri = {v: i for i, v in enumerate(rows)}; ci = {v: i for i, v in enumerate(cols)}
    order = [ri[a] * len(cols) + ci[b] for a, b in pairs]
    return dict(rows=len(rows), cols=len(cols), order=order, snake=False, method="file names", confidence=None,
                candidates=[])


def _features(xs, ys, npc=N_COMPONENTS):
    x0 = np.sort(np.asarray(xs[0], float))
    Y = np.array([np.interp(x0, np.sort(np.asarray(x, float)), np.asarray(y, float)[np.argsort(x)]) for x, y in zip(xs, ys)])
    Y = Y - Y.mean(0)
    U, S, _ = np.linalg.svd(Y, full_matrices=False)
    k = min(npc, len(S))
    return U[:, :k] * S[:k]


def _below(i, W, snake):
    r, c = divmod(i, W)
    return (r + 1) * W + ((W - 1 - c) if snake else c)


def _ratio(Z, W, snake):
    N = len(Z)
    vb = [np.linalg.norm(Z[i] - Z[_below(i, W, snake)]) for i in range(N - W)]
    rh = [np.linalg.norm(Z[i] - Z[i + 1]) for i in range(N - 1) if (i + 1) % W]
    return float(np.median(vb) / max(np.median(rh), 1e-12))


def order_for(rows, cols, snake):
    """Position (row-major index in the image) of each spectrum in file order."""
    idx = np.arange(rows * cols).reshape(rows, cols)
    if snake:
        idx[1::2] = idx[1::2, ::-1]
    return [int(v) for v in idx.ravel()]


def detect(ids, xs, ys, margin=MARGIN):
    """Returns dict(rows, cols, order, snake, method, confidence, candidates), or dict(rows=None, reason=...)."""
    named = _from_names(ids)
    if named:
        return named
    N = len(ids)
    widths = [W for W in range(3, N // 3 + 1) if N % W == 0]
    if not widths:
        return dict(rows=None, reason=f"{N} spectra cannot form a grid with both sides >= 3")
    Z = _features(xs, ys)
    cands = sorted((_ratio(Z, W, sn), W, sn) for W in widths for sn in (False, True))
    best = cands[0]
    conf = (cands[1][0] / best[0]) if len(cands) > 1 else np.inf
    table = [dict(shape=f"{N // W}x{W}", serpentine=sn, ratio=round(r, 3)) for r, W, sn in cands[:5]]
    if conf < margin:
        return dict(rows=None, reason=f"ambiguous: best {table[0]['shape']} ratio {best[0]:.2f}, runner-up "
                                      f"{table[1]['shape']} {cands[1][0]:.2f} (margin {conf:.2f} < {margin})",
                    candidates=table)
    rows, cols = N // best[1], best[1]
    return dict(rows=rows, cols=cols, order=order_for(rows, cols, best[2]), snake=bool(best[2]),
                method="spatial continuity", confidence=round(float(conf), 3), candidates=table)


def fixed(rows, cols, snake=False):
    return dict(rows=rows, cols=cols, order=order_for(rows, cols, snake), snake=snake, method="given", confidence=None,
                candidates=[])


def image(values, g):
    """Arrange per-spectrum values (file order) into the rows x cols map."""
    img = np.full(g["rows"] * g["cols"], np.nan)
    img[np.asarray(g["order"])] = np.asarray(values, float)
    return img.reshape(g["rows"], g["cols"])
