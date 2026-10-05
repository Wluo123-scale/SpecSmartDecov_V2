"""Legacy SpecSmartDecov model: 12 pseudo-Voigt components + constant background.

Mirrors legacy/{IGgen,BoundGen,sumvoigt,multiv,writer}.m with IG='auto', bg='bg1', mode=1.

Parameter vector per spectrum (length 1 + 4*NP):
    [bg, a_1, c_1, nu_1, s_1, ..., a_12, c_12, nu_12, s_12]
a = peak height, c = centre (nm), nu = Gaussian fraction (0 = Lorentz, 1 = Gauss), s = FWHM (nm).

sumvoigt.m evaluates a*(f1 + f2)/f3. The 1/s factors cancel, leaving
    a * (w*G(u) + (1-w)*L(u)),  u = (x-c)/s,
    G = exp(-4 ln2 u^2),  L = 1/(1 + 4u^2),  w = nu*K / (nu*K + (1-nu)/pi),  K = sqrt(ln2/pi).
"""
import numpy as np

ANCHORS_NM = np.array([611, 617, 627, 672, 680, 684, 689, 700, 707, 710, 717, 724], dtype=float)
NP = len(ANCHORS_NM)
NPAR = 1 + 4 * NP

# rdnpeak.m resets rng(0,'twister') before every draw, so its "random" offset is a constant:
# first rand() of MT19937 seeded with 0.
MATLAB_RNG0_FIRST_RAND = 0.8147236863931789
RDNPEAK_HALF_WINDOW_NM = 0.5
CENTER_REL_BOUND = 0.005   # BoundGen peakerrbound
HEIGHT_REL_BOUND = 0.1     # BoundGen scaleerrbound
WIDTH_DEPRESSION = 4.0     # BoundGen depression
MIN_WIDTH_NM = 1e-6        # legacy lower bound is 0; keep strictly positive to avoid 0/0
MIN_HEIGHT_UB = 1e-6       # interp1 can return a negative height -> legacy ub < lb; clamp

LN2 = np.log(2.0)
K_GAUSS = np.sqrt(LN2 / np.pi)
INV_PI = 1.0 / np.pi


def initial_centers(half_window=RDNPEAK_HALF_WINDOW_NM):
    h = half_window
    return ANCHORS_NM + h - 2.0 * h * MATLAB_RNG0_FIRST_RAND


def load_spectrum(path):
    """fscanf('%f %f') + fitV 'Flipper': ascending wavelength."""
    v = np.loadtxt(path, dtype=float, ndmin=2)[:, :2]
    if v[0, 0] >= v[-1, 0]:
        v = v[::-1]
    return np.ascontiguousarray(v[:, 0]), np.ascontiguousarray(v[:, 1])


def setup(x, y, width_start="mid", half_window=RDNPEAK_HALF_WINDOW_NM):
    """Initial guess and bounds (IGgen + BoundGen).

    The legacy width guess equals its upper bound; MATLAB's trust-region solver moves
    on-bound components to the box centre, which is what width_start='mid' reproduces.
    A float gives an explicit starting FWHM in nm. half_window is the rdnpeak.m search
    half-width (0.5 nm in the frozen code; some legacy runs used 1.0-3.0 nm).
    """
    c0 = initial_centers(half_window)
    a0 = np.interp(c0, x, y)
    wmax = (x[-1] - x[0]) / WIDTH_DEPRESSION
    p0 = np.empty(NPAR); lb = np.empty(NPAR); ub = np.empty(NPAR)
    p0[0], lb[0], ub[0] = abs(y.min()), 0.0, y.max()
    pk0 = p0[1:].reshape(NP, 4); pkl = lb[1:].reshape(NP, 4); pku = ub[1:].reshape(NP, 4)
    pk0[:, 0], pkl[:, 0], pku[:, 0] = a0, 0.0, np.maximum(a0 * (1 + HEIGHT_REL_BOUND), MIN_HEIGHT_UB)
    pk0[:, 1], pkl[:, 1], pku[:, 1] = c0, c0 * (1 - CENTER_REL_BOUND), c0 * (1 + CENTER_REL_BOUND)
    pk0[:, 2], pkl[:, 2], pku[:, 2] = 0.5, 0.0, 1.0
    pkl[:, 3], pku[:, 3] = MIN_WIDTH_NM, wmax
    pk0[:, 3] = 0.5 * (MIN_WIDTH_NM + wmax) if width_start == "mid" else min(float(width_start), wmax)
    # any other component on/outside its box -> box centre (MATLAB startx behaviour)
    bad = (p0 <= lb) | (p0 >= ub)
    p0[bad] = 0.5 * (lb[bad] + ub[bad])
    return p0, lb, ub


def components(p, x):
    """Per-component curves, shape (..., NP, n). p shape (..., NPAR), x shape (..., n)."""
    pk = p[..., 1:].reshape(p.shape[:-1] + (NP, 4))
    a, c, nu, s = (pk[..., i, None] for i in range(4))
    nu = np.clip(nu, 0.0, 1.0)
    s = np.abs(s)
    u = (x[..., None, :] - c) / s
    w = nu * K_GAUSS / (nu * K_GAUSS + (1 - nu) * INV_PI)
    return a * (w * np.exp(-4 * LN2 * u * u) + (1 - w) / (1 + 4 * u * u))


def evaluate(p, x):
    return p[..., :1] + components(p, x).sum(axis=-2)


def areas(p):
    """writer.m: area = 0.5*a / (nu*sqrt(ln2/pi)/|s| + (1-nu)/(pi*|s|))."""
    pk = p[..., 1:].reshape(p.shape[:-1] + (NP, 4))
    a, nu, s = pk[..., 0], pk[..., 2], np.abs(pk[..., 3])
    return 0.5 * a / (nu * K_GAUSS / s + (1 - nu) * INV_PI / s)


# --------------------------------------------------------------------------- general basis
# A Basis is the set of principal emission lines (the paper's principal components P in M = W x P):
# how many lines, where they may sit, how wide they may be, and the background type.
# Parameter layout for a basis with K lines and NBG background terms:
#     [b0, (b1), a_1, c_1, nu_1, s_1, ..., a_K, c_K, nu_K, s_K]
# background = b0 + b1 * (x - X_REF) / X_SCALE   (b1 only when background == "linear")
X_REF, X_SCALE = 700.0, 100.0


class Basis:
    def __init__(self, centers, center_lo=None, center_hi=None, width_min=1.0, width_max=None,
                 background="const", height_rule="free", name="custom", meta=None):
        self.centers = np.asarray(centers, float)
        k = self.centers.size
        self.center_lo = np.asarray(center_lo if center_lo is not None else self.centers - 1.0, float)
        self.center_hi = np.asarray(center_hi if center_hi is not None else self.centers + 1.0, float)
        self.width_min = np.broadcast_to(np.asarray(width_min, float), (k,)).copy()
        self.width_max = None if width_max is None else np.broadcast_to(np.asarray(width_max, float), (k,)).copy()
        self.background = background            # "const" | "linear"
        self.height_rule = height_rule          # "legacy" (<= 1.1 * interpolated y) | "free"
        self.name, self.meta = name, dict(meta or {})

    @property
    def K(self):
        return self.centers.size

    @property
    def nbg(self):
        return 2 if self.background == "linear" else 1

    @property
    def npar(self):
        return self.nbg + 4 * self.K

    @classmethod
    def legacy(cls, half_window=RDNPEAK_HALF_WINDOW_NM):
        """The frozen SpecSmartDecov pillar-9 basis: 12 anchors, +-0.5 % centre boxes, constant background."""
        c0 = initial_centers(half_window)
        return cls(c0, c0 * (1 - CENTER_REL_BOUND), c0 * (1 + CENTER_REL_BOUND), width_min=MIN_WIDTH_NM,
                   background="const", height_rule="legacy", name="legacy-12")

    def to_dict(self):
        return dict(name=self.name, background=self.background, height_rule=self.height_rule,
                    centers=self.centers.tolist(), center_lo=self.center_lo.tolist(), center_hi=self.center_hi.tolist(),
                    width_min=self.width_min.tolist(), width_max=None if self.width_max is None else self.width_max.tolist(),
                    meta=self.meta)

    @classmethod
    def from_dict(cls, d):
        return cls(d["centers"], d["center_lo"], d["center_hi"], d["width_min"], d.get("width_max"),
                   d.get("background", "const"), d.get("height_rule", "free"), d.get("name", "custom"), d.get("meta"))


def setup_basis(x, y, basis, width_start="mid", baseline=None):
    """Seeds and bounds for one spectrum under a Basis. baseline: optional (b0, b1) seed."""
    K, nbg = basis.K, basis.nbg
    wmax_default = (x[-1] - x[0]) / WIDTH_DEPRESSION
    wmax = basis.width_max if basis.width_max is not None else np.full(K, wmax_default)
    p0 = np.empty(basis.npar); lb = np.empty(basis.npar); ub = np.empty(basis.npar)
    yr = max(float(y.max() - y.min()), 1.0)
    if baseline is None:
        baseline = (abs(float(y.min())), 0.0) if nbg == 1 else _edge_baseline(x, y)
    p0[0], lb[0], ub[0] = baseline[0], (0.0 if nbg == 1 else -yr), float(y.max())
    if nbg == 2:
        p0[1], lb[1], ub[1] = baseline[1], -2 * yr, 2 * yr
    t = (x - X_REF) / X_SCALE
    base = baseline[0] + (baseline[1] * t if nbg == 2 else 0.0)
    a0 = np.interp(basis.centers, x, y - base)
    pk0 = p0[nbg:].reshape(K, 4); pkl = lb[nbg:].reshape(K, 4); pku = ub[nbg:].reshape(K, 4)
    pkl[:, 0] = 0.0
    if basis.height_rule == "legacy":                 # IGgen/BoundGen: seed from raw intensity, ub = 1.1 x seed
        a0 = np.interp(basis.centers, x, y)
        pk0[:, 0] = a0; pku[:, 0] = np.maximum(a0 * (1 + HEIGHT_REL_BOUND), MIN_HEIGHT_UB)
    else:
        pk0[:, 0] = np.maximum(a0, 0.0); pku[:, 0] = max(2.0 * yr, 1.0)
    pk0[:, 1], pkl[:, 1], pku[:, 1] = basis.centers, basis.center_lo, basis.center_hi
    pk0[:, 2], pkl[:, 2], pku[:, 2] = 0.5, 0.0, 1.0
    pkl[:, 3], pku[:, 3] = basis.width_min, wmax
    pk0[:, 3] = 0.5 * (basis.width_min + wmax) if width_start == "mid" else np.minimum(float(width_start), wmax)
    bad = (p0 <= lb) | (p0 >= ub)
    p0[bad] = 0.5 * (lb[bad] + ub[bad])
    return p0, lb, ub


def _edge_baseline(x, y):
    """Line through the medians of the first and last 8 % of the window (emission-poor ends)."""
    n = max(5, int(0.08 * x.size))
    t = (x - X_REF) / X_SCALE
    tl, tr, yl, yr = np.median(t[:n]), np.median(t[-n:]), np.median(y[:n]), np.median(y[-n:])
    b1 = (yr - yl) / (tr - tl) if tr != tl else 0.0
    return yl - b1 * tl, b1


def components_k(p, x, K, nbg):
    pk = p[..., nbg:].reshape(p.shape[:-1] + (K, 4))
    a, c, nu, s = (pk[..., i, None] for i in range(4))
    nu = np.clip(nu, 0.0, 1.0); s = np.abs(s)
    u = (x[..., None, :] - c) / s
    w = nu * K_GAUSS / (nu * K_GAUSS + (1 - nu) * INV_PI)
    return a * (w * np.exp(-4 * LN2 * u * u) + (1 - w) / (1 + 4 * u * u))


def background_k(p, x, nbg):
    b = p[..., :1] + 0.0 * x
    if nbg == 2:
        b = b + p[..., 1:2] * (x - X_REF) / X_SCALE
    return b


def evaluate_k(p, x, K, nbg):
    return background_k(p, x, nbg) + components_k(p, x, K, nbg).sum(axis=-2)


def areas_k(p, K, nbg):
    pk = p[..., nbg:].reshape(p.shape[:-1] + (K, 4))
    a, nu, s = pk[..., 0], pk[..., 2], np.abs(pk[..., 3])
    return 0.5 * a / (nu * K_GAUSS / s + (1 - nu) * INV_PI / s)
