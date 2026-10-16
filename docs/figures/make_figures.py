# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Draw the README figures from recorded numbers only, in a light and a dark version each.

    python docs/figures/make_figures.py

Inputs: results/summary/metrics.json (speed), docs/figures/synthetic_validation.json (held-out synthetic maps).
Outputs: docs/figures/*_{light,dark}.png.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = json.loads((ROOT / "results" / "summary" / "metrics.json").read_text())

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df", neutral="#9a9893",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a", band="#f1efe9"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#34332f", neutral="#77756f",
                 s1="#3987e5", s2="#d95926", s3="#199e70", band="#25241f"),
}


def style(ax, t, xgrid=True):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    if xgrid:
        ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def figure(t, w, h, ncols=1, **kw):
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), **kw)
    fig.patch.set_facecolor(t["surface"])
    return fig, axes


def save(fig, name, theme):
    fig.savefig(HERE / f"{name}_{theme}.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def speed(t, theme):
    """Wall time per map, with figures: the original MATLAB code under Octave against fastdecov V3 (benchmark in results/,
    before the 2.5 local pass)."""
    v3 = METRICS["v3"]
    rows = [("Original MATLAB code\n(Octave, parallel processes)", "neutral",
             (METRICS["octave_p1"]["total_s"], METRICS["octave_p2"]["total_s"])),
            ("fastdecov V3 reference\n(Python + C++, 14 threads)", "s1", (v3["reference"]["P1"]["total_s"], v3["reference"]["P2"]["total_s"])),
            ("fastest port\n(Go, 14 threads)", "s3", (v3["go"]["P1"]["total_s"], v3["go"]["P2"]["total_s"]))]
    fig, axes = figure(t, 9, 3.4, ncols=2, sharey=True)
    for j, (ax, title) in enumerate(zip(axes, ("Pillar 1: 315 spectra", "Pillar 2: 121 spectra"))):
        style(ax, t)
        base = rows[0][2][j]
        for i, (label, c, vals) in enumerate(rows):
            y = len(rows) - 1 - i
            ax.barh(y, vals[j], height=0.56, color=t[c], edgecolor=t["surface"], linewidth=2)
            txt = f"{vals[j]:.1f} s" if i == 0 else f"{vals[j]:.1f} s  ({base / vals[j]:.0f}x faster)"
            ax.text(vals[j] * 1.12, y, txt, va="center", ha="left", fontsize=9, color=t["ink"])
        ax.set_xscale("log"); ax.set_xlim(0.5, 3000)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9, color=t["ink"])
        ax.set_title(title, loc="left", fontsize=10, color=t["ink"], pad=8)
        ax.set_xlabel("wall time per map, including fit figures (s, log scale)", fontsize=8.5, color=t["ink2"])
    fig.tight_layout(w_pad=2)
    save(fig, "speed", theme)


def validation(t, theme):
    """Held-out synthetic maps (12, known truth): emitter recovery of V2 (best-fit basis) and V3 2.5 (physical emitters)."""
    syn = json.loads((HERE / "synthetic_validation.json").read_text())
    v2, v3 = syn["heldout_v2"], syn["v3_2.5"]
    fig, axes = figure(t, 9, 3.0, ncols=2, gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axes[0]; style(ax, t)
    keys = [("precision", "precision"), ("recall", "recall"), ("f1", "F1"), ("nu_accuracy", "line-shape (ν) accuracy")]
    h = 0.36
    for i, (k, lab) in enumerate(keys):
        y = len(keys) - 1 - i
        for off, (name, d, c) in zip((h / 2, -h / 2), (("V2", v2, "s2"), ("V3", v3, "s1"))):
            ax.barh(y + off, d[k], height=h, color=t[c], edgecolor=t["surface"], linewidth=2)
            ax.text(d[k] + 0.012, y + off, f"{name} {d[k]:.2f}", va="center", fontsize=8.5, color=t["ink"])
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([lab for _, lab in keys][::-1], fontsize=9, color=t["ink"])
    ax.set_xlim(0, 1.12); ax.set_xlabel("score (1 = perfect)", fontsize=8.5, color=t["ink2"])
    ax.set_title("Emitter recovery", loc="left", fontsize=10, color=t["ink"], pad=8)
    ax = axes[1]; style(ax, t)
    for y, (name, d, c) in enumerate((("V3", v3, "s1"), ("V2", v2, "s2"))):
        ax.barh(y, d["splits"], height=0.56, color=t[c], edgecolor=t["surface"], linewidth=2)
        ax.text(d["splits"] + 0.05, y, f"{d['splits']:.2f}", va="center", fontsize=9, color=t["ink"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["V3", "V2"], fontsize=9, color=t["ink"]); ax.set_xlim(0, 2.6)
    ax.set_xlabel("emitters split into several lines,\nper map (0 = never)", fontsize=8.5, color=t["ink2"])
    ax.set_title("Emitter splitting", loc="left", fontsize=10, color=t["ink"], pad=8)
    fig.tight_layout(w_pad=2)
    save(fig, "validation", theme)



def main():
    for theme, t in THEMES.items():
        speed(t, theme); validation(t, theme)
    print("wrote", sorted(p.name for p in HERE.glob("*.png")))


if __name__ == "__main__":
    main()
