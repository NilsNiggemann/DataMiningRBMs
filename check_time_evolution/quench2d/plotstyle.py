"""Shared matplotlib style (python-plotting skill preamble)."""
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter

STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10"],             # LaTeX look without a TeX install
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no unicode minus
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 10.5,               # axis labels larger than tick labels (single-column figures, 3.4 in wide)
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "xtick.direction": "in", "ytick.direction": "in",  # ticks point into the axes, also on the top/right edges
    "xtick.top": True, "ytick.right": True,
}
# black, red, blue, then the remaining Wong colors (Makie.wong_colors()[4:end])
COLORS = ["black", "red", "blue", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]

SIMPLE = FuncFormatter(lambda v, _: f"${v:g}$")  # 0, 0.5, 24 — not 0.0, 10^1


def simple_ticks(ax, x=None, y=None):
    """Plain tick labels on both axes of `ax`; `x`/`y` optionally fix the tick positions."""
    for axis, ticks in ((ax.xaxis, x), (ax.yaxis, y)):
        if ticks is not None:
            axis.set_ticks(ticks)
            axis.set_minor_formatter(NullFormatter())  # log axes label minor ticks otherwise
        axis.set_major_formatter(SIMPLE)


def _pi_label(v, _):
    f = Fraction(v / np.pi).limit_denominator(12)
    if f == 0:
        return "$0$"
    num = {1: "", -1: "-"}.get(f.numerator, str(f.numerator))
    return rf"${num}\pi$" if f.denominator == 1 else rf"${num}\pi/{f.denominator}$"


def pi_ticks(axis, step=1):
    """Ticks at multiples of `step`·π on `axis` (e.g. `ax.xaxis`)."""
    axis.set_major_locator(MultipleLocator(step * np.pi))
    axis.set_major_formatter(FuncFormatter(_pi_label))


panel = lambda i: rf"$\mathrm{{({'abcdefghijklmnopqrstuvwxyz'[i]})}}$"


# white text legible on any colormap: stacked faint strokes fake a soft, semi-transparent halo (no blur in matplotlib)
RIM = [patheffects.Stroke(linewidth=w, foreground="black", alpha=0.25) for w in (3.2, 2.4, 1.6)] + [patheffects.Normal()]


def top_cbar(fig, mesh, ax, label, ticks=None):
    """Horizontal colorbar above `ax` (saves width); `ticks` optionally fixes the tick positions."""
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", location="top", pad=0.03, aspect=14, label=label)
    simple_ticks(cbar.ax, x=ticks)
    return cbar


def panel_label(ax, i, on_image=True):
    """(a), (b), ... inside the axes (top left); needed when a colorbar sits above and the title slot is taken."""
    kw = dict(color="white", path_effects=RIM) if on_image else dict(color="black")
    ax.text(0.04, 0.96, panel(i), transform=ax.transAxes, ha="left", va="top", zorder=7, **kw)


def model_label(ax, text, lat=None):
    """Model description (upper right) and lattice sketch (lower right) drawn into a heatmap, white with a black rim."""
    if text:
        ax.text(0.97, 0.96, text, transform=ax.transAxes, ha="right", va="top", color="white", fontsize=7,
                linespacing=1.3, path_effects=RIM, zorder=6)
    if lat is None:
        return
    ins = ax.inset_axes([0.66, 0.04, 0.31, 0.36], zorder=6)
    ins.set_axis_off()
    pos = lat["pos"]
    bond = min(np.linalg.norm(pos[i] - pos[j]) for i, j in lat["nn"])
    for i, j in lat["nn"]:
        if np.linalg.norm(pos[i] - pos[j]) < 1.01 * bond:
            segs = [(pos[i], pos[j])]
        else:  # bond across the periodic boundary: two stubs pointing out of the cluster
            u = 0.4 * bond * lat["D"][i, j] / np.linalg.norm(lat["D"][i, j])
            segs = [(pos[i], pos[i] + u), (pos[j], pos[j] - u)]
        for a, b in segs:
            ins.plot(*np.transpose([a, b]), color="black", lw=2.0, solid_capstyle="round")
            ins.plot(*np.transpose([a, b]), color="white", lw=0.8, solid_capstyle="round")
    ins.scatter(*pos.T, s=7, color="white", edgecolor="black", linewidth=0.6, zorder=3)
    ins.set_aspect("equal")
