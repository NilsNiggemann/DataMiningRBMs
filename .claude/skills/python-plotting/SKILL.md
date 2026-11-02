---
name: python-plotting
description: The user's matplotlib plotting conventions, mirroring their Julia/Makie style — plain tick labels and π ticks, LaTeX (Computer Modern) labels, heatmap colorbars and colormaps, legend placement, multipanel (a)/(b) labels, categorical colors, commented save line. Use this skill whenever writing or editing Python plotting code or making a figure in Python (matplotlib, pyplot, pcolormesh/imshow, plot, scatter, subplots, S(q) plots, phase diagrams, scaling plots), even if the user just says "plot this" or "make a figure".
---

# Python plotting conventions

Python counterpart of the `julia-plotting` skill; keep figures from both languages looking alike.

- **Stack**: `matplotlib.pyplot`, `numpy`, `pandas`. No seaborn/plotly unless asked.
- **Theme and ticks**: paste the preamble below at the top of the script (move it into a shared module once a second script needs it). Wrap each figure in `with plt.rc_context(STYLE):` (the analog of `with_theme`), create it with `layout="constrained"`, and call `simple_ticks(ax)` on every axis. Use `pi_ticks(ax.xaxis)` instead when the axis is an angle or momentum, where multiples of π are the natural ticks.
- **Minor ticks**: always on, for both axes and colorbars (`xtick.minor.visible`/`ytick.minor.visible` in `STYLE`, so no per-axis call is needed). On log axes the default log minor ticks apply; never switch them off.
- **Ticks on heatmaps**: ticks point inward, so they lie on the image. Where the colormap is black or nearly black at the edges (viridis/inferno/magma/cividis low end), black ticks vanish: call `image_ticks(ax)` to draw ticks (major and minor) in white. If the edges are light (e.g. the yellow end of viridis, `bwr` near zero), use `image_ticks(ax, color="black")`; check the rendered figure per panel. Tick *labels* stay black, they sit outside the image.
- **Ticks are never overridden by hand**: no `set_xticklabels`, no `(positions, labels)` with hand-formatted strings, no leaving matplotlib's default `ScalarFormatter` (it gives `0.0, 0.5, 1.0` and offset/`1e3` labels; log axes give `10^1`). For explicit positions use `simple_ticks(ax, x=[12, 18, 24, 36])`. Colorbars too: `simple_ticks(cbar.ax)`. Only categorical axes may set string labels.
- **Colors**: in line and scatter plots the first, most important series is black, the second red, the third blue. For further series use the colorblind-friendly Wong colors, i.e. `COLORS` from the preamble. Grouped categories follow the same order.
- **Labels**: always LaTeX math via raw strings, e.g. `r"$\varphi$"`. The style sets Computer Modern for both text and math, so plain text (`"Text"`) already matches; no `text.usetex` needed.
- **Titles**: skip any title that only repeats the axis labels.
- **Heatmaps**: use `pcolormesh` (or `imshow` on a regular grid) and always add `fig.colorbar(..., label=...)`. Use `cmap="viridis"` by default and `cmap="bwr"` for data symmetric around zero, in which case also set symmetric `vmin=-m, vmax=m`.
- **Colorbars above the axes**: place heatmap colorbars horizontally on top to save width: `fig.colorbar(mesh, ax=axs[:, col], orientation="horizontal", location="top", pad=0.03, aspect=14, label=...)` (one per column when panels share the scale), then `simple_ticks(cbar.ax)`. The title slot is then taken, so put the panel label inside the axes: `ax.text(0.04, 0.96, r"$\mathrm{(a)}$", transform=ax.transAxes, ha="left", va="top", color="white")` with the halo below (black text on white-background scatter plots).
- **Paper figures**: default to single-column width, `figsize=(3.4, h)`, with large axis labels and 7–8 pt legends; stack panels or use at most 3 narrow columns. Keep long descriptions (Hamiltonians etc.) in the caption, not in the axes.
- **Text on heatmaps**: annotations drawn on top of a colormap are white with a soft dark halo, so they read on any color. Matplotlib has no blur, so stack faint strokes from wide to narrow: `path_effects=[patheffects.Stroke(linewidth=w, foreground="black", alpha=0.25) for w in (3.2, 2.4, 1.6)] + [patheffects.Normal()]` (`from matplotlib import patheffects`). Never a single hard opaque `withStroke`. It reads as a heavy outline.
- **Legends**: never `ax.legend()` with the default `loc="best"` (floats wherever there is space). Use an explicit `loc="center right"` etc., or give the legend its own space outside: `fig.legend(*ax.get_legend_handles_labels(), loc="outside right center")`.
- **Categories**: a small fixed set of discrete categories (e.g. a few system sizes) gets one fixed color each from `COLORS`, not slices of a continuous colormap.
- **Multipanel**: label panels with `ax.set_title(r"$\mathrm{(a)}$", loc="left")`. Panels that share an axis use `plt.subplots(..., sharex=/sharey=True)` (it hides the duplicated tick labels); set the shared axis label only on the outer panel.
- **Reference lines**: every dashed/dotted guide line (power laws, critical values, ...) needs a `label` so it shows up in the legend, or an `ax.text` annotation next to it. Never leave an unexplained line.
- **Many files / runs**: load them into one `pandas.DataFrame` (`pd.concat(map(pd.read_csv, files))`, one row per measurement, with columns like `x`, `L`, `T`), then select with `df[df["T"] == T]` / `groupby` instead of parallel arrays, boolean-mask lists, or tuples of strings parsed back into numbers.
- **Readable scripts**: write the plotting script the way a person would revise it later. Load data at the top, then a short block of named parameters (`Ls`, `temps`, `colors`), then the figure. Give intermediate results names, use one loop level per concept (panel, then group), and prefer plain loops over nested comprehensions and index arithmetic like `i // ncols`, `i % ncols`. Do not mix analysis (fits, tables, printing) into the plotting block; do it in a separate step that produces a DataFrame. Short comments say *why*, not what.
- **Cell delimiters**: in a script that makes more than one figure, start each figure block with a `# %%` line, so each figure runs as its own cell in the editor. Single-figure scripts need none.
- **Save**: end with a commented-out save line and a descriptive filename, e.g. `# fig.savefig("flux_L2_sweep_scaling.pdf")`, followed by `plt.show()`.

## Preamble

```python
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter

STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10"],             # LaTeX look without a TeX install
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no unicode minus
    "axes.labelsize": 10.5,               # axis labels larger than tick labels (readable in a 3.4 in column)
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in",  # ticks point into the axes, also on top/right edges
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,  # minor ticks, also on top/right edges
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


def image_ticks(ax, color="white"):
    """Ticks drawn over a heatmap: white where the colormap is dark (viridis, inferno, ...), `color="black"` where it is light."""
    ax.tick_params(which="both", color=color)


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
```

## Example

```python
import glob

import pandas as pd

# --- data: one table with a row per (x, L, T) --------------------------------
files = sorted(glob.glob("results/fss/*.csv"))
df = pd.concat(map(pd.read_csv, files))   # columns: x, L, T, Smax

# --- what to plot --------------------------------------------------------------
Ls = [12, 18, 24, 36]
temps = [0.03, 0.25]
panel_labels = [r"$\mathrm{(a)}$", r"$\mathrm{(b)}$"]

# --- figure ----------------------------------------------------------------------
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, len(temps), figsize=(3.2 * len(temps) + 1.2, 3.2),
                            sharey=True, layout="constrained")

    for ax, T, panel_label in zip(axs, temps, panel_labels):
        ax.set(xlabel=r"$L$", xscale="log", yscale="log")

        # one curve per x, in the categorical color order
        for (x, group), color in zip(df[df["T"] == T].groupby("x"), COLORS):
            group = group.sort_values("L")
            ax.plot(group["L"], group["Smax"], "o-", color=color, label=rf"$x = {x:g}$")

        # guide for fully correlated order; the prefactor only sets the vertical offset
        ax.plot(Ls, 0.1 * np.array(Ls) ** 2, "--", color="gray", label=r"$\propto L^2$")

        simple_ticks(ax, x=Ls)
        ax.set_title(panel_label, loc="left")

    axs[0].set_ylabel(r"$S(Q_\mathrm{max})$")
    fig.legend(*axs[-1].get_legend_handles_labels(), loc="outside right center")

    # fig.savefig("fss_Smax_vs_L.pdf")
    plt.show()
```

A heatmap panel pair (shared y axis, one colorbar):

```python
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True, layout="constrained")

    for ax, sq, panel_label in zip(axs, (sq_a, sq_b), (r"$\mathrm{(a)}$", r"$\mathrm{(b)}$")):
        mesh = ax.pcolormesh(kgrid, kgrid, sq, cmap="viridis", vmin=sq_min, vmax=sq_max)
        ax.set(xlabel=r"$q_x$", aspect=1)
        pi_ticks(ax.xaxis)
        pi_ticks(ax.yaxis)
        ax.set_title(panel_label, loc="left")
    axs[0].set_ylabel(r"$q_y$")

    cbar = fig.colorbar(mesh, ax=axs, label=r"$S(\mathbf{q})$")
    simple_ticks(cbar.ax)

    # fig.savefig("sq_comparison.pdf")
    plt.show()
```
