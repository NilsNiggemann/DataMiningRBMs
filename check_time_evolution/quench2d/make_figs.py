"""Figures for the 4x3 quench-family study. Run analysis.py first; writes figs/*.pdf (+ .png previews)."""
import os, sys
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter
from plotstyle import model_label
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rydberg"))  # rydberg2d.py
from rydberg2d import lattice

STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10"],             # LaTeX look without a TeX install
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,  # cmr10 has no unicode minus
    "font.size": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
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


def save(fig, name):
    fig.savefig(f"figs/{name}.pdf")
    fig.savefig(f"figs/{name}.png", dpi=110)
    plt.close(fig)


os.makedirs("figs", exist_ok=True)

# --- data ------------------------------------------------------------------------
long = pd.read_csv("data/table_long.csv")
long["log_inf"] = np.log10(long["inf_a1"].clip(lower=1e-12))
long["log_w"] = np.log10(long["w_hi"].clip(lower=1e-12))
tstar = pd.read_csv("data/table_tstar.csv")
plane = pd.read_csv("data/table_tstar_plane.csv")
cal = np.load("data/ed_cal1d.npz")

T_MAX = 1.5
T_NEVER = 1.9  # where 'never' points are drawn in t* plots, just above the time window
FAM_LABEL = {"A": r"$h_f$", "B": r"$J_{xx}$"}
FAM_NAME = {"A": "field quench", "B": "XY-coupling quench"}  # descriptive names of families A, B
FAM_TITLE = {"A": FAM_NAME["A"] + r" ($J_{xx}=0$)", "B": FAM_NAME["B"] + r" ($h_f = 1$)"}
PSI0 = r"from gs$(-\sum ZZ-4\sum X)$" "\n" r"pair terms on nn bonds"
FAM_MODEL = {"A": r"$H_f=-\sum Z_iZ_j-h_f\sum X$" "\n" + PSI0,
             "B": r"$H_f=\sum[-ZZ+J_{xx}(XX+YY)]-\sum X$" "\n" + PSI0,
             "plane": r"$H_f=\sum[-ZZ+J_{xx}(XX+YY)]-h_f\sum X$" "\n" + PSI0}
CYL = lattice("sq4x3")  # same 4x3 cylinder and site order (i = x L_y + y) as quench2d.py
panel = lambda i: rf"$\mathrm{{({'abcdefgh'[i]})}}$"


def grid(df, col):
    """(p values, t values, 2D array p x t) of column `col` for one family."""
    g = df.pivot(index="p", columns="t", values=col)
    return g.index.values, g.columns.values, g.values


# %% 1D calibration: mu, near-nodes and entanglement for the L = 12 TFI chain
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), sharex=True, layout="constrained")
    quantities = [("mu", r"$\mu$"), ("minlog", r"$\min_s \log(2^N|\psi_s|^2)$"), ("S", r"$S_{L/2}$")]
    for i, (ax, (key, label)) in enumerate(zip(axs, quantities)):
        for g_f, y, color in zip(cal["vals"], cal[key], COLORS):
            ax.plot(cal["times"], y, color=color, lw=1.2, label=rf"$h_f = {g_f:g}$")
        ax.set(xlabel=r"$t$", ylabel=label, xlim=(0, T_MAX))
        ax.set_title(panel(i), loc="left")
        simple_ticks(ax)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside right center")
    save(fig, "cal1d")

# %% one row per family: mu | mu_ph | S | log10 I of the alpha = 1 RBM over (t, p); onset lines on every panel
quantities = [("mu", r"$\mu$", "viridis", None), ("mu_ph", r"$\mu_\mathrm{ph}$", "viridis", None),
              ("S", r"$S$ [nats]", "cividis", None), ("log_inf", r"$\log_{10}I$", "inferno", (-6, 0))]
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 4, figsize=(11, 4.8), sharex=True, sharey="row", layout="constrained")
    for row, fam in enumerate(["B", "A"]):
        ed = long[long.fam == fam]
        ts = tstar[tstar.fam == fam]
        for col, (key, label, cmap, lims) in enumerate(quantities):
            ax = axs[row, col]
            src = ed[ed.inf_a1.notna()] if key == "log_inf" else ed  # RBM runs exist only on part of the grid
            ps, tt, Z = grid(src, key)
            vmin, vmax = lims if lims else (None, None)
            mesh = ax.pcolormesh(tt, ps, Z, cmap=cmap, shading="nearest", vmin=vmin, vmax=vmax, rasterized=True)
            cbar = fig.colorbar(mesh, ax=ax, label=label, pad=0.01)
            simple_ticks(cbar.ax)
            ax.plot(ts.t_mu.where(np.isfinite(ts.t_mu)), ts.p, "w-", lw=1.3, label=r"$t^*_{\mu,\mathrm{rel}}$")
            ax.plot(ts.t_dev.where(np.isfinite(ts.t_dev)), ts.p, "w:", lw=1.3, label=r"$t_\mathrm{dev}$")
            tr = ts[ts.t_inf.notna()]
            ax.plot(tr.t_inf.where(np.isfinite(tr.t_inf)), tr.p, "c--", lw=1.3, label=r"$t^*_\mathrm{inf}$")
            ax.set_xlim(0, T_MAX)
            ax.set_title(panel(4 * row + col), loc="left")
            simple_ticks(ax)
        axs[row, 0].set_ylabel(FAM_NAME[fam] + ": " + FAM_LABEL[fam])
        model_label(axs[row, 0], FAM_MODEL[fam], CYL)
    for ax in axs[-1]:
        ax.set_xlabel(r"$t$")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=3, facecolor="0.85")
    save(fig, "heatmaps")

# %% mu(t) from ED above the RBM infidelity at the same representative p; alpha = 2 dashed where run
cuts = {"B": [-1.5, -0.9, -0.5, 0.0, 0.5, 0.9], "A": [0.2, 1.0, 1.8, 2.4, 3.0, 3.4]}  # all have RBM runs
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.8), sharex=True, layout="constrained")
    for col, fam in enumerate(["B", "A"]):
        for p, color in zip(cuts[fam], COLORS):
            g = long[(long.fam == fam) & np.isclose(long.p, p)]
            lbl = FAM_LABEL[fam][:-1] + rf" = {p:g}$"
            axs[0, col].plot(g.t, g.mu, color=color, lw=1.2, label=lbl)
            axs[1, col].plot(g.t, g.inf_a1, color=color, lw=1.2)
            if g.inf_a2.notna().any():
                axs[1, col].plot(g.t, g.inf_a2, color=color, lw=1.0, ls="--")
        axs[0, col].axhline(long.mu.iloc[0] + 5, color="gray", ls="--", lw=0.8, label=r"$\mu_0 + 5$")
        axs[1, col].plot([], [], "k-", lw=1.2, label=r"RBM $\alpha = 1$")
        axs[1, col].plot([], [], "k--", lw=1.0, label=r"RBM $\alpha = 2$")
        axs[1, col].axhline(1e-3, color="gray", ls=":", lw=0.8, label=r"$I = 10^{-3}$")
        axs[0, col].set_title(panel(col) + " " + FAM_TITLE[fam], loc="left")
        axs[1, col].set_title(panel(2 + col), loc="left")
        axs[0, col].legend(loc="best", ncol=2, fontsize=6.5)
        axs[1, col].legend(loc="lower right", fontsize=6.5)
        axs[1, col].set(xlabel=r"$t$", xlim=(0, T_MAX), yscale="log", ylim=(1e-7, 1.5))
        for ax in axs[:, col]:
            simple_ticks(ax)
        axs[1, col].yaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"$10^{{{np.log10(v):.0f}}}$"))
        axs[1, col].set_yticks([1e-6, 1e-4, 1e-2, 1])
    axs[0, 0].set_ylabel(r"$\mu$ (exact state)")
    axs[1, 0].set_ylabel(r"RBM infidelity $I$")
    save(fig, "mu_inf_cuts")

# %% t*_mu and plateau length t_dev over the (Jxx, h_f) plane
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, layout="constrained")
    for i, (ax, key, label) in enumerate(zip(axs, ["t_mu", "t_dev"], [r"$t^*_{\mu,\mathrm{rel}}$", r"$t_\mathrm{dev}$"])):
        g = plane.pivot(index="hf", columns="p", values=key)
        Z = np.where(np.isfinite(g.values), g.values, np.nan)
        mesh = ax.pcolormesh(g.columns, np.arange(len(g.index)), Z, cmap="viridis", shading="nearest",
                             norm=LogNorm(vmin=0.02, vmax=T_MAX))
        never = ~np.isfinite(g.values)
        jj, ii = np.nonzero(never)
        ax.plot(g.columns.values[ii], jj, "x", color="red", ms=5, label=r"never (within $t\leq 1.5$)")
        cbar = fig.colorbar(mesh, ax=ax, label=label)
        simple_ticks(cbar.ax, y=[0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5])
        ax.set_xlabel(r"$J_{xx}$")
        ax.set_title(panel(i), loc="left")
        simple_ticks(ax)
        ax.set_yticks(np.arange(len(g.index)), [f"${h:g}$" for h in g.index])  # categorical h_f rows
    axs[0].set_ylabel(r"$h_f$")
    model_label(axs[1], FAM_MODEL["plane"], CYL)  # panel (b) is nearly uniform
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center")
    save(fig, "plane_tstar")

# %% robustness: h_0 = 6 initial state and dt / 2 for the phase unwrapping
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4), layout="constrained")
    for i, fam in enumerate(["B", "A"]):
        ax = axs[i]
        d4, d6 = np.load(f"data/ed_{fam}.npz"), np.load(f"data/ed_{fam}_hi6.npz")
        for p, mu6, color in zip(d6["vals"], d6["mu"], COLORS):
            j = np.argmin(abs(d4["vals"] - p))
            ax.plot(d4["times"], d4["mu"][j], color=color, lw=1.2, label=FAM_LABEL[fam][:-1] + rf" = {p:g}$")
            ax.plot(d6["times"], mu6, color=color, lw=1.0, ls="--")
        ax.plot([], [], "k-", lw=1.2, label=r"$h_0 = 4$")
        ax.plot([], [], "k--", lw=1.0, label=r"$h_0 = 6$")
        ax.set(xlabel=r"$t$", ylabel=r"$\mu$", xlim=(0, T_MAX))
        ax.set_title(panel(i) + " " + FAM_NAME[fam], loc="left")
        ax.legend(loc="lower right", fontsize=6.5)
        simple_ticks(ax)
    ax = axs[2]
    d, dh = np.load("data/ed_B.npz"), np.load("data/ed_B_dthalf.npz")
    for p, muh, color in zip(dh["vals"], dh["mu_ph"], COLORS):
        j = np.argmin(abs(d["vals"] - p))
        ax.plot(d["times"], d["mu_ph"][j], color=color, lw=1.2, label=rf"$J_{{xx}} = {p:g}$")
        ax.plot(dh["times"], muh, color=color, lw=1.0, ls="--")
    ax.plot([], [], "k-", lw=1.2, label=r"$dt = 0.01$")
    ax.plot([], [], "k--", lw=1.0, label=r"$dt = 0.005$")
    ax.set(xlabel=r"$t$", ylabel=r"$\mu_\mathrm{ph}$", xlim=(0, T_MAX))
    ax.set_title(panel(2) + " " + FAM_NAME["B"], loc="left")
    ax.legend(loc="lower right", fontsize=6.5)
    simple_ticks(ax)
    save(fig, "robustness")

# %% onset times t*_mu, t*_w and t*_inf against the control parameter
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, layout="constrained")
    for i, (ax, fam) in enumerate(zip(axs, ["B", "A"])):
        ts = tstar[tstar.fam == fam]
        rb = ts[ts.t_inf.notna()]
        never = lambda s: s.replace(np.inf, T_NEVER)
        ax.plot(ts.p, never(ts.t_mu), "o-", color="black", ms=3, lw=1, label=r"$t^*_{\mu,\mathrm{rel}}$ ($\mu > \mu_0 + 5$)")
        ax.plot(ts.p, never(ts.t_w), "s-", color="blue", ms=3, lw=1, label=r"$t^*_w$ ($w_\mathrm{hi} > 1\%$)")
        ax.plot(rb.p, never(rb.t_inf), "D--", color="red", ms=3.5, lw=1,
                label=r"$t^*_\mathrm{inf}$ ($I > 10^{-3}$)")
        ax.axhspan(1.65, 2.2, color="0.88", zorder=0, label=r"never within $t \leq 1.5$")
        ax.set(xlabel=FAM_LABEL[fam], yscale="log", ylim=(0.015, 2.2))
        ax.set_title(panel(i) + " " + FAM_TITLE[fam], loc="left")
        simple_ticks(ax, y=[0.02, 0.05, 0.1, 0.2, 0.5, 1, 1.5])
    axs[0].set_ylabel(r"onset time")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    save(fig, "tstar_overlay")

# %% pointwise relation: mu against infidelity coloured by S, and S against infidelity coloured by mu
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.2), sharey=True, layout="constrained")
    for row, fam in enumerate(["B", "A"]):
        rb = long[(long.fam == fam) & long.inf_a1.notna() & (long.t > 0)]
        for col, (x, c, xl, cl, cmap) in enumerate([("mu", "S", r"$\mu$", r"$S_\mathrm{half}$", "cividis"),
                                                    ("S", "mu", r"$S_\mathrm{half}$", r"$\mu$", "viridis")]):
            ax = axs[row, col]
            sc = ax.scatter(rb[x], rb.log_inf, c=rb[c], s=3, cmap=cmap, rasterized=True)
            cbar = fig.colorbar(sc, ax=ax, label=cl)
            simple_ticks(cbar.ax)
            ax.axhline(-3, color="gray", ls="--", lw=0.8, label=r"$I = 10^{-3}$")
            ax.set_xlabel(xl)
            ax.set_title(panel(2 * row + col) + " " + FAM_TITLE[fam], loc="left")
            simple_ticks(ax)
        axs[row, 0].set_ylabel(r"$\log_{10}I$")
    axs[0, 0].legend(loc="upper left")
    save(fig, "scatter_mu_inf")
