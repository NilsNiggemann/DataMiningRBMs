"""Figures for the 4x3 quench-family study. Run analysis.py first; writes figs/*.pdf (+ .png previews)."""
import os, sys
from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from plotstyle import COLORS, STYLE, model_label, panel, panel_label, simple_ticks, top_cbar
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rydberg"))  # rydberg2d.py
from rydberg2d import lattice

W = 3.4  # single-column width in inches (revtex)


def save(fig, name):
    fig.savefig(f"figs/{name}.pdf")
    fig.savefig(f"figs/{name}.png", dpi=110)
    plt.close(fig)


os.makedirs("figs", exist_ok=True)

# --- data ------------------------------------------------------------------------
long = pd.read_csv("data/table_long.csv")
long = long.sort_values(["fam", "p", "t"])
long["mumax"] = long.groupby(["fam", "p"]).mu.cummax()  # running max_{t'<=t} mu
long["log_inf"] = np.log10(long["inf_a1"].clip(lower=1e-12))
long["log_w"] = np.log10(long["w_hi"].clip(lower=1e-12))
tstar = pd.read_csv("data/table_tstar.csv")
plane = pd.read_csv("data/table_tstar_plane.csv")
cal = np.load("data/ed_cal1d.npz")

T_MAX = 1.5
MUMAX = os.environ.get("MUMAX") == "1"  # MUMAX=1: heatmaps/scatter with mu_max instead of mu (saved as *_mumax)
CONT_COLORS = ["white", "red", "black"] if MUMAX else "white"  # per-level colours, contrast with viridis
MU, MU_TEX, SUF = ("mumax", r"$\mu_{\max}$", "_mumax") if MUMAX else ("mu", r"$\mu$", "")
S_HARD = (6 * np.log(2) - 0.5) / 2  # MPS-hard threshold of the plan (half the random-state entropy), N = 12
S_LEVELS, S_STYLES = [1.0, S_HARD, 3.0], ["--", "-", "--"]  # entanglement contours in nats
T_NEVER = 1.9  # where 'never' points are drawn in t* plots, just above the time window
FAM_LABEL = {"A": r"$h_f$", "B": r"$J_{xx}$"}
FAM_NAME = {"A": "field quench", "B": "XY-coupling quench"}  # descriptive names of families A, B
FAM_TITLE = {"A": FAM_NAME["A"] + r" ($J_{xx}=0$)", "B": FAM_NAME["B"] + r" ($h_f = 1$)"}
PSI0 = r"from gs$(-\sum ZZ-4\sum X)$" "\n" r"pair terms on nn bonds"
FAM_MODEL = {"A": r"$H_f=-\sum Z_iZ_j-h_f\sum X$" "\n" + PSI0,
             "B": r"$H_f=\sum[-ZZ+J_{xx}(XX+YY)]-\sum X$" "\n" + PSI0,
             "plane": r"$H_f=\sum[-ZZ+J_{xx}(XX+YY)]-h_f\sum X$" "\n" + PSI0}
CYL = lattice("sq4x3")  # same 4x3 cylinder and site order (i = x L_y + y) as quench2d.py


def grid(df, col):
    """(p values, t values, 2D array p x t) of column `col` for one family."""
    g = df.pivot(index="p", columns="t", values=col)
    return g.index.values, g.columns.values, g.values


# %% 1D calibration: mu, near-nodes and entanglement for the L = 12 TFI chain
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(3, 1, figsize=(W, 5.0), sharex=True, layout="constrained")
    quantities = [("mu", r"$\mu$"), ("minlog", r"$\min_s \log(2^N|\psi_s|^2)$"), ("S", r"$S$ [nats]")]
    for i, (ax, (key, label)) in enumerate(zip(axs, quantities)):
        for g_f, y, color in zip(cal["vals"], cal[key], COLORS):
            ax.plot(cal["times"], y, color=color, lw=1.2, label=rf"$h_f = {g_f:g}$")
        ax.set(ylabel=label, xlim=(0, T_MAX))
        ax.set_title(panel(i), loc="left")
        simple_ticks(ax)
    axs[-1].set_xlabel(r"$t$")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncols=4, fontsize=7)
    save(fig, "cal1d")

# %% one row per family: log10 I of the alpha = 1 RBM | mu | S over (t, p). Contours of S (black) and mu (white) and the
# onset line are drawn on the infidelity panel. Model details are in the paper caption (no room in a single column).
quantities = [("log_inf", r"$\log_{10}I$", "inferno", (-6, 0)), (MU, MU_TEX, "viridis", None),
              ("S", r"$S$ [nats]", "cividis", (0, 6 * np.log(2)))]
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 3, figsize=(W, 3.3), sharex=True, sharey="row", layout="constrained")
    fig.get_layout_engine().set(wspace=0.01, w_pad=0.02)  # tight columns -> wider, squarer panels
    for row, fam in enumerate(["B", "A"]):
        ed = long[long.fam == fam]
        ts = tstar[tstar.fam == fam]
        ps_S, tt_S, S_grid = grid(ed, "S")
        I_levels = [-5, -4, -3]  # log10 I contours drawn on the mu panel
        for col, (key, label, cmap, lims) in enumerate(quantities):
            ax = axs[row, col]
            src = ed[ed.inf_a1.notna()] if key == "log_inf" else ed  # RBM runs exist only on part of the grid
            ps, tt, Z = grid(src, key)
            vmin, vmax = lims if lims else (None, None)
            mesh = ax.pcolormesh(tt, ps, Z, cmap=cmap, shading="nearest", vmin=vmin, vmax=vmax, rasterized=True)
            if row == 0:
                top_cbar(fig, mesh, axs[:, col], label)  # one colorbar per column
            if key == MU:
                pI, tI, ZI = grid(ed[ed.inf_a1.notna()], "log_inf")  # RBM runs cover only part of the (t, p) grid
                ax.contour(tI, pI, np.ma.masked_invalid(ZI), levels=I_levels, colors=CONT_COLORS, linestyles="-", linewidths=0.9)
            ax.set_xlim(0, T_MAX)
            panel_label(ax, 3 * row + col)
            simple_ticks(ax, x=[0, 0.5, 1])
        axs[row, 0].set_ylabel(FAM_LABEL[fam])
        model_label(axs[row, 0], "", CYL)
    for ax in axs[-1]:
        ax.set_xlabel(r"$t$")
    if MUMAX:
        fig.legend([plt.Line2D([], [], color=c, lw=1.2) for c in CONT_COLORS], [rf"$\log_{{10}}I = {l}$" for l in I_levels],
                   loc="outside lower center", ncols=3, fontsize=6.5, columnspacing=0.8, handlelength=1.2, facecolor="0.85")
    else:
        fig.legend([plt.Line2D([], [], color="0.6", lw=0.8)], [r"$\log_{10}I = -5,\ -4,\ -3$"],
                   loc="outside lower center", fontsize=7, facecolor="0.85")
    save(fig, "heatmaps" + SUF)

# %% mu(t) from ED above the RBM infidelity at the same representative p; alpha = 2 dashed where run
cuts = {"B": [-1.5, -0.9, -0.5, 0.0, 0.5, 0.9], "A": [0.2, 1.0, 1.8, 2.4, 3.0, 3.4]}  # all have RBM runs
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.4), sharex=True, layout="constrained")
    for col, fam in enumerate(["B", "A"]):
        for p, color in zip(cuts[fam], COLORS):
            g = long[(long.fam == fam) & np.isclose(long.p, p)]
            lbl = FAM_LABEL[fam][:-1] + rf" = {p:g}$"
            axs[0, col].plot(g.t, g.mu, color=color, lw=1.2, label=lbl)
            axs[1, col].plot(g.t, g.inf_a1, color=color, lw=1.2)
            if g.inf_a2.notna().any():
                axs[1, col].plot(g.t, g.inf_a2, color=color, lw=1.0, ls="--")
        axs[0, col].axhline(long.mu.iloc[0] + 5, color="gray", ls="--", lw=0.8, label=r"$\mu_0 + 5$")
        axs[1, col].plot([], [], "k-", lw=1.2, label=r"$\alpha = 1$")
        axs[1, col].plot([], [], "k--", lw=1.0, label=r"$\alpha = 2$")
        axs[1, col].axhline(1e-3, color="gray", ls=":", lw=0.8, label=r"$I = 10^{-3}$")
        axs[0, col].set_title(panel(col), loc="left")
        axs[1, col].set_title(panel(2 + col), loc="left")
        axs[0, col].set_ylim(-26, -3)
        axs[0, col].legend(loc="lower right", ncol=2, fontsize=5.5, columnspacing=0.8, handlelength=1.2)
        axs[1, col].legend(loc="lower right", fontsize=6)
        axs[1, col].set(xlabel=r"$t$", xlim=(0, T_MAX), yscale="log", ylim=(1e-7, 1.5))
        for ax in axs[:, col]:
            simple_ticks(ax)
        axs[1, col].yaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"$10^{{{np.log10(v):.0f}}}$"))
        axs[1, col].set_yticks([1e-6, 1e-4, 1e-2, 1])
    axs[0, 0].set_ylabel(r"$\mu$")
    axs[1, 0].set_ylabel(r"$I$")
    save(fig, "mu_inf_cuts")

# %% t*_mu and plateau length t_dev over the (Jxx, h_f) plane
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 1, figsize=(W, 4.4), sharex=True, layout="constrained")
    for i, (ax, key, label) in enumerate(zip(axs, ["t_mu", "t_dev"], [r"$t^*_{\mu,\mathrm{rel}}$", r"$t_\mathrm{dev}$"])):
        g = plane.pivot(index="hf", columns="p", values=key)
        Z = np.where(np.isfinite(g.values), g.values, np.nan)
        mesh = ax.pcolormesh(g.columns, np.arange(len(g.index)), Z, cmap="viridis", shading="nearest",
                             norm=LogNorm(vmin=0.02, vmax=T_MAX))
        jj, ii = np.nonzero(~np.isfinite(g.values))
        ax.plot(g.columns.values[ii], jj, "x", color="red", ms=4, label=r"never ($t\leq 1.5$)")
        top_cbar(fig, mesh, ax, label, ticks=[0.02, 0.1, 0.5, 1.5])
        ax.set_ylabel(r"$h_f$")
        panel_label(ax, i)
        simple_ticks(ax)
        ax.set_yticks(np.arange(len(g.index)), [f"${h:g}$" for h in g.index])  # categorical h_f rows
    axs[-1].set_xlabel(r"$J_{xx}$")
    model_label(axs[1], "", CYL)  # panel (b) is nearly uniform
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", fontsize=7)
    save(fig, "plane_tstar")

# %% robustness: h_0 = 6 initial state and dt / 2 for the phase unwrapping
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(3, 1, figsize=(W, 6.0), layout="constrained")
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
        ax.set_title(panel(i), loc="left")
        ax.legend(loc="lower right", fontsize=6, ncol=2)
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
    ax.set_title(panel(2), loc="left")
    ax.legend(loc="lower right", fontsize=6, ncol=2)
    simple_ticks(ax)
    save(fig, "robustness")

# %% onset times t*_mu, t*_w and t*_inf against the control parameter
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 1, figsize=(W, 4.6), layout="constrained")
    for i, (ax, fam) in enumerate(zip(axs, ["B", "A"])):
        ts = tstar[tstar.fam == fam]
        rb = ts[ts.t_inf.notna()]
        never = lambda s: s.replace(np.inf, T_NEVER)
        ax.plot(ts.p, never(ts.t_mu), "o-", color="black", ms=3, lw=1, label=r"$t^*_{\mu,\mathrm{rel}}$")
        ax.plot(ts.p, never(ts.t_w), "s-", color="blue", ms=3, lw=1, label=r"$t^*_w$")
        ax.plot(rb.p, never(rb.t_inf), "D--", color="red", ms=3.5, lw=1, label=r"$t^*_\mathrm{inf}$")
        ax.axhspan(1.65, 2.2, color="0.88", zorder=0, label=r"never")
        ax.set(xlabel=FAM_LABEL[fam], ylabel=r"onset time", yscale="log", ylim=(0.015, 2.2))
        ax.set_title(panel(i), loc="left")
        simple_ticks(ax, y=[0.02, 0.1, 0.5, 1.5])
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=7)
    save(fig, "tstar_overlay")

# %% pointwise relation: mu against infidelity coloured by S, and S against infidelity coloured by mu
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.4), sharey=True, layout="constrained")
    for row, fam in enumerate(["B", "A"]):
        rb = long[(long.fam == fam) & long.inf_a1.notna() & (long.t > 0)]
        for col, (x, c, xl, cl, cmap) in enumerate([(MU, "S", MU_TEX, r"$S$ [nats]", "cividis"),
                                                    ("S", MU, r"$S$ [nats]", MU_TEX, "viridis")]):
            ax = axs[row, col]
            sc = ax.scatter(rb[x], rb.log_inf, c=rb[c], s=2, cmap=cmap, rasterized=True)
            if row == 0:
                top_cbar(fig, sc, axs[:, col], cl)
            ax.axhline(-3, color="gray", ls="--", lw=0.8, label=r"$I = 10^{-3}$")
            ax.set_xlabel(xl)
            panel_label(ax, 2 * row + col, on_image=False)
            simple_ticks(ax)
        axs[row, 0].set_ylabel(r"$\log_{10}I$")
    axs[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 0.86), fontsize=6.5)
    save(fig, "scatter_mu_inf" + SUF)
