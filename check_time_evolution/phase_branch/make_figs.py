"""Predictor-vs-RBM metrics (data/metrics.csv) and figures (figs/) from phase_branch.py output."""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "quench2d"))
from plotstyle import COLORS, SIMPLE, STYLE, plt, simple_ticks  # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation  # noqa: E402
import phase_branch as pb  # noqa: E402

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("figs", exist_ok=True)

FLOOR_FACTOR = 10  # keep points where the RBM infidelity exceeds 10x its t=0 value
MODEL_SHORT = r"$H=-\Sigma_i Z_iZ_{i+1}-g_f\Sigma_i X_i$, $L=10$, $g_i=1.5\to g_f=0.5$"
MODEL = (r"$H=-\Sigma_i Z_iZ_{i+1}-g_f\Sigma_i X_i$ on an $L=10$ open chain, "
         r"$|\psi(0)\rangle$ = ground state at $g_i=1.5$")

# --- data ------------------------------------------------------------------------
pred = pd.read_csv("data/predictors.csv")
rbm = pd.read_csv("data/rbm.csv")
for df in (pred, rbm):
    df["t"] = df["t"].round(10)
rbm_stats = rbm.groupby(["g_f", "alpha", "t"])["inf"].agg(rbm="median", rbm_lo="min", rbm_hi="max").reset_index()
both = pred.merge(rbm_stats, on=["g_f", "alpha", "t"])
rbm0 = rbm_stats[rbm_stats.t == 0].set_index(["g_f", "alpha"])["rbm"].rename("rbm0")
both = both.join(rbm0, on=["g_f", "alpha"])
both = both[both.t > 0]
# per-run floor: the early plateau is the initial fit / integrator, not the dynamics
both["above"] = both.rbm > FLOOR_FACTOR * both.rbm0

METHODS = ["wrapped", "unwrap", "branch", "weighted", "direct", "tracked"]
# one fixed colour per model, in every figure
METHOD_COLORS = dict(rbm=COLORS[0], tracked=COLORS[1], direct=COLORS[2], unwrap=COLORS[3], wrapped=COLORS[4],
                     branch=COLORS[5], weighted=COLORS[6])
NAMES = {"wrapped": "wrapped phase", "unwrap": "unwrapped phase", "branch": "branch-optimized",
         "weighted": r"$|\psi|^2$-weighted fit", "direct": "single-time fidelity fit", "tracked": "tracked fidelity fit"}


# --- metrics: one row per (method, subset), points above the per-run floor --------
def metrics(g):
    lp, lr = np.log10(g["inf"].clip(lower=1e-14)), np.log10(g["rbm"])
    d = lp - lr
    # rank correlation across runs (g_f, alpha) at fixed t, which removes the common growth in t
    fixed_t = [spearmanr(x["inf"], x["rbm"])[0] for _, x in g.groupby("t") if len(x) >= 8]
    return pd.Series(dict(n=len(g), spearman=spearmanr(lp, lr)[0], spearman_fixed_t=np.mean(fixed_t) if fixed_t else np.nan,
                          median_log_ratio=d.median(), spread_log_ratio=d.std()))


rows = []
above = both[both.above]
subsets = [("all", above)] + [(f"g_f={g_f:g}", g) for g_f, g in above.groupby("g_f")]
for label, sub in subsets:
    for m in METHODS:
        rows.append(dict(subset=label, method=m, **metrics(sub[sub.method == m])))
table = pd.DataFrame(rows)
table.to_csv("data/metrics.csv", index=False)
pd.set_option("display.width", 200)
print(table.round(3).to_string())

# %% Fig. 1: infidelity vs t at g_f = 0.5, one panel per alpha
shown = ["tracked", "direct", "unwrap", "wrapped"]
colors = METHOD_COLORS
alphas = [0.5, 1.0, 2.0, 3.0]
labels = [r"$\mathrm{(a)}$", r"$\mathrm{(b)}$", r"$\mathrm{(c)}$", r"$\mathrm{(d)}$"]
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True, layout="constrained")
    for ax, alpha, lab in zip(axs, alphas, labels):
        r = rbm_stats[(rbm_stats.g_f == 0.5) & (rbm_stats.alpha == alpha)]
        ax.fill_between(r.t, r.rbm_lo, r.rbm_hi, color="gray", alpha=0.3, lw=0)
        ax.plot(r.t, r.rbm, color=colors["rbm"], lw=1.5, label="RBM (TDVP)")
        for m in shown:
            p = pred[(pred.g_f == 0.5) & (pred.alpha == alpha) & (pred.method == m)].sort_values("t")
            ax.plot(p.t, p["inf"].clip(lower=1e-13), color=colors[m], lw=1, label=NAMES[m])
        ax.set(xlabel=r"$t$", yscale="log", ylim=(1e-12, 1.5))
        ax.set_title(lab + rf"$\ \alpha={alpha:g}$", loc="left")
        simple_ticks(ax, x=[0, 0.5, 1])
        ax.yaxis.set_major_formatter(LogFormatterSciNotation())  # 10^{-6}, not 1e-06
    axs[0].set_ylabel(r"infidelity $1-F$")
    fig.suptitle(MODEL + r", quench to $g_f=0.5$", fontsize=8)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=5)
    fig.savefig("figs/inf_vs_t_gf0.5.pdf")
    fig.savefig("figs/inf_vs_t_gf0.5.png", dpi=200)

# %% Fig. 2: predictor vs RBM over all g_f, alpha, t > 0
alpha_colors = dict(zip(alphas, plt.cm.viridis([0.0, 0.33, 0.66, 0.9])))
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharex=True, sharey=True, layout="constrained")
    for ax, m, lab in zip(axs, ["unwrap", "direct", "tracked"], labels):
        g = both[both.method == m]
        ax.scatter(g.rbm[~g.above], g["inf"][~g.above].clip(lower=1e-13), s=3, color="lightgray",
                   label="below floor")
        for alpha in alphas:
            ga = g[g.above & (g.alpha == alpha)]
            ax.scatter(ga.rbm, ga["inf"].clip(lower=1e-13), s=3, color=alpha_colors[alpha], label=rf"$\alpha={alpha:g}$")
        ax.plot([1e-13, 1], [1e-13, 1], "--", color="gray", lw=0.8, label="equal")
        ax.set(xscale="log", yscale="log", xlabel=r"RBM infidelity")
        ax.set_title(lab + r"$\ $" + NAMES[m], loc="left")
    axs[0].set_ylabel(r"cumulant-model infidelity")
    fig.suptitle(MODEL + r", $g_f\in\{0.3,0.5,0.7,1,2,3\}$", fontsize=8)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside right center")
    fig.savefig("figs/scatter_all.pdf")
    fig.savefig("figs/scatter_all.png", dpi=200)

# %% Fig. 3: near-zeros and the fragility of the unwrapped phase, g_f = 0.5
psi_a, ph_a = pb.evolve_unwrapped(0.5)
_, ph_b = pb.evolve_unwrapped(0.5 + 1e-3)
n_small = (np.abs(psi_a) ** 2 * pb.DIM < 1e-2).sum(1)
n_flip = (np.abs(ph_a - ph_b) > np.pi).sum(1)
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(3.6, 2.5), layout="constrained")
    ax.plot(pb.TIMES, n_small, color=COLORS[0], label=r"$|\psi(s)|^2 < 10^{-2}/2^L$")
    ax.plot(pb.TIMES, n_flip, color="gray", ls="--", label=r"$|\varphi_{g_f}(s)-\varphi_{g_f+10^{-3}}(s)| > \pi$")
    ax.set(xlabel=r"$t$", ylabel=r"number of configurations $s$", yscale="symlog", ylim=(0, 2000))
    simple_ticks(ax, x=[0, 0.5, 1])
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax.legend(loc="upper left")
    ax.set_title(MODEL_SHORT, loc="right", fontsize=7)
    fig.savefig("figs/fragility.pdf")
    fig.savefig("figs/fragility.png", dpi=200)
print("near-zero configs at t=0.8:", n_small[80], " flipped windings at t=0.8:", n_flip[80], " first flip at t =", pb.TIMES[np.argmax(n_flip > 0)])

# %% Fig. 4: truncation infidelity vs N_trunc (exact and RBM states), g_f = 0.5
trunc = pd.read_csv("data/truncation.csv")
tracked = pred[(pred.g_f == 0.5) & (pred.method == "tracked")]
t_show = sorted(trunc.t.unique())
rbm_alphas = sorted(trunc.alpha.dropna().unique())  # alpha=3 is added once its TDVP run finishes
rows = [("rbm_natural", "RBM, network branch"), ("rbm_unwrap", "RBM, continuous phase")]
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, len(t_show), figsize=(7.4, 3.9), sharex=True, sharey=True, layout="constrained")
    for i, (method, row_name) in enumerate(rows):
        for j, t in enumerate(t_show):
            ax = axs[i, j]
            ex = trunc[(trunc.method == "exact_unwrap") & (trunc.t == t)]
            ax.plot(ex.N, ex["inf"].clip(lower=1e-14), color="gray", lw=1.3, label="exact state")  # black is the RBM elsewhere
            for alpha in rbm_alphas:
                r = trunc[(trunc.method == method) & (trunc.t == t) & (trunc.alpha == alpha)]
                ax.plot(r.N, r["inf"].clip(lower=1e-14), color=alpha_colors[alpha], lw=1, label=rf"RBM $\alpha={alpha:g}$")
                ax.axvline(pb.n_var(alpha), color=alpha_colors[alpha], ls=":", lw=0.8)
                tr = tracked[(tracked.alpha == alpha) & (np.isclose(tracked.t, t))]
                ax.plot([pb.n_var(alpha)], tr["inf"].clip(lower=1e-14), "*", ms=6, mec="black", mew=0.4,
                        color=METHOD_COLORS["tracked"], label=r"tracked fit at $N_\mathrm{var}$" if alpha == rbm_alphas[0] else None)
            ax.set(xscale="log", yscale="log", ylim=(1e-13, 2), yticks=10.0 ** np.arange(-12, 1, 2))
            ax.yaxis.set_major_formatter(LogFormatterSciNotation())
            if i == 0:
                ax.set_title(rf"$t={t:g}$")
            if i == 1:
                ax.set_xlabel(r"$N_\mathrm{trunc}$")
        axs[i, 0].set_ylabel(row_name + "\n" + r"infidelity $1-F$")
    handles, labels_ = axs[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="gray", ls=":", lw=0.8))
    labels_.append(r"$N_\mathrm{var}(\alpha)$")
    fig.suptitle(MODEL + r", quench to $g_f=0.5$", fontsize=8)
    fig.legend(handles, labels_, loc="outside lower center", ncol=5, fontsize=6.5)
    fig.savefig("figs/truncation_vs_N.pdf")
    fig.savefig("figs/truncation_vs_N.png", dpi=200)

# %% Fig. 5: agreement metrics by model and by quench (replaces the tables in the main text)
# markers in (a), (b) are greys: red/blue/pink are reserved for the models
NEAR_ZERO_FRAC = 1e-2  # |psi(s)|^2 < 1e-2 / 2^L counts as a near-zero
g_fs = sorted(pred.g_f.unique())
near_zeros = pd.Series({g_f: (np.abs(pb.evolve_unwrapped(g_f)[0]) ** 2 * pb.DIM < NEAR_ZERO_FRAC).sum(1).max()
                        for g_f in g_fs})
print("max near-zero configs per g_f:", near_zeros.to_dict())
table["g_f"] = table.subset.str.extract(r"g_f=(.*)")[0].astype(float)
overall = table[table.subset == "all"].set_index("method").loc[METHODS]
at05 = table[table.g_f == 0.5].set_index("method").loc[METHODS]
per_gf = table[table.subset != "all"]
y = np.arange(len(METHODS))
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 3.1), layout="constrained",
                            gridspec_kw=dict(width_ratios=[1, 0.9, 1, 0.9]))
    ax = axs[0]
    ax.plot(overall.spearman, y, "o", color=COLORS[0], ms=4, label=r"$\rho$, all quenches")
    ax.plot(overall.spearman_fixed_t, y, "s", color="gray", ms=3.5, label=r"$\rho_t$, all quenches")
    ax.plot(at05.spearman, y, "^", color="white", mec="black", ms=4, label=r"$\rho$, $g_f=0.5$")
    ax.set_yticks(y, [NAMES[m] for m in METHODS])  # categorical axis
    ax.set(xlabel=r"rank correlation with RBM", xlim=(0.3, 1.02), xticks=[0.4, 0.6, 0.8, 1])
    ax.xaxis.set_major_formatter(SIMPLE)  # y is categorical, so not simple_ticks
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=6.5)
    ax.set_title(labels[0], loc="left")

    ax = axs[1]
    ax.axvline(0, color="gray", ls="--", lw=0.8, label="model $=$ RBM")
    ax.plot(overall.median_log_ratio, y, "o", color=COLORS[0], ms=4, label="all quenches")
    ax.plot(at05.median_log_ratio, y, "^", color="white", mec="black", ms=4, label=r"$g_f=0.5$")
    ax.set_yticks(y, [])
    ax.set(xlabel=r"offset $\Delta$ (decades)", xlim=(-1.3, 3.6), xticks=[-1, 0, 1, 2, 3])
    ax.xaxis.set_major_formatter(SIMPLE)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=6.5)
    ax.set_title(labels[1], loc="left")

    ax = axs[2]
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    for m in ["unwrap", "tracked"]:
        g = per_gf[per_gf.method == m].sort_values("g_f")
        ax.plot(g.g_f, g.median_log_ratio, "o-", color=METHOD_COLORS[m], ms=3.5, label=NAMES[m])
    ax.set(xlabel=r"$g_f$", ylabel=r"offset $\Delta$ (decades)", xscale="log")
    simple_ticks(ax, x=[0.3, 0.5, 1, 2, 3])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=6.5)
    ax.set_title(labels[2], loc="left")

    ax = axs[3]
    ax.bar(np.arange(len(g_fs)), near_zeros.values, color="gray")
    ax.set_xticks(np.arange(len(g_fs)), [f"${g:g}$" for g in g_fs])  # categorical axis
    ax.set(xlabel=r"$g_f$", ylabel=r"near-zero configurations (max.)")
    ax.yaxis.set_major_formatter(SIMPLE)
    ax.set_title(labels[3], loc="left")
    fig.suptitle(MODEL + r", points above the RBM floor", fontsize=8)
    fig.savefig("figs/metrics.pdf")
    fig.savefig("figs/metrics.png", dpi=200)

# %% Fig. 0: graphical abstract, alpha = 1 at g_f = 0.5
alpha = 1.0
with plt.rc_context(STYLE):
    fig, ax = plt.subplots(figsize=(3.6, 2.5), layout="constrained")
    r = rbm_stats[(rbm_stats.g_f == 0.5) & (rbm_stats.alpha == alpha)]
    ax.fill_between(r.t, r.rbm_lo, r.rbm_hi, color="gray", alpha=0.3, lw=0)
    ax.plot(r.t, r.rbm, color=METHOD_COLORS["rbm"], lw=1.5, label="RBM (TDVP)")
    for m in ["tracked", "unwrap"]:
        p = pred[(pred.g_f == 0.5) & (pred.alpha == alpha) & (pred.method == m)].sort_values("t")
        ax.plot(p.t, p["inf"].clip(lower=1e-13), color=METHOD_COLORS[m], lw=1.2, label=NAMES[m])
    ax.set(xlabel=r"$t$", ylabel=r"infidelity $1-F$", yscale="log", ylim=(1e-10, 1.5))
    simple_ticks(ax, x=[0, 0.5, 1])
    ax.yaxis.set_major_formatter(LogFormatterSciNotation())
    ax.legend(loc="lower right")
    ax.set_title(MODEL_SHORT + r", $\alpha=1$", loc="right", fontsize=7)
    fig.savefig("figs/graphical_abstract.pdf")
    fig.savefig("figs/graphical_abstract.png", dpi=200)
