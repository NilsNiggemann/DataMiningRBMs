"""Appendix figures on the tracked fidelity fit (g_f=0.5, alpha=1, K=120), from tracked_demo.py output."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "quench2d"))
from plotstyle import COLORS, STYLE, plt, simple_ticks  # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation  # noqa: E402
import phase_branch as pb  # noqa: E402

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- data ------------------------------------------------------------------------
G_F, ALPHA, T_MECH = 0.5, 1.0, 0.8
K = pb.n_var(ALPHA)
demo = {v: np.load(f"data/tracked_demo_{v}.npz") for v in ["full", "warm", "warm_swaps", "fresh", "single", "refit"]}
t = demo["full"]["t"]
k_mech = int(np.argmin(np.abs(t - T_MECH)))
rbm = pd.read_csv("data/rbm.csv")
rbm = rbm[(rbm.g_f == G_F) & (rbm.alpha == ALPHA)].groupby("t")["inf"].median()

psi_t, phase = pb.evolve_unwrapped(G_F)
psi_t, phase = psi_t[::pb.T_STRIDE], phase[::pb.T_STRIDE]
psi = psi_t[k_mech] / np.linalg.norm(psi_t[k_mech])
weight = np.abs(psi) ** 2 * pb.DIM  # 1 = uniform weight


def dense(S, c):
    """Coefficient vector over all 2^L masks (zero off the support)."""
    out = np.zeros(pb.DIM, complex)
    out[S] = c
    return out


def model(c_dense):
    lp = pb.W @ c_dense
    phi = np.exp(lp - lp.real.max())
    return phi / np.linalg.norm(phi)


def aligned_error(phi):
    """Per-configuration |psi - e^{i theta} phi|^2 with the optimal global phase; sums to 2(1 - sqrt F)."""
    ov = np.vdot(phi, psi)
    return np.abs(psi - ov / abs(ov) * phi) ** 2


def mask(sites):
    return sum(1 << (pb.L - 1 - i) for i in sites)  # site 0 is the most significant bit


# models at T_MECH: tracked fit, single-time fit, largest-K truncation of the unwrapped log
c_tracked = dense(demo["full"]["S"][k_mech], demo["full"]["c"][k_mech])
c_single = dense(demo["single"]["S"][k_mech], demo["single"]["c"][k_mech])
_, lp_unwrap = pb.wht_truncation(np.log(np.abs(psi)) + 1j * phase[k_mech], K)
phi_unwrap = np.exp(lp_unwrap - lp_unwrap.real.max())
phi_unwrap /= np.linalg.norm(phi_unwrap)
err = {"tracked": aligned_error(model(c_tracked)), "unwrap": aligned_error(phi_unwrap)}

# support vs starting point: refit at T_MECH on each support from several starts
S_tr, S_st = demo["full"]["S"][k_mech], demo["single"]["S"][k_mech]
sw = np.sqrt(np.abs(psi) ** 2)
lp_un = np.log(np.abs(psi)) + 1j * phase[k_mech]
c_ls, *_ = np.linalg.lstsq(sw[:, None] * pb.W[:, S_tr], sw * lp_un, rcond=None)  # log-space fit on S_tr
c_prev = dense(demo["full"]["S"][k_mech - 1], demo["full"]["c"][k_mech - 1])
starts = [(S_tr, "tracked", "previous step", c_prev[S_tr]),
          (S_tr, "tracked", "zero", np.zeros(len(S_tr), complex)),
          (S_tr, "tracked", r"weighted fit of $\log\psi$", c_ls),
          (S_tr, "tracked", "single-time optimum", c_single[S_tr]),
          (S_st, "single-time", "single-time optimum", c_single[S_st]),
          (S_st, "single-time", "tracked optimum", c_tracked[S_st])]
refits = pd.DataFrame([dict(support=sup, start=start, inf=pb.direct_fit(psi, S, c0)[1]) for S, sup, start, c0 in starts])

# support statistics of the tracked fit
S_full = demo["full"]["S"]
order_counts = pd.DataFrame([np.bincount(pb.ORDER[S[1:]], minlength=pb.L + 1) for S in S_full])
turnover = [0] + [len(np.setdiff1d(S_full[i], S_full[i - 1])) for i in range(1, len(t))]
swap_gain = demo["refit"]["inf"] / demo["full"]["inf"]

# --- what to plot ----------------------------------------------------------------
# method colours as in make_figs.py (METHOD_COLORS); warm_swaps is not drawn, it coincides with the tracked fit
variants = [("full", "tracked fit", COLORS[1], "-"),
            ("warm", "terms fixed at their $t=0$ choice", COLORS[5], "-"),
            ("single", "single-time fit (unwrapped start)", COLORS[2], "-"),
            ("fresh", "single-time fit (wrapped start)", "gray", "--")]
masks = [((4, 5), r"$s_4s_5$"), ((4, 6), r"$s_4s_6$"), ((3, 4, 5, 6), r"$s_3s_4s_5s_6$")]
mask_colors = ["black", "#D55E00", "#009E73"]  # categories, not methods
order_colors = {2: "black", 4: "#D55E00", 6: "#009E73", 8: "gray"}
title = rf"$L=10$, $g_i=1.5\to g_f={G_F:g}$, $K={K}$ ($\alpha={ALPHA:g}$)"

# %% Fig. D1: ablation, support turnover and the gain of the swaps
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), sharex=True, layout="constrained")
    ax = axs[0]
    ax.plot(rbm.index, rbm.values, color=COLORS[0], lw=1.5, label="RBM (TDVP)")
    for v, name, color, ls in variants:
        ax.plot(t, demo[v]["inf"], color=color, ls=ls, lw=1, label=name)
    ax.set(yscale="log", ylabel=r"infidelity $1-F$", ylim=(1e-10, 1))
    ax.set_title(r"$\mathrm{(a)}$", loc="left")
    axs[1].plot(t, turnover, "o-", ms=2, color=COLORS[0], lw=0.8)
    axs[1].set(ylabel=r"terms exchanged per step")
    axs[1].set_title(r"$\mathrm{(b)}$", loc="left")
    axs[2].plot(t, swap_gain, "o-", ms=2, color=COLORS[0], lw=0.8)
    axs[2].axhline(1, color="gray", ls=":", lw=0.8, label="no gain")
    axs[2].set(ylabel=r"$(1-F)_\mathrm{refit}/(1-F)_\mathrm{swaps}$")
    axs[2].set_title(r"$\mathrm{(c)}$", loc="left")
    axs[2].legend(loc="upper right")
    for ax in axs:
        ax.set_xlabel(r"$t$")
        simple_ticks(ax, x=[0, 0.5, 1])
    axs[0].yaxis.set_major_formatter(LogFormatterSciNotation())
    fig.suptitle(title, fontsize=8)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.savefig("figs/tracked_ablation.pdf")
    fig.savefig("figs/tracked_ablation.png", dpi=200)

# %% Fig. D2: support composition and coefficient trajectories
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), sharex=True, layout="constrained")
    ax = axs[0]
    for n, color in order_colors.items():
        ax.plot(t, order_counts[n], color=color, label=rf"$|A|={n}$")
    ax.plot(t, order_counts[list(range(1, pb.L + 1, 2))].sum(1), color="gray", ls="--", label=r"$|A|$ odd")
    ax.set(ylabel=r"number of kept terms", ylim=(-3, 118))
    ax.set_title(r"$\mathrm{(a)}$ kept terms, tracked fit", loc="left")
    ax.legend(loc="upper center", ncol=3, fontsize=6.5, columnspacing=0.8, handlelength=1.2)
    for part, ax, lab in [(np.imag, axs[1], r"$\mathrm{(b)}$"), (np.real, axs[2], r"$\mathrm{(c)}$")]:
        for (sites, name), color in zip(masks, mask_colors):
            A = mask(sites)
            tr = [part(dense(S, c)[A]) for S, c in zip(demo["full"]["S"], demo["full"]["c"])]
            si = [part(dense(S, c)[A]) for S, c in zip(demo["single"]["S"], demo["single"]["c"])]
            ax.plot(t, tr, color=color, lw=1.2, label=name + " tracked fit")
            ax.plot(t, si, "o", ms=2, color=color, mfc="none", mew=0.6, label=name + " single-time fit")
        ax.set_title(lab, loc="left")
    axs[1].set(ylabel=r"$\mathrm{Im}\,c_A$")
    axs[2].set(ylabel=r"$\mathrm{Re}\,c_A$")
    for ax in axs:
        ax.set_xlabel(r"$t$")
        simple_ticks(ax, x=[0, 0.5, 1])
    fig.suptitle(title, fontsize=8)
    fig.legend(*axs[1].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.savefig("figs/tracked_coeffs.pdf")
    fig.savefig("figs/tracked_coeffs.png", dpi=200)

# %% Fig. D3: where the error sits at T_MECH, and support vs starting point
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), layout="constrained")
    names = {"tracked": ("tracked fit", COLORS[1]), "unwrap": ("largest-$K$ truncation, unwrapped", COLORS[3])}
    order = np.argsort(weight)
    for key, (name, color) in names.items():
        axs[0].scatter(weight, err[key] * pb.DIM, s=2, color=color, label=name)
        axs[1].plot(weight[order], np.cumsum(err[key][order]) / err[key].sum(), color=color)
    axs[0].set(xscale="log", yscale="log", xlabel=r"$2^L|\psi(s)|^2$", ylabel=r"$2^L|\psi(s)-e^{i\theta}\phi(s)|^2$")
    axs[0].set_title(r"$\mathrm{(a)}$", loc="left")
    axs[1].axvline(1e-2, color="gray", ls=":", lw=0.8, label=r"near-zero threshold")
    axs[1].set(xscale="log", xlabel=r"$2^L|\psi(s)|^2$", ylabel=r"fraction of error below", ylim=(0, 1.02))
    axs[1].set_title(r"$\mathrm{(b)}$", loc="left")
    axs[1].legend(loc="upper left", fontsize=6.5)
    # one bar per (support, start); colour = support
    support_colors = {"tracked": COLORS[1], "single-time": COLORS[2]}
    y = np.arange(len(refits))[::-1]
    axs[2].barh(y, refits["inf"], color=[support_colors[s] for s in refits.support], height=0.6, left=1e-5)
    axs[2].set_yticks(y, [f"from {s}" for s in refits.start], fontsize=6.5)  # categorical axis
    axs[2].set(xscale="log", xlim=(1e-5, 10), xlabel=r"$1-F$ after refit")
    axs[2].set_title(r"$\mathrm{(c)}$", loc="left")
    axs[2].legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in support_colors.values()],
                  [r"terms of tracked fit", r"terms of single-time fit"], loc="upper right", fontsize=6.5)
    simple_ticks(axs[1], y=[0, 0.5, 1])
    for ax in axs:
        ax.xaxis.set_major_formatter(LogFormatterSciNotation())
    axs[0].yaxis.set_major_formatter(LogFormatterSciNotation())
    fig.suptitle(title + rf", $t={T_MECH:g}$", fontsize=8)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, markerscale=3)
    fig.savefig("figs/tracked_mechanism.pdf")
    fig.savefig("figs/tracked_mechanism.png", dpi=200)

# numbers quoted in the report
print("inf at t=0.8:", {v: f"{demo[v]['inf'][k_mech]:.2e}" for v in demo}, f"RBM {rbm.loc[np.isclose(rbm.index, T_MECH)].item():.2e}")
print("winner counts (full):", pd.Series(demo["full"]["winner"]).value_counts().to_dict())
print("full == warm_swaps:", np.allclose(demo["full"]["inf"], demo["warm_swaps"]["inf"]))
for key in err:
    print(key, "error fraction from near-zeros (2^L|psi|^2<1e-2):", f"{err[key][weight < 1e-2].sum() / err[key].sum():.3f}",
          " n near-zero:", (weight < 1e-2).sum())
print("swap gain median, max:", np.median(swap_gain[1:]), swap_gain.max(), " turnover total:", sum(turnover))
print(refits.to_string())
