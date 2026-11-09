"""Alternative to Fig. 3 (d-f): same heatmaps with the onset time t*(g_f) overlaid.
Observed t* (white): first t with I > 10 I(0). Predicted t* (red, on the mu_max panel): first t with mu > threshold,
threshold = value minimising mean |t*_pred - t*_obs| over g_f with an observed onset. Writes figs/tfi1d_heatmaps_onset.pdf."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import STYLE, image_ticks, simple_ticks, top_cbar, panel_label

D = "../../data/cumulants_Fermi_dirac_TDVP"
run, ce, ee = (np.load(f"{D}/{f}") for f in ("rbm_tdvp_quench.npz", "ED_CE/rbm_tdvp_quench.npz", "ED_CE/ee_rbm_tdvp_quench.npz"))
t, gf = run["times"], run["g_f_values"]
I, mu, S = run["infidelity_vs_exact"], ce["mu_grid"], ee["ee_grid"]
M = np.maximum.accumulate(mu, axis=1)

def onset(X, thr):  # first time X > thr (per row); nan if never
    hit = X > thr[:, None] if np.ndim(thr) else X > thr
    return np.where(hit.any(1), t[hit.argmax(1)], np.nan)

t_obs = onset(I, 10 * I[:, 0])
ok = ~np.isnan(t_obs)
thrs = np.linspace(M.min(), M.max(), 400)
err = [np.nanmean(np.abs(np.where(np.isnan(p := onset(M, th)), 1.0, p)[ok] - t_obs[ok])) for th in thrs]
thr = thrs[int(np.argmin(err))]
t_mu = onset(M, thr)
print(f"mu threshold {thr:.2f}, mean |dt| = {min(err):.3f}, {ok.sum()}/{len(gf)} quenches with observed onset")

with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 3, figsize=(3.4, 2.3), sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.01, w_pad=0.02)
    for i, (ax, Z, lab, cmap, kw) in enumerate(zip(axs, [np.log10(np.clip(I, 1e-12, None)), M, S],
                                  [r"$\log_{10}I$", r"$\mu_{\max}$", r"$S$ [nats]"], ["inferno", "viridis", "cividis"], [dict(vmin=-6), {}, {}])):
        m = ax.pcolormesh(t, gf, Z, cmap=cmap, shading="nearest", rasterized=True, **kw)
        top_cbar(fig, m, ax, lab)
        ax.plot(t_obs, gf, "w-", lw=1.2)  # observed onset, RBM
        if i == 1:
            ax.plot(t_mu, gf, "r--", lw=1.2)  # predicted onset from mu
        ax.set(xlabel=r"$t$", xlim=(0, t.max()))
        panel_label(ax, i + 3); simple_ticks(ax, x=[0, 0.5, 1]); image_ticks(ax)
    axs[0].set_ylabel(r"$g_f$")
    fig.legend([plt.Line2D([], [], color="w", lw=1.2, path_effects=[]), plt.Line2D([], [], color="r", ls="--", lw=1.2)],
               [r"$t^\ast$ from $I>10I(0)$", rf"$t^\ast$ from $\mu>{thr:.1f}$"], loc="outside lower center", ncols=2,
               fontsize=6.5, columnspacing=0.8, handlelength=1.2, facecolor="0.6")
    fig.savefig("figs/tfi1d_heatmaps_onset.pdf"); fig.savefig("figs/tfi1d_heatmaps_onset.png", dpi=200)
