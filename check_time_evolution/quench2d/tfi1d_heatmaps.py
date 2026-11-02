"""1D TFI quench figure: RBM infidelity | mu | S over (t, g_f), g_i = 1.5. Contours of mu (white) on (a).
Reads the cached runs in ../../data/cumulants_Fermi_dirac_TDVP; writes figs/tfi1d_heatmaps.pdf."""
import numpy as np, matplotlib
from scipy.ndimage import gaussian_filter
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from plotstyle import STYLE, simple_ticks, top_cbar, panel_label

D = "../../data/cumulants_Fermi_dirac_TDVP"
run, ce, ee = (np.load(f"{D}/{f}") for f in ("rbm_tdvp_quench.npz", "ED_CE/rbm_tdvp_quench.npz", "ED_CE/ee_rbm_tdvp_quench.npz"))
t, gf = run["times"], run["g_f_values"]
assert np.allclose(t, ce["times"]) and np.allclose(gf, ee["g_f_values"])
I, mu, S = run["infidelity_vs_exact"], ce["mu_grid"], ee["ee_grid"]

with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 3, figsize=(3.4, 2.3), sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.01, w_pad=0.02)
    for ax, Z, lab, cmap, kw in zip(axs, [np.log10(np.clip(I, 1e-12, None)), mu, S], [r"$\log_{10}I$", r"$\mu$", r"$S$ [nats]"],
                                    ["inferno", "viridis", "cividis"], [dict(vmin=-6), {}, {}]):
        m = ax.pcolormesh(t, gf, Z, cmap=cmap, shading="nearest", rasterized=True, **kw)
        top_cbar(fig, m, ax, lab)
        ax.set(xlabel=r"$t$", xlim=(0, t.max()))
    I_lv = np.array([-5.5, -4, -3])  # log10 I contours drawn on the mu panel
    axs[1].contour(t, gf, np.log10(np.clip(I, 1e-12, None)), levels=I_lv, colors="white", linestyles="-", linewidths=0.8)
    axs[1].plot([], [], "-", lw=0.8, color="0.6", label=rf"$\log_{{10}}I = {I_lv[0]:g}, {I_lv[1]:g}, {I_lv[2]:g}$")
    axs[0].set_ylabel(r"$g_f$")
    for i, a in enumerate(axs):
        panel_label(a, i + 3)
        simple_ticks(a, x=[0, 0.5, 1])
    fig.legend(*axs[1].get_legend_handles_labels(), loc="outside lower center", ncols=1, fontsize=7, facecolor="0.85")
    fig.savefig("figs/tfi1d_heatmaps.pdf"); fig.savefig("figs/tfi1d_heatmaps.png", dpi=110)
