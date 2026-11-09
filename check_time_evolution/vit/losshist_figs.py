"""Training loss vs inner iteration for selected time steps (g_f=3.0, ViT V2, 600 it/step), two diag_shift values."""
import sys, glob, numpy as np, matplotlib
sys.path[:0] = ["..", "../quench2d"]
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import STYLE
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True, layout="constrained")
    for ax, tag in zip(axs, ["g3_ds1e-4", "g3_ds1e-6"]):
        d = np.load(glob.glob(f"../../data/vit_tdvp_losshist/{tag}/L10_*.npz")[0]); h, t = d["loss_hist"], d["times"]
        ks = [k for k in (2, 6, 10, 14, 18, 22, 24) if k < len(h) and not np.isnan(h[k, 0])]
        for k, c in zip(ks, plt.cm.viridis(np.linspace(0, 0.9, len(ks)))):
            ax.plot(np.arange(h.shape[1]), h[k], color=c, lw=1, label=rf"$t={t[k]:.2f}$")
        ax.set(yscale="log", xlabel="SR iteration within step", title=rf"$g_f=3$, diag_shift ${d['diag_shift']:.0e}$".replace("e-0", "e-"))
        ax.legend(fontsize=6, ncols=2)
    axs[0].set_ylabel(r"loss $\mathcal{L}$")
    fig.savefig("figs_losshist.pdf"); fig.savefig("figs_losshist.png", dpi=100)
