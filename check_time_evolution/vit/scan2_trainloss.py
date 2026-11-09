"""Training loss (final inner-loop infidelity per time step) of ViT p-tVMC infidelity I(t) for longer / other-SR-settings runs (data/vit_tdvp_scan2) vs the 100-it scan and RBM. Partial runs ok."""
import sys, glob, numpy as np, matplotlib
sys.path[:0] = ["..", "../quench2d"]
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import STYLE, simple_ticks

S, S1 = "../../data/vit_tdvp_scan2", "../../data/vit_tdvp_scan"
panels = [(3.0, "g3", "3.000000"), (0.692308, "g0.69", "0.692308")]
fmt = lambda t: t.split("_", 1)[1].replace("_", ", ")
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True, layout="constrained")
    for ax, (g, key, gs) in zip(axs, panels):
        base = sorted(glob.glob(f"{S1}/vit/*gf{g:.4f}.npz")); rbm = sorted(glob.glob(f"{S1}/rbm/*gf{g:.4f}.npz"))
        for f, c, l in [(rbm[0], "red", "RBM 30 it"), (base[0], "0.6", "ViT 100 it (scan)")]:
            d = np.load(f); ax.plot(d["times"], d["opt_infidelity"], color=c, lw=1.5, label=l)
        for f, c in zip(sorted(glob.glob(f"{S}/*_{key}_*")), plt.cm.viridis(np.linspace(0, 0.9, 6))):
            d = np.load(glob.glob(f + "/*.npz")[0]); tag = f.split("/")[-1]
            ax.plot(d["times"], d["opt_infidelity"], color=c, lw=1, label=fmt(tag))
        ax.set(yscale="log", xlabel=r"$t$", ylim=(1e-9, 1e-1), title=rf"$g_f={g:.2f}$"); simple_ticks(ax, x=[0, 0.5, 1])
    axs[0].set_ylabel(r"$\mathcal{L}$")
    for ax in axs: ax.legend(fontsize=5.5, loc="lower right", ncols=1)
    fig.savefig("figs_scan2_trainloss.pdf"); fig.savefig("figs_scan2_trainloss.png", dpi=100)
