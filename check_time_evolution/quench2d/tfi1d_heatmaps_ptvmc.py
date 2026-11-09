"""p-tVMC infidelity heatmaps (RBM, ViT V2) vs mu_max(g_f,t), 1D TFI quench g_i=1.5, L=10.
Reads ../../data/vit_tdvp_scan/{rbm,vit} (partial runs -> NaN) and the cached cumulant grid; writes figs/tfi1d_heatmaps_ptvmc*.pdf/png."""
import glob, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plotstyle import STYLE, image_ticks, simple_ticks, top_cbar, panel_label

D = "../../data/cumulants_Fermi_dirac_TDVP"; S = "../../data/vit_tdvp_scan"
ce = np.load(f"{D}/ED_CE/rbm_tdvp_quench.npz"); gfa, ta = np.load(f"{D}/rbm_tdvp_quench.npz")["g_f_values"], ce["times"]
Mfull = np.maximum.accumulate(ce["mu_grid"], axis=1)           # mu_max(t) on the TDVP grid

def load(arch):
    fs = sorted(glob.glob(f"{S}/{arch}/*_gf*.npz")); d = [np.load(f) for f in fs]
    gf = np.array([float(x["g_f"]) for x in d]); o = np.argsort(gf)
    return gf[o], d[0]["times"], np.array([d[i]["infidelity"] for i in o])
gfr, t, Ir = load("rbm"); gfv, tv, Iv = load("vit"); assert np.allclose(gfr, gfv) and np.allclose(t, tv)
gf = gfr; M = np.array([np.interp(t, ta, Mfull[np.argmin(abs(gfa - g))]) for g in gf])   # nearest grid g_f (exact for every 3rd point)
assert np.allclose(gfa[[np.argmin(abs(gfa - g)) for g in gf]], gf, atol=1e-5)
# new longer runs (scan2) replace the 100-it ViT rows at g_f = 0.69 (run H: 500 it, ds 1e-6) and 3.0 (run B: 2000 it); partial runs -> NaN
Iv2, BEST = Iv.copy(), {0.692308: "H_g0.69_it500_ds1e-6", 3.0: "B_g3_it2000"}
for g, tag in BEST.items():
    d = np.load(glob.glob(f"{S}2/{tag}/L10_*.npz")[0]); k = np.argmin(abs(gf - g)); Iv2[k] = d["infidelity"]
lg = lambda I: np.log10(np.clip(I, 1e-12, None))

with plt.rc_context(STYLE):
    fig, axs = plt.subplots(1, 4, figsize=(4.4, 2.3), sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.01, w_pad=0.02)
    for ax, Z, lab, cmap, kw in zip(axs, [lg(Ir), lg(Iv), lg(Iv2), M], [r"$\log_{10}I$ RBM", r"$\log_{10}I$ ViT", r"$\log_{10}I$ ViT$^\ast$", r"$\mu_{\max}$"],
                                    ["inferno"] * 3 + ["viridis"], [dict(vmin=-6, vmax=0)] * 3 + [{}]):
        m = ax.pcolormesh(t, gf, Z, cmap=cmap, shading="nearest", rasterized=True, **kw)
        top_cbar(fig, m, ax, lab); ax.set(xlabel=r"$t$", xlim=(0, t.max()))
    for lv, c in zip([-4, -2], ["white", "red"]):   # RBM I contours on the mu_max panel
        axs[3].contour(t, gf, lg(Ir), levels=[lv], colors=c, linewidths=0.9)
    for g in BEST: axs[2].plot([0.97], [g], "<", ms=2.5, color="cyan", clip_on=False)   # rows replaced by longer runs
    axs[0].set_ylabel(r"$g_f$")
    for i, a in enumerate(axs): panel_label(a, i); simple_ticks(a, x=[0, 0.5, 1]); image_ticks(a)
    fig.savefig("figs/tfi1d_heatmaps_ptvmc.pdf"); fig.savefig("figs/tfi1d_heatmaps_ptvmc.png", dpi=110)

    fig, ax = plt.subplots(figsize=(3.4, 2.4), layout="constrained")
    for I, c, l in [(Ir, "red", "RBM"), (Iv, "black", "ViT V2")]:
        ax.plot(M.ravel(), I.ravel(), ".", ms=2, color=c, label=l, rasterized=True)
    ax.set(yscale="log", xlabel=r"$\mu_{\max}$", ylabel=r"$I$", ylim=(1e-8, 1)); ax.legend()
    fig.savefig("figs/tfi1d_scatter_ptvmc_mumax.pdf"); fig.savefig("figs/tfi1d_scatter_ptvmc_mumax.png", dpi=110)
