"""Infidelity (ViT p-tVMC, RBM p-tVMC, RBM TDVP) and cumulant level mu of the exact state vs t, for three TFI quenches.
Run from this directory after vit_ptvmc_quench.py (ViT V2 and RBM) has finished."""
import sys
sys.path[:0] = ["..", "../quench2d"]
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ed_definitions as ed
from quench2d import analyze          # mu(t) of an exact trajectory
from plotstyle import STYLE, simple_ticks

DATA = "../../data/vit_tdvp"
S3 = "/tmp/claude-1006/-home-nniggemann-Projects-DataMiningRBMs/50b9c10a-a18a-4a30-8f93-f4fb00f592b3/scratchpad/s3"  # RBM TDVP RK45 runs (g_f = 0.5, 3.0)
OLD = np.load("../../data/cumulants_Fermi_dirac_TDVP/rbm_tdvp_quench.npz")                                      # RBM TDVP grid (g_f = 1.0 is index 13)
CE = np.load("../../data/cumulants_Fermi_dirac_TDVP/ED_CE/rbm_tdvp_quench.npz")
L, g_i = 10, 1.5
gfs = [0.5, 1.0, 3.0]

# --- analysis: exact mu(t) --------------------------------------------------------------------------------------
_, hi, H_i = ed.build_TFI_hamiltonian(L, g_i, 1.0, False); _, psi0 = ed.ground_state(H_i)
t_fine = np.linspace(0, 1, 101)
mu = {}
for g_f in gfs:
    ev, V = ed.full_diagonalization(ed.build_TFI_hamiltonian(L, g_f, 1.0, False)[2])
    mu[g_f] = analyze(ed.evolve_state(ev, V, psi0, t_fine), L, t_fine)["mu2"]  # stored mu_grid is the A0 = 2 N_real fit
ref = CE["mu_grid"][13]                                   # same quench from the stored grid, to check my mu
assert np.allclose(np.interp(t_fine, CE["times"], ref), mu[1.0], atol=1e-2, equal_nan=True), "mu differs from stored grid"

# --- figure -------------------------------------------------------------------------------------------------------
with plt.rc_context(STYLE):
    fig, axs = plt.subplots(2, 3, figsize=(7, 4.2), sharex=True, sharey="row", layout="constrained")
    for col, g_f in enumerate(gfs):
        vit = np.load(f"{DATA}/L10_V2_ptvmc_dt0.025_gf{g_f:.4f}.npz")
        rbm = np.load(f"{DATA}/L10_rbm_ptvmc_dt0.025_gf{g_f:.4f}.npz")
        if g_f == 1.0: tdvp_t, tdvp_i = OLD["times"], OLD["infidelity_vs_exact"][13]
        else:          d = np.load(f"{S3}/L10_rbm_rk45_dt0.01_gf{g_f:.4f}.npz"); tdvp_t, tdvp_i = d["times"], d["infidelity"]
        ax = axs[0, col]
        ax.plot(vit["times"], vit["infidelity"], color="black", label=r"ViT V2, p-tVMC")
        ax.plot(rbm["times"], rbm["infidelity"], color="red", label=r"RBM, p-tVMC")
        ax.plot(tdvp_t, tdvp_i, color="blue", label=r"RBM, TDVP")
        ax.set(yscale="log", ylim=(1e-7, 1), title=rf"$g_f = {g_f:g}$")
        axs[1, col].plot(t_fine, mu[g_f], color="black")
        axs[1, col].set(xlabel=r"$t$")
        simple_ticks(axs[1, col], x=[0, 0.5, 1])
        simple_ticks(ax, y=[1e-6, 1e-4, 1e-2, 1])
    axs[0, 0].set_ylabel(r"$\mathcal{I}$"); axs[1, 0].set_ylabel(r"$\mu$")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=3, fontsize=7.5)
    fig.savefig("figs_ptvmc_infidelity_mu.pdf"); fig.savefig("figs_ptvmc_infidelity_mu.png", dpi=130)
