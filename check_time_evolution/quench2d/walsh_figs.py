"""Appendix figure: RBM infidelity vs mu_max and vs the fixed-basis Walshness W^Z_1 (Paul et al., arXiv:2610.00505)
for the 1D TFI quench (L=10, g_i=1.5) and the two 4x3 quench families. Caches W in data/walsh.npz; writes figs/walsh_scatter.pdf.
W^Z_alpha[psi] = (1/alpha) log sum_r p(r) exp(alpha |r|), p(r) = |psi_hat(r)|^2 normalised, psi_hat = Walsh-Hadamard transform."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.special import logsumexp
from scipy.stats import spearmanr
from quench2d import *

ALPHA = 1.0
D1 = "../../data/cumulants_Fermi_dirac_TDVP"


def walshness(psi, alpha=ALPHA):
    n = int(psi.size).bit_length() - 1
    x = np.array(psi, dtype=complex)
    for j in range(n):  # in-place fast Walsh-Hadamard transform
        v = x.reshape(2**j, 2, -1)
        a = v[:, 0].copy(); v[:, 0] += v[:, 1]; v[:, 1] = a - v[:, 1]
    p = np.abs(x) ** 2
    ham = sum((np.arange(2**n) >> k) & 1 for k in range(n))
    with np.errstate(divide="ignore"):
        return logsumexp(np.log(p / p.sum()) + alpha * ham) / alpha


def run(task):
    fam, p, times = task  # fam: "1D" (p = g_f), "A" (p = h_f), "B" (p = J_xx)
    if fam == "1D":
        N, nn = 10, [(i, i + 1) for i in range(9)]
        psi0 = ground_state(hamiltonian(N, nn, J=-1, hx=-1.5)); kw = dict(J=-1, hx=-p)
    else:
        N = LX * LY; nn, _ = lattice(LX, LY, True, False)
        psi0 = ground_state(hamiltonian(N, nn, **initial_kwargs(4.0))); kw = family_kwargs(fam, p)
    return [walshness(s) for s in evolve(hamiltonian(N, nn, **kw), psi0, times)]


if __name__ == "__main__":
    long = pd.read_csv("data/table_long.csv").sort_values(["fam", "p", "t"])
    long["mumax"] = long.groupby(["fam", "p"]).mu.cummax()
    run1 = np.load(f"{D1}/rbm_tdvp_quench.npz"); ce = np.load(f"{D1}/ED_CE/rbm_tdvp_quench.npz")
    t1, gf = run1["times"], run1["g_f_values"]
    if not os.path.exists("data/walsh.npz"):
        tasks = [("1D", g, t1) for g in gf]
        for fam in "AB":
            f = long[(long.fam == fam) & long.inf_a1.notna()]
            tasks += [(fam, p, np.sort(f[f.p == p].t.values)) for p in np.unique(f.p)]
        with ProcessPoolExecutor(8) as ex:
            res = list(ex.map(run, tasks))
        np.savez("data/walsh.npz", **{f"{i}": np.array(r) for i, r in enumerate(res)},
                 meta=np.array([(t[0], t[1]) for t in tasks], dtype=object))
    W = np.load("data/walsh.npz", allow_pickle=True)
    meta = W["meta"]
    # long-format frame: fam, p, t, logI, mumax, W
    rows = []
    for i, (fam, p) in enumerate(meta):
        w = W[str(i)]
        if fam == "1D":
            k = list(gf).index(p)
            rows.append(pd.DataFrame(dict(fam=fam, p=p, t=t1, logI=np.log10(np.clip(run1["infidelity_vs_exact"][k], 1e-12, None)),
                                          mumax=np.maximum.accumulate(ce["mu_grid"][k]), W=w)))
        else:
            f = long[(long.fam == fam) & np.isclose(long.p, p) & long.inf_a1.notna()]
            rows.append(pd.DataFrame(dict(fam=fam, p=p, t=f.t.values, logI=np.log10(f.inf_a1.clip(lower=1e-12).values),
                                          mumax=f.mumax.values, W=w)))
    df = pd.concat(rows)
    df.to_csv("data/walsh_long.csv", index=False)

    import matplotlib.pyplot as plt
    from plotstyle import STYLE, panel_label, simple_ticks
    names = {"1D": "1D TFI", "A": r"2D, $h_f$ quench", "B": r"2D, $J_{xx}$ quench"}
    with plt.rc_context(STYLE):
        fig, axs = plt.subplots(2, 3, figsize=(3.4 * 1.6, 3.6), sharey="col", layout="constrained")
        for c, fam in enumerate(["1D", "A", "B"]):
            d = df[df.fam == fam]
            for r, (x, lab) in enumerate([("mumax", r"$\mu_{\max}$"), ("W", r"$W_{1}^{Z}$")]):
                ax = axs[r, c]
                sc = ax.scatter(d[x], d.logI, c=d.t, s=2, cmap="viridis", rasterized=True)
                rho = spearmanr(d[x], d.logI)[0]
                ax.text(0.04, 0.95, rf"$\rho_s = {rho:.2f}$", transform=ax.transAxes, va="top", fontsize=7)
                ax.set_xlabel(lab); panel_label(ax, 3 * r + c, loc="lower right") if False else None
                ax.set_title(names[fam] if r == 0 else "", fontsize=8)
                simple_ticks(ax)
            axs[0, c].set_ylabel(r"$\log_{10} I$" if c == 0 else "")
            axs[1, c].set_ylabel(r"$\log_{10} I$" if c == 0 else "")
        fig.colorbar(sc, ax=axs, location="right", shrink=0.8, label=r"$t$")
        fig.savefig("figs/walsh_scatter.pdf"); fig.savefig("figs/walsh_scatter.png", dpi=110)
    print(df.groupby("fam")[["mumax", "W", "logI"]].corr(method="spearman").round(2))
