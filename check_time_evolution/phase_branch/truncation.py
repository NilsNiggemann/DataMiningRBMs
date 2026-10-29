"""Infidelity of the top-N_trunc cumulant truncation vs N_trunc (as in Fig. 6 of arXiv:2512.11632), g_f = 0.5.

Truncated logs, each compared with the exact psi(t):
  exact_unwrap : exact state, phase continuous in t (phase_branch.evolve_unwrapped)
  rbm_natural  : RBM-TDVP state, the network's own log psi (a.s + sum_j log cosh theta_j)
  rbm_unwrap   : RBM-TDVP state, phase made continuous in t on the stored grid (spacing 1e-3)
The N_trunc largest |c_A| of each log are kept (plus c_0). Reads data/rbm_states_a*.npz, writes data/truncation.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_branch as pb  # noqa: E402
from quench2d import infidelity  # noqa: E402

T_SHOW = [0.0, 0.3, 0.5, 0.7, 0.9]
N_GRID = np.unique(np.concatenate([np.round(np.logspace(0, np.log10(pb.DIM - 1), 60)).astype(int),
                                   [pb.n_var(a) for a in pb.ALPHAS]]))


def truncation_curve(logpsi, psi_ref):
    c = pb.coeffs(logpsi)
    order = 1 + np.argsort(-np.abs(c[1:]))
    return [infidelity(np.exp(pb.W[:, S] @ c[S]), psi_ref) for S in (np.concatenate([[0], order[:n]]) for n in N_GRID)]


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    psi_ex, ph_ex = pb.evolve_unwrapped(0.5)
    rows = []

    def add(method, alpha, t, infs):
        rows.extend(dict(method=method, alpha=alpha, t=t, N=n, inf=v) for n, v in zip(N_GRID, infs))

    for t in T_SHOW:
        k = int(round(t / 0.01))
        add("exact_unwrap", np.nan, t, truncation_curve(np.log(np.abs(psi_ex[k])) + 1j * ph_ex[k], psi_ex[k]))
    for alpha in pb.ALPHAS:
        if not os.path.exists(f"data/rbm_states_a{alpha:g}.npz"):
            print(f"alpha={alpha}: no RBM states yet, skipped")
            continue
        d = np.load(f"data/rbm_states_a{alpha:g}.npz")
        psi, logpsi = d["psi"][:, ::-1], d["logpsi"][:, ::-1]  # NetKet ordering -> phase_branch ordering
        phase = np.unwrap(np.angle(psi), axis=0)
        step = np.abs(np.angle(psi[1:] / psi[:-1])).max()
        print(f"alpha={alpha}: largest phase step on the 1e-3 grid {step:.2f} rad")
        for t in T_SHOW:
            k, ke = int(round(t / 1e-3)), int(round(t / 0.01))
            ref = psi_ex[ke]
            assert abs(infidelity(psi[k], ref) - d["inf"][k]) < 1e-8 + 1e-3 * d["inf"][k], "ordering mismatch"
            add("rbm_natural", alpha, t, truncation_curve(logpsi[k], ref))
            add("rbm_unwrap", alpha, t, truncation_curve(np.log(np.abs(psi[k])) + 1j * phase[k], ref))
    pd.DataFrame(rows).to_csv("data/truncation.csv", index=False)
