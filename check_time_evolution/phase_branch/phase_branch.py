"""Complex cumulant expansion of log psi(t) along the TFI quench g_i=1.5 -> g_f (L=10 open chain, Z basis),
with different choices of the phase branch, as predictors of RBM-TDVP infidelity.

Each method returns a K-term model log phi(s) = c_0 + sum_{A in S} c_A s^A (|S| = K = N_var(alpha)) and its
infidelity w.r.t. psi(t):
  wrapped  : top-K WHT coefficients of log|psi| + i angle(psi)
  unwrap   : same, phase made continuous in t on a fine grid (DT_FINE)
  branch   : unwrap, then per-config windings n(s) by alternating projection
             (n <- round((Im model - phi)/2pi), model <- top-K WHT); minimizes ||log psi_n - P_K log psi_n||_2
  weighted : |psi|^2-weighted least squares (weighted OMP support + branch updates). To second order the
             infidelity is the |psi|^2-weighted variance of the log error, so this is linearized infidelity.
  direct   : maximize the fidelity of exp(model) itself on the 'weighted' support (no log of psi, branch-free)
  tracked  : direct fit warm-started from the previous time step, with fidelity-gradient support swaps

Writes data/predictors.csv (one row per g_f, t, alpha, method) and data/rbm.csv.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from scipy.optimize import minimize

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "quench2d"))
from quench2d import evolve, ground_state, hamiltonian, infidelity, lattice  # noqa: E402

L, G_I = 10, 1.5
DIM = 2 ** L
W = hadamard(DIM).astype(float)  # W[s, A] = prod_{i in A} s_i (row s, column mask A); symmetric
ORDER = np.array([bin(a).count("1") for a in range(DIM)])
NN, _ = lattice(L, 1, pbc_x=False, pbc_y=False)
TIMES = np.round(np.arange(100) * 0.01, 10)  # grid of the RBM pickle
DT_FINE = 1e-4
ALPHAS = [0.5, 1.0, 2.0, 3.0]
G_FS = [0.3, 0.5, 0.7, 1.0, 2.0, 3.0]
RBM_PKL = "/home/nniggemann/shared/Inf_RBM_L10_g1.5_t1.0_Ntimes100.pkl"
TWO_PI = 2 * np.pi
T_STRIDE = 2  # every other RBM time point
N_PROC = 24  # shared machine: <= 30 CPUs; run with OMP_NUM_THREADS=1 (multithreaded BLAS is 3x slower here)


def n_var(alpha):
    return int(alpha * L * L + alpha * L + L)


def H(g):
    return hamiltonian(L, NN, J=-1.0, hx=-g)


def psi0():
    p = ground_state(H(G_I))
    return p * np.sign(p.sum())  # stoquastic: Perron-Frobenius positive


def evolve_unwrapped(g_f, dt_fine=DT_FINE):
    """psi(t) on TIMES and its phase made continuous in t on a grid of spacing dt_fine."""
    step = int(round(0.01 / dt_fine))
    t_fine = np.arange(len(TIMES) * step - step + 1) * dt_fine
    psi_f = evolve(H(g_f), psi0().astype(complex), t_fine)
    phase = np.unwrap(np.angle(psi_f), axis=0)[::step]
    return psi_f[::step], phase


def coeffs(logpsi):
    return W @ logpsi / DIM


def top_k(c, K):
    """Support: c_0 plus the K largest |c_A|, A != 0."""
    return np.concatenate([[0], 1 + np.argsort(-np.abs(c[1:]))[:K]])


def wht_truncation(logpsi, K):
    c = coeffs(logpsi)
    S = top_k(c, K)
    return S, W[:, S] @ c[S]


def branch_fix(logpsi, K, n_iter=200):
    """Alternating projection on the windings; each step lowers ||logpsi_n - P_K logpsi_n||_2."""
    for _ in range(n_iter):
        S, model = wht_truncation(logpsi, K)
        n = np.round((model.imag - logpsi.imag) / TWO_PI)
        if not n.any():
            break
        logpsi = logpsi + 1j * TWO_PI * n
    return S, model


def weighted_fit(logpsi, w, K, n_rounds=3):
    """Weighted OMP (batches of ~K/20) for the support, refit, then branch update; repeated n_rounds times."""
    sw = np.sqrt(w)
    batch = max(1, K // 20)
    for _ in range(n_rounds):
        S = [0]
        while True:
            A = sw[:, None] * W[:, S]
            c, *_ = np.linalg.lstsq(A, sw * logpsi, rcond=None)
            if len(S) == K + 1:
                break
            r = w * (logpsi - W[:, S] @ c)
            score = np.abs(W @ r)
            score[S] = -1
            S += list(np.argsort(-score)[:min(batch, K + 1 - len(S))])
        model = W[:, S] @ c
        n = np.round((model.imag - logpsi.imag) / TWO_PI)
        if not n[w > 1e-12 * w.max()].any():
            break
        logpsi = logpsi + 1j * TWO_PI * n
    return np.array(S), c


def direct_fit(psi, S, c_init):
    """Maximize |<psi|phi>|^2 / <phi|phi> over complex c_S, phi = exp(W_S c); L-BFGS on -log F."""
    WS = W[:, S]
    k = len(S)
    psic = psi.conj() / np.linalg.norm(psi)

    def f(x):
        lp = WS @ (x[:k] + 1j * x[k:])
        shift = lp.real.max()
        phi = np.exp(lp - shift)
        ov = psic @ phi
        nrm = np.sum(np.abs(phi) ** 2)
        g_ov = WS.T @ (psic * phi) / ov  # d log ov / dc
        g_n = WS.T @ (np.abs(phi) ** 2) / nrm
        val = -np.log(np.abs(ov) ** 2 / nrm)
        grad = np.concatenate([-2 * g_ov.real + 2 * g_n, 2 * g_ov.imag])
        return val, grad

    x0 = np.concatenate([c_init.real, c_init.imag])
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options=dict(maxiter=5000, ftol=1e-16, gtol=1e-13))
    return res.x[:k] + 1j * res.x[k:], -np.expm1(-res.fun)  # infidelity = 1 - F, accurate for small values


def full_gradient(psi, S, c):
    """|d(-log F)/dc_A| for every mask A at the model exp(W_S c)."""
    psic = psi.conj() / np.linalg.norm(psi)
    lp = W[:, S] @ c
    phi = np.exp(lp - lp.real.max())
    g_ov = W.T @ (psic * phi) / (psic @ phi)
    g_n = W.T @ (np.abs(phi) ** 2) / np.sum(np.abs(phi) ** 2)
    return np.hypot(-2 * g_ov.real + 2 * g_n, 2 * g_ov.imag)


def pursuit(psi, S, c, rounds):
    """Swap moves on the support: add the K/10 masks with largest fidelity gradient, refit, drop the K/10
    smallest |c_A|, refit; keep the move if it lowers the infidelity."""
    S = np.asarray(S)
    K = len(S) - 1
    m = max(1, K // 10)
    c, inf = direct_fit(psi, S, c)
    for _ in range(rounds):
        g = full_gradient(psi, S, c)
        g[S] = -1
        S2 = np.concatenate([S, np.argsort(-g)[:m]])
        c2, _ = direct_fit(psi, S2, np.concatenate([c, np.zeros(m)]))
        keep = np.concatenate([[0], 1 + np.argsort(-np.abs(c2[1:]))[:K]])
        c2, inf2 = direct_fit(psi, S2[keep], c2[keep])
        if inf2 < inf:
            S, c, inf = S2[keep], c2, inf2
    return S, c, inf


def track(psi_t, K, rounds=2):
    """Time-continuous fidelity-optimal K-term model: warm start (S, c) from the previous time, then
    pursuit. A fresh weighted+direct fit is tried as well and the better one kept."""
    out, S, c = [], None, None
    for psi in psi_t:
        S0, c0 = weighted_fit(np.log(psi.astype(complex)), np.abs(psi) ** 2, K)
        c0, inf0 = direct_fit(psi, S0, c0)
        if S is not None:
            S, c, inf = pursuit(psi, S, c, rounds)
        if S is None or inf0 < inf:
            S, c, inf = S0, c0, inf0
        out.append(inf)
    return np.array(out)


def run_point(args):
    """All methods and alphas at one (g_f, t)."""
    g_f, t, psi, phase = args
    la = np.log(np.abs(psi))
    log_wr = la + 1j * np.angle(psi)
    log_un = la + 1j * phase
    w = np.abs(psi) ** 2
    rows = []
    for alpha in ALPHAS:
        K = n_var(alpha)
        res = {}
        for name, lp in [("wrapped", log_wr), ("unwrap", log_un)]:
            _, model = wht_truncation(lp, K)
            res[name] = infidelity(np.exp(model), psi)
        _, model = branch_fix(log_un, K)
        res["branch"] = infidelity(np.exp(model), psi)
        S, c = weighted_fit(log_un, w, K)
        res["weighted"] = infidelity(np.exp(W[:, S] @ c), psi)
        res["direct"] = min(direct_fit(psi, S, c)[1], res["weighted"])
        for m, v in res.items():
            rows.append(dict(g_f=g_f, t=t, alpha=alpha, method=m, inf=v))
    print(f"done g_f={g_f} t={t:.2f}", flush=True)
    return rows


def run_chain(args):
    g_f, alpha, points = args
    infs = track([p[2] for p in points], n_var(alpha))
    print(f"done chain g_f={g_f} alpha={alpha}", flush=True)
    return [dict(g_f=g_f, t=p[1], alpha=alpha, method="tracked", inf=v) for p, v in zip(points, infs)]


def rbm_table():
    d = pickle.load(open(RBM_PKL, "rb"))
    rows = [dict(g_f=g_f, alpha=a, trial=tr, t=t, inf=v)
            for g_f in d for a in d[g_f] for tr in d[g_f][a] for t, v in zip(TIMES, d[g_f][a][tr])]
    return pd.DataFrame(rows)


def self_check():
    # unwrapped phase converged in dt_fine; c_0 + all-coefficient truncation is exact; direct >= weighted
    _, ph1 = evolve_unwrapped(0.5, 1e-4)
    _, ph2 = evolve_unwrapped(0.5, 2.5e-5)
    assert np.abs(ph1 - ph2).max() < 1e-6, "unwrap not converged in dt_fine"
    psi_t, ph = evolve_unwrapped(0.5)
    lp = np.log(np.abs(psi_t[50])) + 1j * ph[50]
    assert np.allclose(wht_truncation(lp, DIM - 1)[1], lp)
    # classical quench (g_f=0): unwrapped phase is -E(s)t, i.e. purely 0- and 2-body
    psi_c, ph_c = evolve_unwrapped(0.0)
    c = coeffs(ph_c[-1])
    assert np.abs(c[(ORDER != 0) & (ORDER != 2)]).max() < 1e-10
    print("self-check ok")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs("data", exist_ok=True)
    if "--check" in sys.argv:
        self_check()
        sys.exit()
    rbm_table().to_csv("data/rbm.csv", index=False)
    tasks = []
    for g_f in G_FS:
        psi_t, phase = evolve_unwrapped(g_f)
        tasks += [(g_f, t, psi_t[k], phase[k]) for k, t in enumerate(TIMES) if k % T_STRIDE == 0]
    with Pool(N_PROC) as pool:
        rows = sum(pool.map(run_point, tasks, chunksize=1), [])
    pd.DataFrame(rows).to_csv("data/predictors.csv", index=False)

    chains = [(g_f, alpha, [x for x in tasks if x[0] == g_f]) for g_f in G_FS for alpha in ALPHAS]
    with Pool(N_PROC) as pool:
        rows = sum(pool.map(run_chain, chains, chunksize=1), [])
    pd.concat([pd.DataFrame(rows), pd.read_csv("data/predictors.csv")]).to_csv("data/predictors.csv", index=False)
