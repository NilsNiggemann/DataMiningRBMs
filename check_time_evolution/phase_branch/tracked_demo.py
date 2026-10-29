"""Instrumented tracked fidelity fit at g_f=0.5, alpha=1 (K=120), for the appendix figures, plus ablations:
  full       : phase_branch.track (warm start + support swaps; fresh fit kept instead when better)
  warm       : warm start only, support fixed at its t=0 choice
  warm_swaps : warm start + support swaps, no fresh fallback
  fresh      : fresh weighted + fidelity fit at each time (the fallback inside track, started from the wrapped log)
  single     : single-time fidelity fit started from the unwrapped log (the 'direct' method of phase_branch.py)
  refit      : post-pass on 'full': fidelity fit of psi(t) on the previous step's (S, c), i.e. before the swaps
Writes data/tracked_demo_{variant}.npz with t, inf, S (T x K+1 masks), c (T x K+1), winner (full only).
Run with OMP_NUM_THREADS=1: python tracked_demo.py [variant ...]
"""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_branch as pb  # noqa: E402

G_F, ALPHA = 0.5, 1.0
VARIANTS = ["full", "warm", "warm_swaps", "fresh", "single"]


def fresh(psi, K):
    # same as inside pb.track (its starting log is the wrapped one)
    S, c = pb.weighted_fit(np.log(psi.astype(complex)), np.abs(psi) ** 2, K)
    c, inf = pb.direct_fit(psi, S, c)
    return np.asarray(S), c, inf


def run(variant):
    psi_t, phase = pb.evolve_unwrapped(G_F)
    psi_t, phase = psi_t[::pb.T_STRIDE], phase[::pb.T_STRIDE]
    K = pb.n_var(ALPHA)
    if variant == "refit":
        d = np.load("data/tracked_demo_full.npz")
        inf = [d["inf"][0]] + [pb.direct_fit(psi, S, c)[1] for psi, S, c in zip(psi_t[1:], d["S"][:-1], d["c"][:-1])]
        np.savez("data/tracked_demo_refit.npz", t=d["t"], inf=np.array(inf))
        return
    if variant == "single":
        out = [pb.weighted_fit(np.log(np.abs(psi)) + 1j * ph, np.abs(psi) ** 2, K) for psi, ph in zip(psi_t, phase)]
        fits = [pb.direct_fit(psi, S, c) for psi, (S, c) in zip(psi_t, out)]
        np.savez("data/tracked_demo_single.npz", t=pb.TIMES[::pb.T_STRIDE], S=np.array([S for S, _ in out]),
                 c=np.array([c for c, _ in fits]), inf=np.array([i for _, i in fits]))
        return
    S = c = None
    out = dict(inf=[], S=[], c=[], winner=[])
    for psi in psi_t:
        winner = "fresh"
        if S is None or variant == "fresh":
            S, c, inf = fresh(psi, K)
        elif variant == "warm":
            c, inf = pb.direct_fit(psi, S, c)
            winner = "warm"
        else:
            S, c, inf = pb.pursuit(psi, S, c, rounds=2)
            winner = "pursuit"
            if variant == "full":
                S0, c0, inf0 = fresh(psi, K)
                if inf0 < inf:
                    S, c, inf, winner = S0, c0, inf0, "fresh"
        for key, v in zip(out, (inf, S, c, winner)):
            out[key].append(v)
        print(f"{variant} t={len(out['inf']) * 0.02 - 0.02:.2f} inf={inf:.2e} {winner}", flush=True)
    np.savez(f"data/tracked_demo_{variant}.npz", t=pb.TIMES[::pb.T_STRIDE], **{k: np.array(v) for k, v in out.items()})


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    variants = sys.argv[1:] or VARIANTS + ["refit"]
    stage1 = [v for v in variants if v != "refit"]
    if stage1:
        with Pool(len(stage1)) as pool:
            pool.map(run, stage1, chunksize=1)
    if "refit" in variants:
        run("refit")
    # the full variant must reproduce the tracked predictor of phase_branch.py
    import pandas as pd
    p = pd.read_csv("data/predictors.csv")
    ref = p[(p.g_f == G_F) & (p.alpha == ALPHA) & (p.method == "tracked")].sort_values("t")["inf"].to_numpy()
    full = np.load("data/tracked_demo_full.npz")["inf"]
    assert np.allclose(full, ref, rtol=1e-6, atol=1e-14), np.abs(full / ref - 1).max()
    print("full variant reproduces data/predictors.csv")
