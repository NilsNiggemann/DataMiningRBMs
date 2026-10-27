"""ED scans of the 4x3 quench families (PLAN_quench_family.md step 2). Writes data/ed_*.npz.

usage: python ed_scan.py [scan ...]   (default: all scans)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from quench2d import *

NPROC = 28
TMAX = 1.5

# scan name: list of task tuples (fam, p, h_f, h_i, dt)
SCANS = {
    "B": [("B", p, 1.0, 4.0, 0.01) for p in np.linspace(-1.5, 1.5, 31)],
    "A": [("A", p, None, 4.0, 0.01) for p in np.linspace(0.2, 4, 20)],
    "plane": [("B", p, hf, 4.0, 0.02) for hf in [0.5, 1, 1.5, 2, 3, 4] for p in np.linspace(-1.5, 1.5, 16)],
    # robustness: same points at dt/2 (phase unwrapping) and at h_i = 6 (initial-state sensitivity)
    "B_dthalf": [("B", p, 1.0, 4.0, 0.005) for p in [-0.9, 0.0, 0.9]],
    "B_hi6": [("B", p, 1.0, 6.0, 0.01) for p in [-0.9, 0.0, 0.9]],
    "A_hi6": [("A", p, None, 6.0, 0.01) for p in [0.8, 2.0, 3.6]],
}


def run(task):
    fam, p, hf, h_i, dt = task
    N = LX * LY
    nn, _ = lattice(LX, LY, True, False)
    times = np.round(np.arange(0, TMAX + 1e-9, dt), 10)
    psi0 = ground_state(hamiltonian(N, nn, **initial_kwargs(h_i)))
    kw = family_kwargs(fam, p) if fam == "A" else family_kwargs(fam, p, hf)
    return analyze(evolve(hamiltonian(N, nn, **kw), psi0, times), N, times)


def calibration_1d():
    """1D TFI L=12 OBC, gs at g=1.5 -> g_f (plan section 2 calibration)."""
    N = 12
    nn = [(i, i + 1) for i in range(N - 1)]
    times = np.round(np.arange(0, TMAX + 1e-9, 0.01), 10)
    psi0 = ground_state(hamiltonian(N, nn, J=-1, hx=-1.5))
    gfs = np.array([0.2, 0.5, 0.8, 1.2])
    res = [analyze(evolve(hamiltonian(N, nn, J=-1, hx=-g), psi0, times), N, times) for g in gfs]
    np.savez("data/ed_cal1d.npz", vals=gfs, times=times, **{k: np.array([r[k] for r in res]) for k in res[0]})


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    names = sys.argv[1:] or list(SCANS) + ["cal1d"]
    with ProcessPoolExecutor(NPROC) as ex:
        cal = ex.submit(calibration_1d) if "cal1d" in names else None
        for name in [n for n in names if n in SCANS]:
            tasks = SCANS[name]
            res = list(ex.map(run, tasks))
            fam, p, hf, h_i, dt = zip(*tasks)
            times = np.round(np.arange(0, TMAX + 1e-9, dt[0]), 10)
            grid = {k: np.array([r[k] for r in res]) for k in res[0]}
            np.savez(f"data/ed_{name}.npz", vals=np.array(p), hf=np.array(hf, dtype=float),
                     h_i=np.array(h_i), times=times, **grid)
            nfail = {k: int(np.isnan(grid[k]).sum()) for k in ["mu", "mu2", "mu_ph"]}
            print(f"{name}: {len(tasks)} pts, NaN fits {nfail}", flush=True)
        if cal:
            cal.result()
            print("cal1d done", flush=True)
