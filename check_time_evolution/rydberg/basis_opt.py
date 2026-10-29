"""Local-basis optimisation for R1 (sq4x3, |+x>, Omega/V_nn = g): minimise J = max(mu, mu_ph) of psi' = W psi,
W = prod_i R_y(theta_i) R_z(phi_i) (plans/plan_rydberg.md).

usage: python basis_opt.py check                  self-checks
       python basis_opt.py snap  T               step 1: fixed basis at one snapshot time -> data/basis/snap_t<T>.npz
       python basis_opt.py traj                  step 2a: one fixed basis for mean_t J   -> data/basis/traj_fixed.npz
       python basis_opt.py spline LAM            step 2b: smooth W(t)                    -> data/basis/traj_spline_lam<LAM>.npz
       python basis_opt.py sup BASIS SEED [ALPHA]  step 3a: supervised RBM chain, BASIS = z | data/basis/<file>.npz
                                                 -> data/basis/sup_<name>_s<SEED>_a<ALPHA>.npz
       step 3b: BASIS=data/basis/<file>.npz python rydberg_rbm.py R1 sq4x3 +x 0.1 rbm1   (TDVP in the rotated frame)
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.interpolate import CubicSpline
from rydberg2d import sparse_parts, evolve, order_of_masks, fwht, fit_mu, NOISE, entanglement, P

ID, LAT, INIT = "R1", "sq4x3", "+x"
N, LY = 12, 3
TMAX, NSAVE, SUB = 30.0, 201, 10  # 201 save times, phase unwrapped on a SUB-times finer grid
FLOOR, PEN = -30.0, 10.0  # mu = -inf -> FLOOR; NaN / runaway fit -> PEN
ORDER = order_of_masks(N)
OUT = "data/basis"
SX = np.array([[0, 1], [1, 0]], complex); SY = np.array([[0, -1j], [1j, 0]]); SZ = np.diag([1.0 + 0j, -1.0])
PAULI = {"X": SX, "Y": SY, "Z": SZ}


# ----------------------------------------------------------------------------- trajectory
def trajectory(g=0.1):
    """Exact psi(t) on the fine grid t_f (TMAX/((NSAVE-1)*SUB) steps); save times are t_f[::SUB]."""
    from rydberg_screen import spec
    s = spec(ID, LAT, INIT, g)
    d, o = sparse_parts(s["segs"][0][1], N)
    t_f = np.linspace(0, TMAX, (NSAVE - 1) * SUB + 1)
    _, ev, _ = evolve(s["psi0"], [(1.0, d, o)], t_f, min(t_f[1], 0.1 / np.abs(o).sum(1).max()))
    return t_f, np.exp(ev["logabs"] + 1j * ev["ph"]), s


# ----------------------------------------------------------------------------- rotation
def site_u(th, ph):
    """W_i = R_y(th) R_z(ph), arrays broadcast; -> (..., 2, 2)."""
    th, ph = np.broadcast_arrays(np.asarray(th, float), np.asarray(ph, float))
    c, s = np.cos(th / 2), np.sin(th / 2)
    e = np.exp(-0.5j * ph)
    Ry = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    return Ry * np.stack([e, e.conj()], -1)[..., None, :]


def kron_all(U):
    """(..., n, 2, 2) -> (..., 2^n, 2^n), site 0 most significant."""
    out = U[..., 0, :, :]
    for k in range(1, U.shape[-3]):
        a, b = out, U[..., k, :, :]
        out = (a[..., :, None, :, None] * b[..., None, :, None, :]).reshape(*a.shape[:-2], a.shape[-2] * 2, a.shape[-1] * 2)
    return out


def rotate(psi, U):
    """psi (T, 2^N) or (2^N,), U (N,2,2) fixed or (T,N,2,2) per time -> W psi."""
    h = N // 2
    A, B = kron_all(U[..., :h, :, :]), kron_all(U[..., h:, :, :])
    m = psi.reshape(*psi.shape[:-1], 2 ** h, 2 ** h)
    return (A @ m @ np.swapaxes(B, -1, -2)).reshape(psi.shape)


def angles(x, level):
    """Parameter vector -> (theta, phi) per site. level: 'uni' (2), 'row' (2*LY, site i=x*LY+y), 'site' (2N)."""
    x = np.asarray(x, float)
    if level == "uni":
        return np.full(N, x[0]), np.full(N, x[1])
    if level == "row":
        y = np.arange(N) % LY
        return x[:LY][y], x[LY:][y]
    return x[:N], x[N:]


# ----------------------------------------------------------------------------- proxies
def _mu(c):
    alive = np.abs(c[1:]) > np.exp(NOISE)
    if ORDER[1:][alive].max(initial=0) <= 2:  # exactly <= 2-body: mu = -inf
        return FLOOR
    mu = fit_mu(c)[0]
    return mu if np.isfinite(mu) else PEN


BITS = (np.arange(2 ** N)[:, None] >> (N - 1 - np.arange(N))) & 1  # (2^N, N), 1 <-> s = -1


def product_phase(U0):
    """Exact (not wrapped) phase sum_i arg <s_i|U0_i|+> of the rotated start W(0)|+x>, a one-body function of s."""
    v = U0 @ (np.ones(2) / np.sqrt(2))
    return np.angle(v)[np.arange(N), BITS].sum(1)


def proxies(psi_rot, idx, ph0=None):
    """mu, mu_ph, w_hi of the rotated fine trajectory psi_rot (T, 2^N) at the fine indices idx; the phase is
    unwrapped along the whole fine grid up to max(idx), starting from the branch ph0 at t = 0 (np.angle would wrap a
    one-body phase sum into many-body 2 pi jumps, which the time unwrapping then keeps)."""
    la = np.log(np.abs(psi_rot[idx]) + 1e-300)
    ph = np.unwrap(np.angle(psi_rot[: max(idx) + 1]), axis=0)
    if ph0 is not None:
        ph = ph + (ph0 - ph[0])
    ph = ph[idx]
    C, Cp = fwht(la), fwht(ph - ph.mean(1, keepdims=True))
    mu = np.array([_mu(c) for c in C])
    mu_ph = np.array([_mu(c) for c in Cp])
    w = C[:, 1:] ** 2
    w_hi = w[:, ORDER[1:] > 4].sum(1) / np.maximum(w.sum(1), 1e-300)
    return mu, mu_ph, w_hi


def J_of(mu, mu_ph):
    return np.maximum(mu, mu_ph)


# ----------------------------------------------------------------------------- rotated Hamiltonian
def so3(u):
    """R_ab with u sigma_a u^dag = sum_b R_ab sigma_b (a, b in X, Y, Z)."""
    S = [SX, SY, SZ]
    return np.array([[0.5 * np.trace(S[b] @ u @ S[a] @ u.conj().T).real for b in range(3)] for a in range(3)])


def rotate_terms(terms, U):
    """Pauli-string terms of H -> terms of W H W^dag (one-body products only: each site op is rotated)."""
    R = [so3(u) for u in U]
    out = {}
    for c, s in terms:
        sites = [(i, "XYZ".index(o)) for i, o in enumerate(s) if o != "I"]
        expansions = [(c, {})]
        for i, a in sites:
            expansions = [(cc * R[i][a, b], {**ops, i: "XYZ"[b]}) for cc, ops in expansions for b in range(3)
                          if abs(R[i][a, b]) > 1e-14]
        for cc, ops in expansions:
            key = P(N, ops)
            out[key] = out.get(key, 0.0) + cc
    return [(c, s) for s, c in out.items() if abs(c) > 1e-14]


def frame_terms(U, dU):
    """One-body generator i (dW/dt) W^dag in Pauli strings (identity part dropped)."""
    out = []
    for i in range(N):
        K = 1j * dU[i] @ U[i].conj().T
        for o, S in PAULI.items():
            c = 0.5 * np.trace(S @ K)
            assert abs(c.imag) < 1e-10
            if abs(c.real) > 1e-14:
                out.append((c.real, P(N, {i: o})))
    return out


# ----------------------------------------------------------------------------- self-checks
def check():
    import scipy.sparse as sp
    t_f, psi, s = trajectory()
    ref = np.load("data/screen/R1_sq4x3_+x.npz")
    k = int(np.argmin(np.abs(ref["g"] - 0.1)))
    idx = np.arange(0, len(t_f), SUB)
    mu, mu_ph, _ = proxies(psi, idx)
    clip = lambda a: np.where(np.isinf(a), FLOOR, a)
    dmu, dph = np.abs(mu - clip(ref["mu"][k])), np.abs(mu_ph - clip(ref["mu_ph"][k]))
    print(f"W=I vs stored: max |d mu| = {np.nanmax(dmu):.3g}, max |d mu_ph| = {np.nanmax(dph):.3g}")
    assert np.nanmax(dmu) < 0.05 and np.nanmax(dph) < 0.05, "W = I must reproduce the stored proxies"
    rng = np.random.default_rng(0)
    U = site_u(rng.uniform(0, np.pi, N), rng.uniform(0, 2 * np.pi, N))
    for kk in (40, 200):
        p, q = psi[kk * SUB], rotate(psi[kk * SUB], U)
        assert abs(np.linalg.norm(q) - 1) < 1e-10, "norm"
        assert abs(entanglement(p, N)["S"] - entanglement(q, N)["S"]) < 1e-8, "S invariant under local W"
    # rotate() equals the explicit kron product
    W = kron_all(U)
    assert np.allclose(rotate(psi[500], U), W @ psi[500])
    # W H W^dag from rotated Pauli strings
    terms = s["segs"][0][1]
    d, o = sparse_parts(terms, N); H = (o + sp.diags(d)).toarray()
    d2, o2 = sparse_parts(rotate_terms(terms, U), N)
    assert np.allclose((o2 + sp.diags(d2)).toarray(), W @ H @ W.conj().T), "H' = W H W^dag"
    # phase unwrapping converged in the grid: mu_ph from every 2nd fine point agrees for a random W
    q = rotate(psi[: 20 * SUB + 1], U)
    a = proxies(q, [20 * SUB])[1][0]; b = proxies(q[::2], [10 * SUB])[1][0]
    print(f"random W at t=3: mu_ph fine {a:.3f}, half-fine {b:.3f}")
    # a pure R_z (one-body phases) leaves J unchanged once the t = 0 phase branch is fixed
    global PSI
    PSI = psi
    for k in (11, 200):
        J0 = J_of(*evaluate(np.zeros(2 * N), "site", [k])[:2])
        Jz = J_of(*evaluate(np.concatenate([np.zeros(N), rng.uniform(-np.pi, np.pi, N)]), "site", [k])[:2])
        assert abs(J0 - Jz) < 1e-6, f"R_z changed J at t={k * 0.15}: {J0} vs {Jz}"
    # exact evolution in the rotated frame equals W(t) psi(t): fixed W, and a spline W(t) with the frame term
    import tempfile
    s3 = dict(s, t_save=s["t_save"][:21])  # t <= 3
    xk = np.concatenate([np.tile(rng.normal(0, 0.3, 2 * N), (len(KNOTS), 1))
                         + rng.normal(0, 0.2, (len(KNOTS), 2 * N))]).ravel()
    with tempfile.TemporaryDirectory() as tmp:
        np.savez(f"{tmp}/fix.npz", x=xk[None, : 2 * N]); np.savez(f"{tmp}/spl.npz", xk=xk, knots=KNOTS)
        for f, kn in [("fix", None), ("spl", KNOTS)]:
            r = rotated_spec(s3, f"{tmp}/{f}.npz", dt_seg=0.015)
            parts = [(dd, *sparse_parts(tt, N)) for dd, tt in r["segs"]]
            _, ev, _ = evolve(r["psi0"], parts, r["t_save"], 0.015, r["period"])
            got = np.exp(ev["logabs"][-1] + 1j * ev["ph"][-1])
            want = rotate(psi[20 * SUB], U_of(xk[: 2 * N], "site") if kn is None else U_of(xk, "site", kn, [3.0])[0])
            fid = abs(np.vdot(want, got)) ** 2
            print(f"rotated-frame evolution ({f}): 1 - fidelity = {1 - fid:.2e}")
            assert 1 - fid < 1e-6, "H'(t) must generate W(t) psi(t)"
    print("ok")


# ----------------------------------------------------------------------------- optimisation
STRIDE = 2  # objectives unwrap on every 2nd fine point (converged to ~0.005 in mu_ph, see check())
NPROC = 30
PSI = None  # fine trajectory, loaded in the parent and shared with the forked workers
T_TRAJ = np.round(np.linspace(8, 200, 25)).astype(int)  # save indices of the trajectory objective (t = 1.2 ... 30)
LEVEL_DIM = {"uni": 2, "row": 2 * LY, "site": 2 * N}
BOUNDS = lambda level: [(-np.pi / 2, np.pi / 2)] * (LEVEL_DIM[level] // 2) + [(-np.pi, np.pi)] * (LEVEL_DIM[level] // 2)


def load_psi():
    f = f"{OUT}/psi_g0.1.npy"
    if not os.path.exists(f):
        os.makedirs(OUT, exist_ok=True)
        np.save(f, trajectory()[1])
    return np.load(f)


def U_of(x, level, knots=None, t=None):
    """Fixed W (N,2,2), or with knots a cubic spline W(t) (len(t),N,2,2); x then holds (n_knots, 2*n_level) values."""
    if knots is None:
        return site_u(*angles(x, level))
    v = CubicSpline(knots, np.asarray(x, float).reshape(len(knots), -1))(t)  # (T, 2*n_level)
    th, ph = np.array([angles(r, level) for r in v]).transpose(1, 0, 2)
    return site_u(th, ph)


def evaluate(x, level, save_idx, knots=None, stride=STRIDE, psi=None):
    """mu, mu_ph, w_hi at the save indices for the (fixed or spline) basis x."""
    psi = PSI if psi is None else psi
    kmax = max(save_idx) * SUB
    sl = psi[: kmax + 1 : stride]
    t = np.arange(0, kmax + 1, stride) * TMAX / ((NSAVE - 1) * SUB)
    U = U_of(x, level, knots, t)
    return proxies(rotate(sl, U), np.asarray(save_idx) * SUB // stride, product_phase(U if knots is None else U[0]))


def frame_cost(x, level, knots, n=601):
    """int_0^T sum_i ||i dW_i/dt W_i^dag||_F^2 / 2 dt: the squared strength of the one-body frame term."""
    t = np.linspace(0, TMAX, n)
    U = U_of(x, level, knots, t)
    dU = np.gradient(U, t, axis=0)
    K = np.einsum("tiab,ticb->tiac", dU, U.conj())
    return np.trapezoid(0.5 * np.sum(np.abs(K) ** 2, axis=(1, 2, 3)), t)


class Budget(Exception):
    pass


class Objective:
    """mean_t J(t) (+ lam * frame cost); records every call and the best point; raises Budget after maxfev calls."""

    def __init__(self, level, save_idx, maxfev, knots=None, lam=0.0):
        self.level, self.save_idx, self.maxfev, self.knots, self.lam = level, list(save_idx), maxfev, knots, lam
        self.nfev, self.best, self.xbest = 0, np.inf, None

    def __call__(self, x):
        if self.nfev >= self.maxfev:
            raise Budget
        self.nfev += 1
        mu, mu_ph, _ = evaluate(x, self.level, self.save_idx, self.knots)
        f = float(np.mean(J_of(mu, mu_ph)))
        if self.lam:
            f += self.lam * frame_cost(x, self.level, self.knots)
        if f < self.best:
            self.best, self.xbest = f, np.array(x, float)
        return f


def run_one(task):
    """One (algorithm, start) optimisation. task: dict(algo, x0, level, save_idx, maxfev, seed, start, knots, lam)."""
    from scipy import optimize as so
    t0 = time.time()
    tk = task
    f = Objective(tk["level"], tk["save_idx"], tk["maxfev"], tk.get("knots"), tk.get("lam", 0.0))
    x0 = np.asarray(tk["x0"], float)
    nk = 1 if tk.get("knots") is None else len(tk["knots"])
    bounds = BOUNDS(tk["level"]) * nk
    lo, hi = np.array(bounds).T
    rng = np.random.default_rng(tk["seed"])
    try:
        if tk["algo"] == "nm":
            simplex = np.vstack([x0, x0 + 0.1 * np.eye(len(x0))])
            so.minimize(f, x0, method="Nelder-Mead", options=dict(maxfev=10 ** 9, xatol=1e-4, fatol=1e-4, adaptive=True,
                                                                  initial_simplex=simplex))
        elif tk["algo"] == "powell":
            so.minimize(f, x0, method="Powell", options=dict(maxfev=10 ** 9, xtol=1e-3, ftol=1e-5, direc=0.3 * np.eye(len(x0))))
        elif tk["algo"] == "de":
            pop = max(4, 120 // len(x0))
            so.differential_evolution(f, bounds, x0=np.clip(x0, lo, hi), popsize=pop, maxiter=10 ** 6, tol=0, seed=tk["seed"],
                                      polish=False, init="latinhypercube")
        elif tk["algo"] == "da":
            so.dual_annealing(f, bounds, x0=np.clip(x0, lo, hi), maxfun=10 ** 9, maxiter=10 ** 6, seed=tk["seed"],
                              minimizer_kwargs=dict(method="L-BFGS-B", bounds=bounds, options=dict(eps=1e-3)))
        elif tk["algo"] == "bh":
            so.basinhopping(f, x0, niter=10 ** 6, stepsize=0.3, seed=tk["seed"],
                            minimizer_kwargs=dict(method="L-BFGS-B", options=dict(eps=1e-3)))
        else:
            raise ValueError(tk["algo"])
    except Budget:
        pass
    return dict(algo=tk["algo"], start=tk["start"], level=tk["level"], J=f.best, x=f.xbest, nfev=f.nfev,
                wall=time.time() - t0)


def pool_map(fn, tasks):
    from multiprocessing import get_context
    with get_context("fork").Pool(min(NPROC, len(tasks))) as p:
        out = []
        for i, r in enumerate(p.imap(fn, tasks)):  # ordered: the grid scan reshapes results by position
            out.append(r)
            if "algo" in r:
                print(f"  [{i + 1}/{len(tasks)}] {r['level']:4s} {r['algo']:6s} {r['start']:14s} J={r['J']:.3f} "
                      f"nfev={r['nfev']} {r['wall']:.0f}s", flush=True)
        return out


def lift(x, src, dst):
    """Parameters of level src -> equivalent parameters of the finer level dst."""
    th, ph = angles(x, src)
    if dst == "site":
        return np.concatenate([th, ph])
    return np.concatenate([th[:LY], ph[:LY]])  # uni -> row


def _grid_eval(args):
    x, save_idx = args
    mu, mu_ph, _ = evaluate(x, "uni", save_idx)
    return dict(x=np.array(x), J=float(np.mean(J_of(mu, mu_ph))))


def hierarchy(save_idx, warm=(), n_random=32, budgets=(1500, 4000, 12000), name=""):
    """(a) uniform grid scan + local polish, (b) row-dependent, (c) site-dependent; several starts and algorithms.
    warm: extra (label, site-level x) starts. Returns the list of all runs."""
    rng = np.random.default_rng(0)
    runs = []
    # (a) uniform grid
    g_th, g_ph = np.linspace(-np.pi / 2, np.pi / 2, 61), np.linspace(-np.pi, np.pi, 61)
    grid = pool_map(_grid_eval, [((a, b), save_idx) for a in g_th for b in g_ph])
    Jg = np.array([r["J"] for r in grid]).reshape(61, 61)
    xg = np.array([r["x"] for r in grid])
    order = np.argsort(Jg.ravel())
    top = []  # best grid points, at least 0.15 rad apart
    for i in order:
        if all(np.linalg.norm(xg[i] - xg[j]) > 0.15 for j in top):
            top.append(i)
        if len(top) == 4:
            break
    print(f"{name} uniform grid: best J={Jg.ravel()[top[0]]:.3f} at {xg[top[0]]}", flush=True)
    tasks = [dict(algo=a, x0=xg[i], level="uni", save_idx=save_idx, maxfev=300, seed=k, start=f"grid{k}")
             for k, i in enumerate(top) for a in ("nm", "powell")]
    tasks += [dict(algo="nm", x0=np.zeros(2), level="uni", save_idx=save_idx, maxfev=300, seed=0, start="identity")]
    runs += pool_map(run_one, tasks)
    best_uni = sorted([r for r in runs if r["level"] == "uni"], key=lambda r: r["J"])
    # (b) row-dependent
    B = budgets[0]
    starts = [("identity", np.zeros(6))] + [(f"uni{k}", lift(r["x"], "uni", "row")) for k, r in enumerate(best_uni[:3])]
    starts += [(f"rand{k}", np.concatenate([rng.normal(0, 0.3, LY), rng.uniform(-np.pi, np.pi, LY)])) for k in range(12)]
    tasks = [dict(algo=a, x0=x0, level="row", save_idx=save_idx, maxfev=B, seed=k, start=lab)
             for k, (lab, x0) in enumerate(starts) for a in ("nm", "powell")]
    tasks += [dict(algo=a, x0=starts[1][1], level="row", save_idx=save_idx, maxfev=3 * B, seed=s, start="uni0")
              for a in ("de", "da", "bh") for s in (0, 1)]
    runs += pool_map(run_one, tasks)
    best_row = sorted([r for r in runs if r["level"] == "row"], key=lambda r: r["J"])
    # (c) site-dependent
    B = budgets[1]
    th_u, ph_u = angles(best_uni[0]["x"], "uni")
    stag = np.where((np.arange(N) // LY + np.arange(N) % LY) % 2 == 0, 1, -1)
    starts = [("identity", np.zeros(2 * N)), ("uni0", lift(best_uni[0]["x"], "uni", "site")),
              ("stag_th", np.concatenate([th_u * stag, ph_u])), ("stag_ph", np.concatenate([th_u, ph_u + np.pi * (stag < 0)]))]
    starts += [(f"row{k}", lift(r["x"], "row", "site")) for k, r in enumerate(best_row[:3])]
    starts += list(warm)
    structured = len(starts)
    starts += [(f"rand{k}", np.concatenate([rng.normal(0, 0.3 if k % 2 else 0.8, N), rng.uniform(-np.pi, np.pi, N)]))
               for k in range(n_random)]
    tasks = [dict(algo=a, x0=x0, level="site", save_idx=save_idx, maxfev=B, seed=k, start=lab)
             for k, (lab, x0) in enumerate(starts) for a in (("nm", "powell") if k < structured else ("powell",))]
    tasks += [dict(algo=a, x0=starts[4][1], level="site", save_idx=save_idx, maxfev=budgets[2], seed=s, start="row0")
              for a in ("de", "da", "bh") for s in (0, 1)]
    runs += pool_map(run_one, tasks)
    return runs, dict(grid_J=Jg, grid_th=g_th, grid_ph=g_ph)


def save_runs(path, runs, save_idx, extra=None):
    """Runs sorted by J; x stored as site-level (theta_i, phi_i)."""
    runs = sorted(runs, key=lambda r: r["J"])
    xs = np.array([lift(r["x"], r["level"], "site") if r["level"] != "site" else r["x"] for r in runs])
    np.savez(path, algo=[r["algo"] for r in runs], start=[r["start"] for r in runs], level=[r["level"] for r in runs],
             J=[r["J"] for r in runs], x=xs, nfev=[r["nfev"] for r in runs], wall=[r["wall"] for r in runs],
             save_idx=save_idx, **(extra or {}))
    best = runs[0]
    print(f"saved {path}: best J={best['J']:.3f} ({best['level']}, {best['algo']}, {best['start']})")
    for lev in ("uni", "row", "site"):
        Js = sorted(r["J"] for r in runs if r["level"] == lev)
        print(f"  {lev}: best {Js[0]:.3f}, 3rd {Js[min(2, len(Js) - 1)]:.3f}, median {np.median(Js):.3f} ({len(Js)} runs)")


def summary(x, level="site", knots=None):
    """mu, mu_ph, J, w_hi at all 200 save times t > 0 on the full fine grid (validation)."""
    idx = np.arange(1, NSAVE)
    mu, mu_ph, w = evaluate(x, level, idx, knots, stride=1)
    J = J_of(mu, mu_ph)
    tt = idx * TMAX / (NSAVE - 1)
    above = np.nonzero(J > -10)[0]
    return dict(t=tt, mu=mu, mu_ph=mu_ph, J=J, w_hi=w, meanJ=J.mean(), maxJ=J.max(),
                t_prox=tt[above[0]] if len(above) else np.inf)


KNOTS = np.array([0, 1.5, 4, 9, 18, 30.0])  # spline knots of W(t), denser where J rises


def spline_runs(lam, fixed, x_row, snaps, budget=6000):
    """Step 2b: W(t) as cubic splines through KNOTS. Stage 1 row level (36 params, all algorithms), stage 2 site level
    (144 params, Powell / Nelder-Mead) from the best row splines, the fixed basis and a spline through the snapshot optima."""
    nk_ = len(KNOTS)
    const = lambda x: np.tile(x, nk_)
    base = dict(save_idx=T_TRAJ, knots=KNOTS, lam=lam)
    tasks = [dict(base, algo=a, x0=const(x_row), level="row", maxfev=budget, seed=s, start="fixed_row")
             for a, s in [("nm", 0), ("powell", 0), ("de", 0), ("da", 0), ("bh", 0), ("bh", 1)]]
    tasks += [dict(base, algo="powell", x0=const(np.zeros(6)), level="row", maxfev=budget, seed=0, start="identity")]
    rows = sorted(pool_map(run_one, tasks), key=lambda r: r["J"])
    # site level
    tk = np.array([t for t, _ in snaps])
    xs = np.array([x for _, x in snaps])
    snap_spline = np.array([xs[np.argmin(np.abs(tk - k))] if k > 0 else fixed[0] for k in KNOTS]).ravel()
    starts = [("fixed", const(fixed[0])), ("fixed2", const(fixed[1])), ("snaps", snap_spline)]
    for i, r in enumerate(rows[:2]):
        v = r["x"].reshape(nk_, -1)
        starts.append((f"row{i}", np.concatenate([lift(vv, "row", "site") for vv in v])))
    tasks = [dict(base, algo=a, x0=x0, level="site", maxfev=budget, seed=k, start=lab)
             for k, (lab, x0) in enumerate(starts) for a in ("nm", "powell")]
    return rows + pool_map(run_one, tasks)


# ----------------------------------------------------------------------------- RBM tests (step 3)
def basis_from(path):
    """'z' or a data/basis/*.npz -> site-level (x, 'site', knots) of its best run (knots None for a fixed basis)."""
    if path == "z":
        return np.zeros(2 * N), "site", None
    d = np.load(path)
    return (d["xk"], "site", d["knots"]) if "knots" in d else (d["x"][0], "site", None)


def rotated_spec(s, path, dt_seg=0.15):
    """Spec of rydberg_screen.spec in the rotated frame: psi0 -> W(0) psi0; H -> W H W^dag (fixed W) or piecewise-constant
    W(t_mid) H W(t_mid)^dag + i dW/dt W^dag on segments of length dt_seg (as one 'period' covering the whole run)."""
    x, level, knots = basis_from(path)
    (_, terms), = s["segs"]
    if knots is None:
        U = U_of(x, level)
        return dict(s, segs=[(1.0, rotate_terms(terms, U))], psi0=rotate(s["psi0"], U))
    T = s["t_save"][-1]
    n = int(round(T / dt_seg))
    tm = (np.arange(n) + 0.5) * T / n
    cs = CubicSpline(knots, np.asarray(x).reshape(len(knots), -1))
    segs = []
    for t in tm:
        U = U_of(x, level, knots, [t])[0]
        eps = 1e-5  # dW/dt by central difference of the spline
        dU = (U_of(cs(t + eps), level) - U_of(cs(t - eps), level)) / (2 * eps)
        segs.append((T / n, rotate_terms(terms, U) + frame_terms(U, dU)))
    return dict(s, segs=segs, psi0=rotate(s["psi0"], U_of(x, level, knots, [0.0])[0]), period=T)


def supervised(path, seed, alpha=1, chain_steps=100, kmax=NSAVE - 1, lr=0.3):
    """Step 3a: SR infidelity minimisation of an RBM against W psi(t), as a chain along the save times: each fit is
    warm-started from the previous one; the product start is exact through the visible biases, the seed sets the
    initial hidden weights (independent fits from a random RBM fail: its overlap with psi(t) is ~1e-4, see report)."""
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax, jax.numpy as jnp, netket as nk
    from jax.flatten_util import ravel_pytree
    from rydberg_rbm import model_for
    from quench2d import infidelity
    jax.config.update("jax_enable_x64", True)
    x, level, knots = basis_from(path)
    model = model_for(f"rbm{alpha}")
    S = jnp.asarray(nk.hilbert.Spin(0.5, N).all_states(), float)  # same order as psi (index 0 = all s = +1)
    p0 = model.init(jax.random.PRNGKey(seed), S[:1])["params"]
    flat0, unravel = ravel_pytree(p0)
    logpsi = lambda w: model.apply({"params": unravel(w)}, S)

    def jac(w):  # analytic RBM log-derivatives (jacfwd costs 40x more)
        p = unravel(w)
        th = jnp.tanh(S @ p["kernel"] + p["hidden_bias"])
        O = {"visible_bias": S.astype(complex), "hidden_bias": th, "kernel": S[:, :, None] * th[:, None, :]}
        return jax.vmap(lambda d: ravel_pytree(d)[0])(O)

    @jax.jit
    def run(w, phi, n):
        """n SR steps on I = 1 - |<phi|psi>|^2 / (<psi|psi><phi|phi>), with the gradient dI/dw* = -F (E[O* R]/E[R] - E[O*]),
        p = |psi|^2 / Z, R = phi / psi (holomorphic log psi); dense S + 5e-4 as in rydberg_rbm.py. lr = 0.3 x 100 steps
        matches lr = 0.01 x 1500 (the setting of rydberg_rbm.py) within a factor 2 in I at t <= 1.65, 10x faster."""
        def step(_, w):
            lp = logpsi(w)
            lp = lp - jnp.max(lp.real)
            psi = jnp.exp(lp)
            p = jnp.abs(psi) ** 2 / jnp.sum(jnp.abs(psi) ** 2)
            O = jac(w)
            Oc = O - p @ O
            ov = jnp.sum(p * phi / psi)  # E[R]
            F = jnp.abs(ov) ** 2 / jnp.sum(p * jnp.abs(phi / psi) ** 2)
            g = -F * ((p * phi / psi) @ Oc.conj() / ov)
            Smat = (Oc.conj().T * p) @ Oc
            return w - lr * jnp.linalg.solve(Smat + 5e-4 * jnp.eye(len(w)), g)
        return jax.lax.fori_loop(0, n, step, w)

    def target(k):
        t = k * TMAX / (NSAVE - 1)
        U = U_of(x, level, knots, [t])[0] if knots is not None else U_of(x, level)
        return rotate(PSI[k * SUB], U)

    psi0 = target(0)
    w = ravel_pytree({**p0, "visible_bias": jnp.asarray([0.5 * np.log(psi0[0] / psi0[1 << (N - 1 - i)]) for i in range(N)])})[0]
    state = lambda w: np.asarray(jnp.exp(logpsi(w)))
    inf = [infidelity(state(w), psi0)]
    t0 = time.time()
    for k in range(1, kmax + 1):
        phi = target(k)
        w = run(w, jnp.asarray(phi), chain_steps)
        inf.append(infidelity(state(w), phi))
        if k % 10 == 0:
            print(f"t={k * TMAX / (NSAVE - 1):.2f} inf={inf[-1]:.2e} wall={time.time() - t0:.0f}s", flush=True)
    return np.arange(kmax + 1) * TMAX / (NSAVE - 1), np.array(inf)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check(); sys.exit()
    os.makedirs(OUT, exist_ok=True)
    PSI = load_psi()
    if cmd == "snap":
        T = float(sys.argv[2])
        k = int(round(T / (TMAX / (NSAVE - 1))))
        runs, extra = hierarchy([k], name=f"t={T}")
        best = min(runs, key=lambda r: r["J"])
        val = evaluate(lift(best["x"], best["level"], "site") if best["level"] != "site" else best["x"], "site", [k], stride=1)
        save_runs(f"{OUT}/snap_t{T:g}.npz", runs, [k], dict(extra, val_mu=val[0], val_mu_ph=val[1], val_w_hi=val[2]))
    elif cmd == "traj":
        snaps = [(T, np.load(f"{OUT}/snap_t{T:g}.npz")["x"][0]) for T in (1.65, 4.95, 30)]
        runs, extra = hierarchy(T_TRAJ, warm=[(f"snap{T:g}", x) for T, x in snaps], n_random=16, name="traj")
        best = min(runs, key=lambda r: r["J"])
        xb = lift(best["x"], best["level"], "site") if best["level"] != "site" else best["x"]
        for lab, xx in [("z", np.zeros(2 * N)), ("opt", xb)]:
            extra.update({f"val_{lab}_{k}": v for k, v in summary(xx).items()})
        save_runs(f"{OUT}/traj_fixed.npz", runs, T_TRAJ, extra)
    elif cmd == "spline":
        lam = float(sys.argv[2])
        d = np.load(f"{OUT}/traj_fixed.npz")
        x_row = np.concatenate([d["x"][list(d["level"]).index("row")][:LY], d["x"][list(d["level"]).index("row")][N:N + LY]])
        snaps = [(T, np.load(f"{OUT}/snap_t{T:g}.npz")["x"][0]) for T in (1.65, 4.95, 30)]
        runs = sorted(spline_runs(lam, d["x"][:2], x_row, snaps), key=lambda r: r["J"])
        xk = np.array([np.concatenate([lift(v, r["level"], "site") if r["level"] != "site" else v
                                       for v in r["x"].reshape(len(KNOTS), -1)]) for r in runs])
        val = summary(xk[0], "site", KNOTS)
        np.savez(f"{OUT}/traj_spline_lam{lam:g}.npz", algo=[r["algo"] for r in runs], start=[r["start"] for r in runs],
                 level=[r["level"] for r in runs], J=[r["J"] for r in runs], nfev=[r["nfev"] for r in runs],
                 xk_all=xk, xk=xk[0], knots=KNOTS, lam=lam, frame=frame_cost(xk[0], "site", KNOTS),
                 **{f"val_{k}": v for k, v in val.items()})
        print(f"spline lam={lam}: best objective {runs[0]['J']:.3f} ({runs[0]['level']} {runs[0]['algo']} {runs[0]['start']}), "
              f"mean J {val['meanJ']:.3f}, max J {val['maxJ']:.3f}, t_prox {val['t_prox']}, frame {frame_cost(xk[0], 'site', KNOTS):.3g}")
    elif cmd == "sup":
        path, seed, alpha = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1
        name = "z" if path == "z" else os.path.basename(path)[:-4]
        t, inf = supervised(path, seed, alpha)
        np.savez(f"{OUT}/sup_{name}_s{seed}_a{alpha}.npz", t=t, inf=inf)
