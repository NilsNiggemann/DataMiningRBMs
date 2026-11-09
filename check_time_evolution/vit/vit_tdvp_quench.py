"""TDVP quench of the 1D TFI chain g_i -> g_f with an RBM or ViT ansatz (exact sums).
One npz per (L, arch, g_f) in data/vit_tdvp/. Run from this directory."""
import argparse, os, sys, functools
sys.path[:0] = ['..']
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp, optax, flax
import netket as nk, netket.experimental as nkx
from scipy.sparse.linalg import expm_multiply
import ed_definitions as ed


def infidelity(x, y):  # same as Netket/analysis.py:infidelity (that module needs h5py)
    x, y = np.asarray(x), np.asarray(y)
    return 1 - abs(np.vdot(y, x)) ** 2 / (np.vdot(x, x).real * np.vdot(y, y).real)

from vit1d import ViT1D, LADDER, n_real

p = argparse.ArgumentParser()
p.add_argument("--ansatz", default="vit", choices=["vit", "rbm"])
p.add_argument("--L", type=int, default=10)
p.add_argument("--rung", default=None, help="V0..V3 shortcut")
p.add_argument("--d", type=int, default=8); p.add_argument("--heads", type=int, default=2)
p.add_argument("--layers", type=int, default=1); p.add_argument("--b", type=int, default=2)
p.add_argument("--ffn", type=int, default=2)
p.add_argument("--gf", nargs="*", default=[], help="floats, or integer grid indices of linspace(0,3,40)")
p.add_argument("--tmax", type=float, default=1.0); p.add_argument("--dt", type=float, default=0.01)
p.add_argument("--ode", default="rk4", choices=["rk4", "rk45"])
p.add_argument("--rtol_smooth", type=float, default=1e-8)
p.add_argument("--fit_only", action="store_true")
p.add_argument("--lr", type=float, default=0.01); p.add_argument("--n_iter", type=int, default=1000); p.add_argument("--seed", type=int, default=1234)
p.add_argument("--out", default="../../data/vit_tdvp")
a = p.parse_args()
# ponytail: --fit_mc (sampled fit for L=20) not implemented; add at S13.

J, g_i, L = 1.0, 1.5, a.L
grid = np.linspace(0, 3, 40)
gfs = [grid[int(x)] if x.lstrip("-").isdigit() else float(x) for x in a.gf]
times = np.arange(0, a.tmax + a.dt / 2, a.dt)

if a.ansatz == "rbm":
    arch = "rbm"
    model = nk.models.RBM(alpha=1, param_dtype=np.complex128); hol = True
else:
    kw = LADDER[a.rung] if a.rung else dict(d=a.d, heads=a.heads, layers=a.layers, ffn=a.ffn)
    kw = dict(kw, b=a.b)
    arch = a.rung or "vit_d{d}h{heads}l{layers}f{ffn}".format(**kw)
    model = ViT1D(**kw); hol = False
os.makedirs(a.out, exist_ok=True)
tag = f"L{L}_{arch}"

_, hi, H_i = ed.build_TFI_hamiltonian(L, g_i, J, False)
_, psi0 = ed.ground_state(H_i)
vs = nk.vqs.FullSumState(hi, model, seed=a.seed)
n_par = n_real(model, L) if a.ansatz == "vit" else 2 * nk.jax.tree_size(vs.parameters)

# --- ground-state fit by infidelity SR (cached) ---
cache = f"{a.out}/init_{tag}_s{a.seed}.msgpack"
if os.path.exists(cache):
    vs.variables = flax.serialization.from_bytes(vs.variables, open(cache, "rb").read())
else:
    tgt = nk.vqs.FullSumState(hi, nk.models.LogStateVector(hi, param_dtype=jnp.complex128),
                              variables={"params": {"logstate": jnp.log(psi0.astype(complex) + 1e-300)}})
    op = nkx.observable.InfidelityOperator(target_state=tgt)
    sr = nk.optimizer.SR(diag_shift=5e-4, holomorphic=hol)
    opt = nk.optimizer.Sgd(a.lr); st = opt.init(vs.parameters)
    for i in range(a.n_iter):
        loss, gr = vs.expect_and_grad(op)
        u, st = opt.update(sr(vs, gr, i), st, vs.parameters)
        vs.parameters = optax.apply_updates(vs.parameters, u)
        if i % max(1, a.n_iter // 10) == 0: print(f"fit {i}: {loss.mean.real:.3e}", flush=True)
    open(cache, "wb").write(flax.serialization.to_bytes(vs.variables))
init_inf = infidelity(vs.to_array(), psi0)
print(f"{tag} N_p={n_par}  I(0)={init_inf:.3e}", flush=True)
if a.fit_only: sys.exit()

# --- TDVP per g_f ---
for g_f in gfs:
    _, _, H_f = ed.build_TFI_hamiltonian(L, g_f, J, False)
    if L <= 12:
        ev, V = ed.full_diagonalization(H_f); psi_ex = ed.evolve_state(ev, V, psi0, times)
    else:
        psi_ex = expm_multiply(-1j * H_f.to_sparse(), psi0, start=0, stop=a.tmax, num=len(times), endpoint=True)
    vst = nk.vqs.FullSumState(hi, model, variables=vs.variables)
    solver = nkx.dynamics.RK4(dt=a.dt) if a.ode == "rk4" else nkx.dynamics.RK45(dt=a.dt, adaptive=True)
    te = nkx.TDVP(H_f, vst, solver, qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=hol),
                  linear_solver=functools.partial(nk.optimizer.solver.pinv_smooth, rtol_smooth=a.rtol_smooth))
    inf = np.full(len(times), np.nan)
    f = f"{a.out}/{tag}_{a.ode}_dt{a.dt:g}_gf{g_f:.4f}.npz"
    for k, _ in enumerate(te.iter(T=a.tmax, tstops=times)):
        inf[k] = infidelity(vst.to_array(), psi_ex[k])
        np.savez(f, times=times, infidelity=inf, g_f=g_f, g_i=g_i, L=L, arch=arch, n_params=n_par,
                 initial_infidelity=init_inf, ode=a.ode, dt=a.dt, rtol_smooth=a.rtol_smooth, seed=a.seed)
    print(f"g_f={g_f:.3f} I(0)={inf[0]:.3e} I(t_max)={inf[-1]:.3e}", flush=True)
