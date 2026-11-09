"""Full-sum projected tVMC (p-tVMC, Gravina-Savona-Vicentini 2025) quench of the 1D TFI chain.
Each step: target = exp(-i H_f dt) psi_theta (exact, dense/sparse), then minimise the infidelity
to it with SR from the previous parameters. Needs the cached t=0 fit from vit_tdvp_quench.py.
Run from this directory."""
import argparse, os, sys
sys.path[:0] = ['..']
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp, optax, flax
import netket as nk, netket.experimental as nkx
from scipy.sparse.linalg import expm_multiply
import ed_definitions as ed
from vit1d import ViT1D, LADDER, n_real

p = argparse.ArgumentParser()
p.add_argument("--ansatz", default="vit", choices=["vit", "rbm"]); p.add_argument("--L", type=int, default=10)
p.add_argument("--rung", default="V2"); p.add_argument("--gf", nargs="+", type=float, required=True)
p.add_argument("--tmax", type=float, default=1.0); p.add_argument("--dt", type=float, default=0.025)
p.add_argument("--lr", type=float, default=0.5); p.add_argument("--diag_shift", type=float, default=1e-4)
p.add_argument("--max_iter", type=int, default=100); p.add_argument("--tol", type=float, default=1e-9)
p.add_argument("--seed", type=int, default=1234); p.add_argument("--out", default="../../data/vit_tdvp")
a = p.parse_args()

J, g_i, L = 1.0, 1.5, a.L
if a.ansatz == "rbm": model, hol, arch = nk.models.RBM(alpha=1, param_dtype=np.complex128), True, "rbm"
else: model, hol, arch = ViT1D(**dict(LADDER[a.rung], b=2)), False, a.rung
_, hi, H_i = ed.build_TFI_hamiltonian(L, g_i, J, False); _, psi0 = ed.ground_state(H_i)
vs0 = nk.vqs.FullSumState(hi, model, seed=a.seed)
vs0.variables = flax.serialization.from_bytes(vs0.variables, open(f"{a.out}/init_L{L}_{arch}_s{a.seed}.msgpack", "rb").read())
n_par = n_real(model, L) if a.ansatz == "vit" else 2 * nk.jax.tree_size(vs0.parameters)
infid = lambda x, y: 1 - abs(np.vdot(y, x)) ** 2 / (np.vdot(x, x).real * np.vdot(y, y).real)
nsteps = int(round(a.tmax / a.dt)); times = a.dt * np.arange(nsteps + 1)

for g_f in a.gf:
    _, _, H_f = ed.build_TFI_hamiltonian(L, g_f, J, False)
    if L <= 12:
        ev, V = ed.full_diagonalization(H_f); U = (V * np.exp(-1j * ev * a.dt)) @ V.conj().T; step = lambda v: U @ v
    else:
        Hs = H_f.to_sparse(); step = lambda v: expm_multiply(-1j * a.dt * Hs, v)
    vs = nk.vqs.FullSumState(hi, model, variables=vs0.variables)
    ex = psi0.astype(complex); inf = np.full(nsteps + 1, np.nan); opt_inf = np.full(nsteps + 1, np.nan); nit = np.zeros(nsteps + 1); hist = np.full((nsteps + 1, a.max_iter), np.nan)  # loss per inner iteration
    inf[0] = infid(vs.to_array(), ex); f = f"{a.out}/L{L}_{arch}_ptvmc_dt{a.dt:g}_gf{g_f:.4f}.npz"
    for k in range(1, nsteps + 1):
        ex = step(ex); tgt_vec = step(vs.to_array())
        tgt = nk.vqs.FullSumState(hi, nk.models.LogStateVector(hi, param_dtype=jnp.complex128),
                                  variables={"params": {"logstate": jnp.log(tgt_vec.conj().astype(complex) + 1e-300)}})  # conj: InfidelityOperator matches the conjugate target
        op = nkx.observable.InfidelityOperator(target_state=tgt)
        lr, ds = a.lr, a.diag_shift; best, bp = np.inf, vs.parameters  # backtracking: reject steps that raise the loss
        for i in range(a.max_iter):
            loss, gr = vs.expect_and_grad(op); l = loss.mean.real
            if not np.isfinite(l) or l > best:
                vs.parameters = bp; lr, ds = lr / 2, ds * 2; loss, gr = vs.expect_and_grad(op); l = loss.mean.real
            best, bp = l, vs.parameters; hist[k, i] = l
            if l < a.tol: break
            u = nk.optimizer.SR(diag_shift=ds, holomorphic=hol)(vs, gr, i)
            vs.parameters = optax.apply_updates(vs.parameters, jax.tree_util.tree_map(lambda x: -lr * x, u))
        opt_inf[k], nit[k], inf[k] = best, i, infid(vs.to_array(), ex)
        np.savez(f, times=times, infidelity=inf, opt_infidelity=opt_inf, n_iter=nit, g_f=g_f, g_i=g_i, L=L, arch=arch,
                 n_params=n_par, dt=a.dt, lr=a.lr, diag_shift=a.diag_shift, seed=a.seed, method="ptvmc", loss_hist=hist)
        print(f"g_f={g_f} t={times[k]:.3f} I={inf[k]:.3e} opt={opt_inf[k]:.2e} iters={i}", flush=True)
