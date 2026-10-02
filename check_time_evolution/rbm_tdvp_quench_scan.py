import os
import sys
import argparse

sys.path.append('.')
sys.path.append('../Netket/')

import numpy as np
import jax
import jax.numpy as jnp
import optax
import netket as nk
import netket.experimental as nkx

import ed_definitions as ed
from analysis import infidelity

p = argparse.ArgumentParser()
p.add_argument('--J2', type=float, default=0.0)
p.add_argument('--gz', type=float, default=0.0)
p.add_argument('--theta', type=float, default=0.0)
p.add_argument('--tag', type=str, required=True)
p.add_argument('--n_iter', type=int, default=1000)
p.add_argument('--ngf', type=int, default=20)
p.add_argument('--gfmin', type=float, default=0.1)
p.add_argument('--gfmax', type=float, default=3.0)
p.add_argument('--alpha', type=int, default=1)
p.add_argument('--learning_rate', type=float, default=0.01)
p.add_argument('--diag_shift', type=float, default=5e-4)
p.add_argument('--skip_tdvp', action='store_true')
args = p.parse_args()

# --- Parameters ---
L = 10
J = 1.0
g_i = 1.5
pbc = False

g_f_values = np.linspace(args.gfmin, args.gfmax, args.ngf)

t_max = 0.8
dt = 0.01
times = np.arange(0, t_max + dt / 2, dt)

alpha = args.alpha
seed = 1234

learning_rate = args.learning_rate
diag_shift = args.diag_shift
n_iter = args.n_iter
param_dtype = np.complex64

# --- Exact g_i ground state (infidelity-optimization target) ---
_, hi, H_i = ed.build_TFI_hamiltonian_2(L, g_i, J, args.J2, gz=args.gz, theta=args.theta, pbc=pbc)
E0, psi0_exact = ed.ground_state(H_i)
print(f"[{args.tag}] J2={args.J2} gz={args.gz} theta={args.theta:.3f}  "
      f"Exact g_i={g_i} ground state energy: {E0}")

# --- Infidelity-optimize an alpha=1 RBM against the exact state (exact summation) ---
target_model = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
vs_target = nk.vqs.FullSumState(
    hi, target_model,
    variables={"params": {"logstate": jnp.log(psi0_exact.astype(jnp.complex128))}},
)

model = nk.models.RBM(alpha=alpha, use_visible_bias=True, use_hidden_bias=True, param_dtype=param_dtype)
vs = nk.vqs.FullSumState(hi, model, seed=seed)

infid_op = nkx.observable.InfidelityOperator(target_state=vs_target)

sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=True)
optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
opt_state = optimizer.init(vs.parameters)

loss_history = np.full(n_iter, np.nan)
for i in range(n_iter):
    loss, grad = vs.expect_and_grad(infid_op)
    grad = sr(vs, grad, i)
    updates, opt_state = optimizer.update(grad, opt_state, vs.parameters)
    vs.parameters = optax.apply_updates(vs.parameters, updates)
    loss_history[i] = loss.mean.real
    if i % max(1, n_iter // 10) == 0:
        print(f"[{args.tag}] infidelity opt step {i}: loss={loss.mean.real:.3e}")

psi0_rbm = vs.to_array()
initial_infidelity = infidelity(psi0_rbm, psi0_exact)
# convergence check: relative change in loss over the last 10% of steps
tail = max(1, n_iter // 10)
rel_change = abs(loss_history[-1] - loss_history[-tail]) / max(loss_history[-tail], 1e-300)
print(f"[{args.tag}] RBM infidelity vs exact g_i ground state: {initial_infidelity}  "
      f"(relative change over last {tail} steps: {rel_change:.2%})")

DATA_DIR = '../data/cumulants_Fermi_dirac_TDVP'

if args.skip_tdvp:
    os.makedirs(DATA_DIR, exist_ok=True)
    outpath = f'{DATA_DIR}/rbm_opt_only_{args.tag}.npz'
    np.savez(outpath, loss_history=loss_history, initial_infidelity=initial_infidelity,
             alpha=alpha, learning_rate=learning_rate, diag_shift=diag_shift, n_iter=n_iter)
    print(f"[{args.tag}] Saved {outpath} (skip_tdvp)")
    sys.exit(0)

# --- TDVP scan over the coarse g_f_values, starting every run from the same optimized RBM ---
model_tdvp = nk.models.RBM(alpha=alpha, use_visible_bias=True, use_hidden_bias=True, param_dtype=np.complex128)
variables_c128 = jax.tree_util.tree_map(lambda x: x.astype(jnp.complex128), vs.variables)

dim = 2 ** L
psi_rbm_t = np.zeros((len(g_f_values), len(times), dim), dtype=complex)
infidelity_vs_exact = np.full((len(g_f_values), len(times)), np.nan)

for i, g_f in enumerate(g_f_values):
    _, _, H_f = ed.build_TFI_hamiltonian_2(L, g_f, J, args.J2, gz=args.gz, theta=args.theta, pbc=pbc)

    vs_t = nk.vqs.FullSumState(hi, model_tdvp, variables=variables_c128)

    evals_f, evecs_f = ed.full_diagonalization(H_f)
    psi_t_exact = ed.evolve_state(evals_f, evecs_f, psi0_exact, times)

    solver = nkx.dynamics.RK45(dt=dt, adaptive=True)
    te = nkx.TDVP(operator=H_f, variational_state=vs_t, ode_solver=solver,
                  qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True))

    for k, _ in enumerate(te.iter(T=t_max, tstops=times)):
        psi_rbm_t[i, k] = vs_t.to_array()
        infidelity_vs_exact[i, k] = infidelity(psi_rbm_t[i, k], psi_t_exact[k])

    print(f"[{args.tag}] g_f[{i}]={g_f:.3f}  infidelity(t=0)={infidelity_vs_exact[i, 0]:.3e}  "
          f"infidelity(t={t_max})={infidelity_vs_exact[i, -1]:.3e}")

# --- Save ---
os.makedirs(DATA_DIR, exist_ok=True)
outpath = f'{DATA_DIR}/rbm_tdvp_scan_{args.tag}.npz'
np.savez(
    outpath,
    g_f_values=g_f_values,
    times=times,
    psi_rbm_t=psi_rbm_t,
    infidelity_vs_exact=infidelity_vs_exact,
    initial_infidelity=initial_infidelity,
    loss_history=loss_history,
    L=L, J=J, g_i=g_i, pbc=pbc, alpha=alpha, J2=args.J2, gz=args.gz, theta=args.theta,
    learning_rate=learning_rate, diag_shift=diag_shift, n_iter=n_iter,
)
print(f"[{args.tag}] Saved {outpath}")
