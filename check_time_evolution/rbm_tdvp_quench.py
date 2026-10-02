import os
import sys

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

# --- Parameters (modifiable) ---
L = 10
J = 1.0
g_i = 1.5
pbc = False

g_f_values = np.linspace(0, 3, 40)

t_max = 1.0
dt = 0.01
times = np.arange(0, t_max + dt / 2, dt)

alpha = 1
seed = 1234

learning_rate = 0.01
diag_shift = 5e-4
n_iter = 1000
param_dtype = np.complex64

# --- Exact g_i ground state (infidelity-optimization target) ---
_, hi, H_i = ed.build_TFI_hamiltonian(L, g_i, J, pbc)
E0, psi0_exact = ed.ground_state(H_i)
print(f"Exact g_i={g_i} ground state energy: {E0}")

# --- Infidelity-optimize an alpha=1 RBM against the exact state (exact summation) ---
target_model = nk.models.LogStateVector(hi, param_dtype=jnp.complex128)
vs_target = nk.vqs.FullSumState(
    hi, target_model,
    variables={"params": {"logstate": jnp.log(psi0_exact.astype(jnp.complex128))}},
)

model = nk.models.RBM(alpha=alpha, use_visible_bias=True, use_hidden_bias=True, param_dtype=param_dtype)
vs = nk.vqs.FullSumState(hi, model, seed=seed)

infid_op = nkx.observable.InfidelityOperator(target_state=vs_target)

# nk.VMC / nkx.driver.Infidelity_SR both reject FullSumState (the latter raises
# "NGD drivers do not support FullSumState"), and nk.VMC requires an
# AbstractOperator with .collect(), which InfidelityOperator (an
# AbstractObservable) lacks. So we replicate nk.VMC's own update rule
# manually: expect_and_grad + SR preconditioning + an optax optimizer.
sr = nk.optimizer.SR(diag_shift=diag_shift, holomorphic=True)
optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
opt_state = optimizer.init(vs.parameters)

for i in range(n_iter):
    loss, grad = vs.expect_and_grad(infid_op)
    grad = sr(vs, grad, i)
    updates, opt_state = optimizer.update(grad, opt_state, vs.parameters)
    vs.parameters = optax.apply_updates(vs.parameters, updates)
    if i % max(1, n_iter // 10) == 0:
        print(f"infidelity opt step {i}: loss={loss.mean.real:.3e}")

psi0_rbm = vs.to_array()
initial_infidelity = infidelity(psi0_rbm, psi0_exact)
print(f"RBM infidelity vs exact g_i ground state: {initial_infidelity}")

# --- TDVP sweep over g_f_values, starting every run from the same optimized RBM ---
# The adaptive RK45 step-size control cannot converge in complex64 precision
# (its error estimate never drops below the float32 noise floor, so it stalls
# shrinking the step forever) -- upcast the optimized parameters to complex128
# for the TDVP phase. The infidelity optimization above still runs in
# complex64 as specified.
model_tdvp = nk.models.RBM(alpha=alpha, use_visible_bias=True, use_hidden_bias=True, param_dtype=np.complex128)
variables_c128 = jax.tree_util.tree_map(lambda x: x.astype(jnp.complex128), vs.variables)

dim = 2 ** L
psi_rbm_t = np.zeros((len(g_f_values), len(times), dim), dtype=complex)
infidelity_vs_exact = np.full((len(g_f_values), len(times)), np.nan)

for i, g_f in enumerate(g_f_values):
    _, _, H_f = ed.build_TFI_hamiltonian(L, g_f, J, pbc)

    vs_t = nk.vqs.FullSumState(hi, model_tdvp, variables=variables_c128)

    evals_f, evecs_f = ed.full_diagonalization(H_f)
    psi_t_exact = ed.evolve_state(evals_f, evecs_f, psi0_exact, times)

    solver = nkx.dynamics.RK45(dt=dt, adaptive=True)
    te = nkx.TDVP(operator=H_f, variational_state=vs_t, ode_solver=solver,
                  qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True))

    for k, _ in enumerate(te.iter(T=t_max, tstops=times)):
        psi_rbm_t[i, k] = vs_t.to_array()
        infidelity_vs_exact[i, k] = infidelity(psi_rbm_t[i, k], psi_t_exact[k])

    print(f"g_f[{i}]={g_f:.3f}  infidelity(t=0)={infidelity_vs_exact[i, 0]:.3e}  "
          f"infidelity(t={t_max})={infidelity_vs_exact[i, -1]:.3e}")

# --- Save ---
DATA_DIR = '../data/cumulants_Fermi_dirac_TDVP'
os.makedirs(DATA_DIR, exist_ok=True)
np.savez(
    f'{DATA_DIR}/rbm_tdvp_quench.npz',
    g_f_values=g_f_values,
    times=times,
    psi_rbm_t=psi_rbm_t,
    infidelity_vs_exact=infidelity_vs_exact,
    initial_infidelity=initial_infidelity,
    L=L, J=J, g_i=g_i, pbc=pbc, alpha=alpha,
    learning_rate=learning_rate, diag_shift=diag_shift, n_iter=n_iter,
)
print(f"Saved {DATA_DIR}/rbm_tdvp_quench.npz")
