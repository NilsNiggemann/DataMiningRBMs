"""RBM TDVP for the quench g_i=1.5 -> g_f=0.5 (L=10, exact summation), saving the RBM states.

Same protocol as ../rbm_tdvp_quench.py (infidelity-SR fit of the g_i ground state, then adaptive RK45 TDVP).
Per alpha, writes data/rbm_states_a{alpha}.npz with
  times     fine grid (spacing DT_SAVE) on which the states are stored
  psi       RBM amplitudes psi_RBM(s, t) in NetKet ordering (= phase_branch ordering reversed, s -> -s)
  logpsi    the network's own log psi = a.s + sum_j log cosh(theta_j(s)) (NetKet's branch)
  inf       infidelity w.r.t. the exact state at each stored time
Run: python rbm_states.py [alpha ...] [--rk4]  (one process per alpha, each pinned to CPUS_PER_RUN cores;
     --rk4: fixed-step RK4 at dt=DT_SAVE/4, for alpha=3 where adaptive RK45 stalled near t=0)
"""
import os
import sys
from multiprocessing import Process

ALPHAS = [0.5, 1.0, 2.0, 3.0]
CPUS_PER_RUN = 6  # 4 x 6 = 24 <= 30 cores (shared machine)
DT_SAVE = 1e-3
T_MAX = 0.99


def polish(model, variables, states, psi0, maxiter=5000):
    """L-BFGS on -log F of the initial fit (complex128), after SR has found the basin."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.flatten_util import ravel_pytree
    from scipy.optimize import minimize
    jax.config.update("jax_enable_x64", True)
    flat, unravel = ravel_pytree(variables)
    n = flat.size
    target = jnp.asarray(psi0 / np.linalg.norm(psi0))

    @jax.jit
    def loss(x):
        lp = model.apply(unravel(x[:n] + 1j * x[n:]), states)
        phi = jnp.exp(lp - lp.real.max())
        return -jnp.log(jnp.abs(jnp.vdot(target, phi)) ** 2 / jnp.vdot(phi, phi).real)

    vg = jax.jit(jax.value_and_grad(loss))
    x0 = np.concatenate([flat.real, flat.imag])
    res = minimize(lambda x: [np.asarray(v, float) for v in vg(x)], x0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, ftol=1e-16, gtol=1e-12))
    return unravel(jnp.asarray(res.x[:n] + 1j * res.x[n:]))


def run(alpha, cpus, rk4=False):
    os.sched_setaffinity(0, cpus)
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    import netket as nk
    import netket.experimental as nkx
    import ed_definitions as ed

    def infidelity(a, b):
        return 1 - abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)

    L, J, g_i, g_f, seed = 10, 1.0, 1.5, 0.5, 1234
    _, hi, H_i = ed.build_TFI_hamiltonian(L, g_i, J, False)
    _, psi0 = ed.ground_state(H_i)

    # infidelity fit of the initial state (as in rbm_tdvp_quench.py)
    target = nk.vqs.FullSumState(hi, nk.models.LogStateVector(hi, param_dtype=jnp.complex128),
                                 variables={"params": {"logstate": jnp.log(psi0.astype(jnp.complex128))}})
    vs = nk.vqs.FullSumState(hi, nk.models.RBM(alpha=alpha, param_dtype=np.complex64), seed=seed)
    infid_op = nkx.observable.InfidelityOperator(target_state=target)
    sr = nk.optimizer.SR(diag_shift=5e-4, holomorphic=True)
    opt = nk.optimizer.Sgd(learning_rate=0.01)
    opt_state = opt.init(vs.parameters)
    for i in range(1000):
        _, grad = vs.expect_and_grad(infid_op)
        updates, opt_state = opt.update(sr(vs, grad, i), opt_state, vs.parameters)
        vs.parameters = optax.apply_updates(vs.parameters, updates)

    model = nk.models.RBM(alpha=alpha, param_dtype=np.complex128)
    variables = jax.tree_util.tree_map(lambda x: x.astype(jnp.complex128), vs.variables)
    variables = polish(model, variables, hi.all_states(), psi0)
    vs_t = nk.vqs.FullSumState(hi, model, variables=variables)
    print(f"alpha={alpha} initial infidelity {infidelity(vs_t.to_array(), psi0):.2e}", flush=True)

    _, _, H_f = ed.build_TFI_hamiltonian(L, g_f, J, False)
    evals, evecs = ed.full_diagonalization(H_f)
    times = np.round(np.arange(0, T_MAX + DT_SAVE / 2, DT_SAVE), 10)
    psi_exact = ed.evolve_state(evals, evecs, psi0, times)
    states = hi.all_states()

    te = nkx.TDVP(operator=H_f, variational_state=vs_t, ode_solver=nkx.dynamics.RK4(dt=DT_SAVE / 4) if rk4 else nkx.dynamics.RK45(dt=0.01, adaptive=True),
                  qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True))
    psi = np.zeros((len(times), hi.n_states), complex)
    logpsi = np.zeros_like(psi)
    inf = np.zeros(len(times))
    for k, _ in enumerate(te.iter(T=T_MAX, tstops=times)):
        lp = np.asarray(model.apply(vs_t.variables, states))
        logpsi[k] = lp
        psi[k] = vs_t.to_array()
        inf[k] = infidelity(psi[k], psi_exact[k])
        if k % 100 == 0:
            print(f"alpha={alpha} t={times[k]:.2f} inf={inf[k]:.2e}", flush=True)
    np.savez(f"data/rbm_states_a{alpha:g}.npz", times=times, psi=psi, logpsi=logpsi, inf=inf, alpha=alpha)
    print(f"done alpha={alpha}", flush=True)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    alphas = [float(a) for a in sys.argv[1:] if a != "--rk4"] or ALPHAS
    procs = [Process(target=run, args=(a, set(range(i * CPUS_PER_RUN, (i + 1) * CPUS_PER_RUN)), "--rk4" in sys.argv))
             for i, a in enumerate(alphas)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
