"""RBM + full-sum TDVP vs ED for one 4x3 quench (PLAN_quench_family.md step 4).

usage: python rbm_run.py FAM VAL [ALPHA=1] [TMAX=1.5]  ->  data/rbm/rbm_{FAM}_{VAL:+.3f}_a{ALPHA}.npz
"""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import flax, numpy as np, jax.numpy as jnp, optax, netket as nk, netket.experimental as nkx
from quench2d import *

fam, val = sys.argv[1], float(sys.argv[2])
alpha = int(sys.argv[3]) if len(sys.argv) > 3 else 1
TM = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
out = f"data/rbm/rbm_{fam}_{val:+.3f}_a{alpha}.npz"
t_start = time.time()

N = LX * LY
nn, _ = lattice(LX, LY, True, False)
hi = nk.hilbert.Spin(0.5, N)
kw_f = family_kwargs(fam, val)
Hf_nk = nk_ham(hi, nn, **kw_f)
psi0 = ground_state(hamiltonian(N, nn, **initial_kwargs(4.0)))
times = np.round(np.arange(0, TM + 1e-9, 0.01), 10)
psi_ex = evolve(hamiltonian(N, nn, **kw_f), psi0, times)

# initial RBM: infidelity minimisation against exact psi0 (recipe of rbm_tdvp_quench.py).
# psi0 is the same for every quench, so the result is cached per alpha.
vs = nk.vqs.FullSumState(hi, nk.models.RBM(alpha=alpha, param_dtype=complex), seed=1234)
init_file = f"data/rbm/init_a{alpha}.msgpack"
if os.path.exists(init_file):
    with open(init_file, "rb") as f:
        vs.variables = flax.serialization.from_bytes(vs.variables, f.read())
else:
    tgt = nk.vqs.FullSumState(hi, nk.models.LogStateVector(hi, param_dtype=jnp.complex128),
                              variables={"params": {"logstate": jnp.log(psi0.astype(complex))}})
    op = nkx.observable.InfidelityOperator(target_state=tgt)
    sr = nk.optimizer.SR(diag_shift=5e-4, holomorphic=True)
    opt = nk.optimizer.Sgd(0.01)
    st = opt.init(vs.parameters)
    for i in range(1000):
        _, g = vs.expect_and_grad(op)
        g = sr(vs, g, i)
        u, st = opt.update(g, st, vs.parameters)
        vs.parameters = optax.apply_updates(vs.parameters, u)
    with open(init_file, "wb") as f:
        f.write(flax.serialization.to_bytes(vs.variables))
init_inf = infidelity(vs.to_array(), psi0)
print(f"{out}: initial infidelity {init_inf:.2e} after {time.time() - t_start:.0f}s", flush=True)

# fixed solver settings across all runs so that comparisons over p are fair
te = nkx.TDVP(operator=Hf_nk, variational_state=vs, ode_solver=nkx.dynamics.RK45(dt=0.01, adaptive=True),
              qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True))
inf = np.full(len(times), np.nan)
for k, _ in enumerate(te.iter(T=TM, tstops=times)):
    inf[k] = infidelity(vs.to_array(), psi_ex[k])
    if not np.isfinite(inf[k]):
        break

np.savez(out, times=times, inf=inf, init_inf=init_inf, fam=fam, val=val, alpha=alpha,
         walltime=time.time() - t_start)
print(f"{out}: done in {time.time() - t_start:.0f}s; inf(t=0.5,1,1.5) =",
      " ".join(f"{x:.1e}" for x in inf[50::50]), flush=True)
