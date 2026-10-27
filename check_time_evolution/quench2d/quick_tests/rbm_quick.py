"""Quick RBM(alpha=1)+full-sum TDVP vs ED on 4x3, for a handful of quenches. usage: rbm_quick.py FAM VAL TMAX"""
import sys, numpy as np, jax, jax.numpy as jnp, optax, netket as nk, netket.experimental as nkx
from netket.operator.spin import sigmax, sigmay, sigmaz
from qt import *
fam, val, TM = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
Lx, Ly = 4, 3; N = Lx*Ly; nn, nn2 = lattice(Lx, Ly, True, False)
kw_i = dict(J=-1, hx=-4.0)
kw_f = dict(tfi=dict(J=-1, hx=-val), xxz=dict(J=-1, Jxx=val, hx=-1.0))[fam]
hi = nk.hilbert.Spin(0.5, N)
def nk_ham(J=0, Jxx=0, hx=0):
    H = sum(hx*sigmax(hi, i) for i in range(N))
    for a, b in nn:
        H = H + J*sigmaz(hi, a)*sigmaz(hi, b)
        if Jxx: H = H + Jxx*(sigmax(hi, a)*sigmax(hi, b) + sigmay(hi, a)*sigmay(hi, b))
    return H.to_local_operator() if hasattr(H, "to_local_operator") else H
Hf_nk = nk_ham(**kw_f)
Hf = hamiltonian(N, nn, **kw_f)
assert np.allclose(Hf_nk.to_dense(), Hf), 'basis/convention mismatch'
psi0 = ground_state(hamiltonian(N, nn, **kw_i))
times = np.round(np.arange(0, TM + 1e-9, 0.02), 10)
psi_ex = evolve(Hf, psi0, times)
inf = lambda a, b: 1 - abs(np.vdot(a, b))**2 / (np.vdot(a, a).real * np.vdot(b, b).real)

# initial RBM: infidelity optimization against exact psi0 (same recipe as rbm_tdvp_quench.py)
tgt = nk.vqs.FullSumState(hi, nk.models.LogStateVector(hi, param_dtype=jnp.complex128),
                          variables={"params": {"logstate": jnp.log(psi0.astype(complex))}})
model = nk.models.RBM(alpha=1, param_dtype=complex)
vs = nk.vqs.FullSumState(hi, model, seed=1234)
op = nkx.observable.InfidelityOperator(target_state=tgt)
sr = nk.optimizer.SR(diag_shift=5e-4, holomorphic=True); opt = nk.optimizer.Sgd(0.01); st = opt.init(vs.parameters)
for i in range(1000):
    _, g = vs.expect_and_grad(op); g = sr(vs, g, i)
    u, st = opt.update(g, st, vs.parameters); vs.parameters = optax.apply_updates(vs.parameters, u)
print(f'{fam} {val}: initial infidelity {inf(vs.to_array(), psi0):.2e}', flush=True)

te = nkx.TDVP(operator=Hf_nk, variational_state=vs, ode_solver=nkx.dynamics.RK45(dt=0.01, adaptive=True),
              qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True))
I = []
for k, _ in enumerate(te.iter(T=TM, tstops=times)):
    I.append(inf(vs.to_array(), psi_ex[k]))
I = np.array(I)
r = analyze(psi_ex, N, times)
np.savez(f'rbm_{fam}_{val}.npz', times=times, inf=I, **r)
print(f'{fam} {val}: done; inf(t) at t=0.1,0.2,..:', ' '.join(f'{x:.1e}' for x in I[5::5]), flush=True)
