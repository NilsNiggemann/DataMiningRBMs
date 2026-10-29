"""Full-sum RBM / Jastrow + TDVP vs ED for one screened quench (PLAN_rydberg_quench.md stages 2-3).

usage: python rydberg_rbm.py ID LAT INIT G ANSATZ      ANSATZ in rbm1, rbm2, rbm4, jas, rbmjas1
       -> data/rbm/<ID>_<LAT>_<INIT>_<G>_<ANSATZ>.npz
Piecewise-constant (Floquet) protocols run one TDVP driver per segment, with the same solver settings everywhere.
"""
import ctypes, functools, os, sys, time
from types import SimpleNamespace
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import flax, flax.linen as nn, jax, jax.numpy as jnp, numpy as np, optax, netket as nk, netket.experimental as nkx
from rydberg_screen import spec, sparse_parts, evolve, nk_operator
from quench2d import infidelity

jax.config.update("jax_enable_x64", True)


def _eager_cond(pred, true_fn, false_fn, *ops):
    """lax.cond that runs the branch directly when the predicate is concrete (outside jit)."""
    if isinstance(pred, jax.core.Tracer):
        return jax.lax.cond(pred, true_fn, false_fn, *ops)
    return true_fn(*ops) if bool(pred) else false_fn(*ops)


# NetKet's adaptive RK step (netket/experimental/dynamics/_integrator.py:341, the module's only use of `jax`, and
# set_flag_jax in _utils.py:61) calls jax.lax.cond eagerly with fresh lambdas: every step compiles and caches a new
# executable (~45 MB/min per run)
nkx.dynamics._integrator.jax = SimpleNamespace(lax=SimpleNamespace(cond=_eager_cond))
nkx.dynamics._integrator.set_flag_jax = lambda cond, flags, flag: _eager_cond(cond, lambda x: x | flag, lambda x: x, flags)
STOP = 0.5  # infidelity at which the run is stopped (the RBM has lost the state; later times stay NaN)


def clear_if_big():
    """NetKet's adaptive RK step (netket/experimental/dynamics/_integrator.py:341) calls jax.lax.cond eagerly with fresh
    lambdas, so every step compiles and caches a new executable: ~45 MB/min per run. Dropping the caches bounds it."""
    if int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") > max_rss:
        jax.clear_caches()
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # hand the freed executables back to the OS


def advance(te, target):
    """Integrate exactly to `target`. Not te.iter/te.run: iter yields as soon as np.isclose(t, tstop) (rtol 1e-5, i.e.
    early), never yields a stop at the interval end, and run() can overshoot T."""
    while te.t < target - 1e-12:
        te._integrator.step(max_dt=target - te.t)
        if te._integrator.errors:
            raise RuntimeError(f"ODE integrator: {te._integrator.errors.message()}")


class Ansatz(nn.Module):
    """log psi = a.s [+ sum_j log cosh(W s + b)_j] [+ s.J.s (i<j)]: RBM, Jastrow or RBM x Jastrow."""
    alpha: int = 1
    jastrow: bool = False

    @nn.compact
    def __call__(self, x):
        N = x.shape[-1]
        out = x @ self.param("visible_bias", nn.initializers.zeros, (N,), complex)
        if self.alpha:
            W = self.param("kernel", nn.initializers.normal(1e-3), (N, self.alpha * N), complex)
            b = self.param("hidden_bias", nn.initializers.zeros, (self.alpha * N,), complex)
            out = out + jnp.sum(nk.nn.log_cosh(x @ W + b), axis=-1)
        if self.jastrow:
            J = self.param("J", nn.initializers.zeros, (N, N), complex) * np.triu(np.ones((N, N)), 1)  # numpy mask: jnp.triu fails under sharding
            out = out + jnp.einsum("...i,ij,...j->...", x, J, x)
        return out


def model_for(name):
    return {"rbm1": Ansatz(1), "rbm2": Ansatz(2), "rbm4": Ansatz(4), "jas": Ansatz(0, True), "rbmjas1": Ansatz(1, True)}[name]


def is_product(psi, N):
    m = np.abs(psi).reshape([2] * N)
    return np.allclose(np.abs(np.linalg.svd(m.reshape(2, -1), compute_uv=False))[1:], 0, atol=1e-12)


if __name__ == "__main__":
    ID, lat, init, g, ansatz = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
    out = f"data/rbm/{ID}_{lat}_{init}_{g:+.4f}_{ansatz}.npz"
    os.makedirs("data/rbm", exist_ok=True)
    t_start = time.time()
    s = spec(ID, lat, init, g)
    if os.environ.get("BASIS"):  # rotated local frame (basis_opt.py): 'z' or a data/basis/*.npz
        from basis_opt import rotated_spec
        b = os.environ["BASIS"]
        s = rotated_spec(s, b)
        out = f"{out[:-4]}_{'z' if b == 'z' else os.path.basename(b)[:-4]}.npz"
    N, ts = s["N"], s["t_save"]
    ts = ts[ts <= float(os.environ.get("TMAX", np.inf)) + 1e-9]  # optional shorter window (used for the N = 16 repeats)
    max_rss = 1e9 if N <= 12 else 2e9  # bytes, see clear_if_big; keeps 26 parallel runs under the 40 GB budget
    parts = [(d, *sparse_parts(terms, N)) for d, terms in s["segs"]]
    rowsum = max(np.abs(o).sum(1).max() if o.nnz else 0.0 for _, _, o in parts)
    _, ev, _ = evolve(s["psi0"], parts, ts, min(np.diff(ts).min(), 0.1 / rowsum) if rowsum else np.diff(ts).min(), s["period"])
    psi_ex = np.exp(ev["logabs"] + 1j * ev["ph"])

    hi = nk.hilbert.Spin(0.5, N)
    vs = nk.vqs.FullSumState(hi, model_for(ansatz), seed=1234)
    psi0 = s["psi0"]
    if init != "gs":
        # product start: exact visible biases a_i = 1/2 log(psi(all up) / psi(site i flipped)); hidden/Jastrow ~ 0
        a = np.array([0.5 * np.log(psi0[0] / psi0[1 << (N - 1 - i)]) for i in range(N)])
        p = flax.core.unfreeze(vs.parameters)
        p["visible_bias"] = jnp.asarray(a)
        vs.parameters = p
    else:  # ground-state start: SR infidelity minimisation, cached per (ansatz, initial state)
        cache = f"data/rbm/init_{ID}_{lat}_{ansatz}.msgpack"
        if os.path.exists(cache):
            with open(cache, "rb") as f:
                vs.variables = flax.serialization.from_bytes(vs.variables, f.read())
        else:
            tgt = nk.vqs.FullSumState(hi, nk.models.LogStateVector(hi, param_dtype=jnp.complex128),
                                      variables={"params": {"logstate": jnp.log(psi0.astype(complex) + 1e-300)}})
            op = nkx.observable.InfidelityOperator(target_state=tgt)
            sr = nk.optimizer.SR(diag_shift=5e-4, holomorphic=True)
            opt = nk.optimizer.Sgd(0.01)
            st = opt.init(vs.parameters)
            for i in range(1500):
                _, gr = vs.expect_and_grad(op)
                u, st = opt.update(sr(vs, gr, i), st, vs.parameters)
                vs.parameters = optax.apply_updates(vs.parameters, u)
            with open(cache, "wb") as f:
                f.write(flax.serialization.to_bytes(vs.variables))
    init_inf = infidelity(vs.to_array(), psi_ex[0])
    print(f"{out}: initial infidelity {init_inf:.2e}", flush=True)

    def save():
        np.savez(out, t=ts, inf=inf, init_inf=init_inf, ID=ID, lat=lat, init=init, g=g, ansatz=ansatz,
                 walltime=time.time() - t_start)

    ops = [nk_operator(hi, terms) for _, terms in s["segs"]]
    bounds = np.cumsum([0] + [d for d, _ in s["segs"]])
    inf = np.full(len(ts), np.nan)
    inf[0] = init_inf
    t, k = 0.0, 1
    T_end = ts[-1]
    while t < T_end - 1e-9 and inf[k - 1] < STOP:
        if s["period"]:
            tau = t % s["period"]
            si = min(int(np.searchsorted(bounds, tau + 1e-9, side="right") - 1), len(ops) - 1)
            t_seg_end = min(t - tau + bounds[si + 1], T_end)
        else:
            si, t_seg_end = 0, T_end
        # rtol_smooth=1e-8 and the QGT error norm (Schmitt & Heyl 2020): with NetKet's defaults the adaptive step collapses
        # near product states, where the hidden units are almost decoupled and the QGT is nearly singular
        te = nkx.TDVP(operator=ops[si], variational_state=vs, ode_solver=nkx.dynamics.RK45(dt=float(os.environ.get("DT0", 0.01)), adaptive=True), t0=t,
                      qgt=nk.optimizer.qgt.QGTJacobianDense(holomorphic=True), error_norm="qgt",
                      linear_solver=functools.partial(nk.optimizer.solver.pinv_smooth, rtol_smooth=1e-8))
        for target in [x for x in ts[k:] if x <= t_seg_end + 1e-9]:
            advance(te, target)
            inf[k] = infidelity(vs.to_array(), psi_ex[k]); k += 1
            print(f"t={ts[k - 1]:.2f} inf={inf[k - 1]:.2e} wall={time.time() - t_start:.0f}s", flush=True)
            save()  # partial results survive a wall-clock limit
            clear_if_big()
            if not inf[k - 1] < STOP:
                break
        if inf[k - 1] < STOP:
            advance(te, t_seg_end)
        t = t_seg_end

    save()
    print(f"{out}: done in {time.time() - t_start:.0f}s; final inf {inf[-1]:.2e}", flush=True)
