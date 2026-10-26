"""Basis optimized for the truncated cluster-expansion (CE) infidelity, plans/02_ce_basis.md.

I_CE(phi; K) = 1 - |<phi|phi_K>|^2 (normalized), phi_K = exp(FWHT^-1 of the K largest |c_r| of
c = FWHT(l)), l = log max(|phi|, eps) + i arg(phi) on the principal branch, after the gauge
phi_{argmax|phi|} = 1. Bit convention as in walshness.py. Kept separate so walshness.py does not
import JAX. Run `python ce_basis.py` for the self-tests T12-T16.
"""
import os
import time
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import walshness as wn

EPS = 1e-12


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _fwht(x, n):
    """Unnormalized Walsh-Hadamard transform (involution up to 2^n)."""
    for j in range(n):
        v = x.reshape(2**j, 2, -1)
        x = jnp.stack([v[:, 0] + v[:, 1], v[:, 0] - v[:, 1]], axis=1).reshape(-1)
    return x


# symmetric real matrix, so the VJP is the transform itself (autodiff of the loop is ~4x slower)
_fwht.defvjp(lambda x, n: (_fwht(x, n), None), lambda n, _, ct: (_fwht(ct, n),))


def _topk_mask(a, K):
    """Mask of the K largest a >= 0, ties broken by lowest index (lax.top_k is ~10x slower on CPU).

    Bisection on the int64 bit pattern of a (monotone for a >= 0) gives the exact K-th value t."""
    bits = jax.lax.bitcast_convert_type(a, jnp.int64)

    def step(_, lh):
        mid = lh[0] + (lh[1] - lh[0]) // 2
        ok = jnp.sum(bits >= mid) >= K
        return jnp.where(ok, mid, lh[0]), jnp.where(ok, lh[1], mid)
    t, _ = jax.lax.fori_loop(0, 64, step, (jnp.int64(0), jnp.max(bits) + 1))
    tied = bits == t
    return (bits > t) | (tied & (jnp.cumsum(tied) <= K - jnp.sum(bits > t)))


def _ce_core(phi, K, n):
    """I_CE of a (not necessarily normalized) state; the top-K set is held fixed under grad."""
    s = jnp.argmax(jnp.abs(jax.lax.stop_gradient(phi)))
    phi = phi * jnp.conj(phi[s]) / jnp.abs(phi[s]) ** 2
    ell = jnp.log(jnp.maximum(jnp.abs(phi), EPS)) + 1j * jnp.angle(phi)
    c = _fwht(ell, n) / 2**n
    lK = _fwht(jnp.where(_topk_mask(jax.lax.stop_gradient(jnp.abs(c)), K), c, 0), n)
    phiK = jnp.exp(lK - jax.lax.stop_gradient(jnp.max(lK.real)))
    ov = jnp.vdot(phi, phiK)
    return 1 - jnp.abs(ov) ** 2 / (jnp.vdot(phi, phi).real * jnp.vdot(phiK, phiK).real)


def su2(a):
    """Rz(a0) Ry(a1) Rz(a2), a: (..., 3)."""
    e0, e2 = jnp.exp(-0.5j * a[..., 0]), jnp.exp(-0.5j * a[..., 2])
    c, s = jnp.cos(a[..., 1] / 2), jnp.sin(a[..., 1] / 2)
    return jnp.stack([jnp.stack([e0 * e2 * c, -e0 * jnp.conj(e2) * s], -1),
                      jnp.stack([jnp.conj(e0) * e2 * s, jnp.conj(e0 * e2) * c], -1)], -2)


_JIT = {}


def _fns(n, K):
    """Jitted I_CE(phi), its (value, cotangent) and the VJP of the Euler-angle map, per (n, K)."""
    if (n, K) not in _JIT:
        core = lambda phi: _ce_core(phi, K, n)
        _JIT[n, K] = (jax.jit(core),
                      jax.jit(lambda phi: (lambda v, f: (v, f(jnp.ones(()))[0]))(*jax.vjp(core, phi))),
                      jax.jit(lambda ang, gV: jax.vjp(lambda a: su2(a.reshape(n, 3)), ang)[1](gV)[0]))
    return _JIT[n, K]


def loss_and_grad(ang, phi, K):
    """I_CE[V(ang) phi] and its gradient wrt the 3n Euler angles.

    The local layer and its backward pass run in numba (JAX's is ~6x slower here); the CE core
    runs in JAX. Uses JAX's cotangent convention: through y = V x, ct_x = V^T ct_y and
    ct_V[a, b] = sum ct_y[a] x[b]; conj(ct) is propagated so wn._site_matrix can do the sum."""
    n = wn.nqubits(phi)
    _, vjp_core, vjp_su2 = _fns(n, K)
    Vs = np.asarray(su2(jnp.asarray(ang).reshape(n, 3)))
    x = np.array(phi, dtype=complex)
    for j in range(n):
        wn._apply_1q_inplace(x, Vs[j], j, n)
    val, ct = vjp_core(x)
    cc = np.conj(np.asarray(ct))
    gV = np.empty((n, 2, 2), complex)
    for j in reversed(range(n)):
        Vd = np.ascontiguousarray(Vs[j].conj().T)
        wn._apply_1q_inplace(x, Vd, j, n)  # x <- layer-j input (V unitary)
        gV[j] = wn._site_matrix(cc, x, j, n)
        wn._apply_1q_inplace(cc, Vd, j, n)
    return float(val), np.asarray(vjp_su2(jnp.asarray(ang), jnp.asarray(gV)), float)


def ce_infidelity(phi, K):
    phi = np.asarray(phi, complex)
    return float(_fns(wn.nqubits(phi), int(K))[0](phi))


def unitary_to_angles(V):
    """Euler angles with su2(angles) = V up to a global phase."""
    V = np.asarray(V, complex)
    V = V / np.sqrt(np.linalg.det(V))
    b = 2 * np.arctan2(abs(V[1, 0]), abs(V[0, 0]))
    p, q = np.angle(V[1, 1]), np.angle(V[1, 0])  # V11 = e^{i(a0+a2)/2} c, V10 = e^{i(a0-a2)/2} s
    return np.array([p + q, b, p - q])


def angles_to_unitaries(ang):
    return list(np.asarray(su2(jnp.asarray(ang).reshape(-1, 3))))


def _local_solve(args):
    phi, K, ang0, maxiter = args
    r = minimize(loss_and_grad, np.asarray(ang0, float), args=(phi, K), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter))
    return float(r.fun), r.x, int(r.nfev)


def ce_local(phi, K, inits, maxiter=150, pool=None):
    """min over V = prod_j V_j (SU(2) per site) of I_CE[V phi; K], L-BFGS from each init.

    inits: list of lists of 2x2 unitaries. Returns dict I (best), Vs, I_all, nfev."""
    phi = np.asarray(phi, complex) / np.linalg.norm(phi)
    tasks = [(phi, int(K), np.concatenate([unitary_to_angles(V) for V in Vs]), maxiter)
             for Vs in inits]
    res = list(pool.map(_local_solve, tasks) if pool is not None else map(_local_solve, tasks))
    I_all = np.array([r[0] for r in res])
    k = int(np.argmin(I_all))
    return dict(I=I_all[k], Vs=angles_to_unitaries(res[k][1]), I_all=I_all,
                nfev=sum(r[2] for r in res))


def random_unitaries(n, rng):
    return angles_to_unitaries(rng.uniform(0, 2 * np.pi, size=(n, 3)))


def ce_inits(phi, rng, extra=(), n_random=8):
    n = wn.nqubits(phi)
    return [[wn.I2] * n, *extra] + [random_unitaries(n, rng) for _ in range(n_random)]


def ce_final(phi, K, rng, extra=(), n_random=32, pool=None):
    return ce_local(phi, K, ce_inits(phi, rng, extra, n_random), maxiter=500, pool=pool)


# ---------------------------------------------------------------- beam-search helpers
def screen(args):
    """I_CE[V g phi] for each gate g (forward only)."""
    phi, Vs, gates, K = args
    return [ce_infidelity(wn.apply_local(wn.apply_gate(phi, g), Vs), K) for g in gates]


def eval_child(args):
    """One search-mode inner solve (one init) on g phi. Returns (I, angles)."""
    phi, g, ang0, K = args
    c = wn.apply_gate(phi, g)
    I, x, _ = _local_solve((c / np.linalg.norm(c), K, ang0, 150))
    return I, x


def search_init_angles(Vs_parent, n, rng, n_random):
    inits = [Vs_parent, [wn.I2] * n] + [random_unitaries(n, rng) for _ in range(n_random)]
    return [np.concatenate([unitary_to_angles(V) for V in Vs]) for Vs in inits]


# ---------------------------------------------------------------- tests
def _graph_state(n, edges):
    s = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1
    sign = (-1.0) ** sum(s[:, i] * s[:, j] for i, j in edges)
    return sign.astype(complex) / 2 ** (n / 2)


def run_tests(verbose=True):
    rng = np.random.default_rng(2)
    results = {}

    def check(name, fn):
        fn()
        results[name] = "pass"
        if verbose:
            print(f"{name}: pass", flush=True)

    def t12():
        phi = wn._rand_state(10, rng)
        assert ce_infidelity(phi, 2**10) < 1e-12

    def t13():
        n = 8
        phi = wn._product([V @ np.array([1, 1]) / 2**0.5 for V in random_unitaries(n, rng)])
        r = ce_local(phi, n + 1, ce_inits(phi, rng, n_random=8), maxiter=500)
        assert r["I"] < 1e-8, r

    def t14():
        n = 10
        edges = [(i, (i + 1) % n) for i in range(n)]
        z = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1)
        h, J = rng.normal(size=n), rng.normal(size=n)  # real Ising-Jastrow ring
        phi = np.exp(z @ h * 0.3 + sum(J[k] * z[:, i] * z[:, j] for k, (i, j) in enumerate(edges)) * 0.3)
        assert ce_infidelity(phi, 1 + n + len(edges)) < 1e-12
        assert ce_infidelity(phi, n + len(edges)) > 1e-6  # the cut actually matters
        # ring graph state: the principal log is i*pi*(sum s_i s_j mod 2), a flat Walsh spectrum
        # on 2^rank_GF2(A) = 2^8 coefficients (+ c_0), not 1 + n + #edges
        phi = _graph_state(n, edges)
        c = wn.fwht(np.log(abs(phi)) + 1j * np.angle(phi))
        nnz = int((abs(c) > 1e-12).sum())
        assert nnz in (2**8, 2**8 + 1), nnz
        assert ce_infidelity(phi, nnz) < 1e-12 and ce_infidelity(phi, 1 + n + len(edges)) > 0.1
        assert ce_infidelity(phi, nnz - 1) > 1e-6  # exact top-K even with all-equal |c| (ties)

    def t15():
        import sys
        import netket as nk
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "../../ClusterExpansion"))
        import cluster_expansion as cx
        n = 10
        phi = wn._rand_state(n, rng)
        phi = phi * np.conj(phi[np.argmax(abs(phi))]) / abs(phi[np.argmax(abs(phi))]) ** 2
        ell = np.log(np.maximum(abs(phi), EPS)) + 1j * np.angle(phi)
        hi = nk.hilbert.Spin(0.5, n)  # NetKet order = walshness order (T1)
        coef = cx.fwht_coeffs_in_cluster_col_order(ell, hi)
        for K in (5, 50, 300, 1000):
            phiK = cx.compress_and_reconstruct_cached(coef, K, hi)
            ref = 1 - abs(np.vdot(phi, phiK)) ** 2 / (np.vdot(phi, phi).real * np.vdot(phiK, phiK).real)
            assert abs(ref - ce_infidelity(phi, K)) < 1e-10, (K, ref, ce_infidelity(phi, K))

    def t16():
        n, K = 8, 40
        phi = wn._rand_state(n, rng)
        x = rng.uniform(0, 2 * np.pi, 3 * n)
        g = loss_and_grad(x, phi, K)[1]
        assert np.isclose(loss_and_grad(x, phi, K)[0],
                          ce_infidelity(wn.apply_local(phi, angles_to_unitaries(x)), K))
        h = 1e-6
        for k in range(3 * n):
            e = np.zeros(3 * n)
            e[k] = h
            fd = (loss_and_grad(x + e, phi, K)[0] - loss_and_grad(x - e, phi, K)[0]) / (2 * h)
            assert abs(fd - g[k]) < 1e-6 * max(1, abs(g[k])), (k, fd, g[k])
        V = random_unitaries(3, rng)  # angle round trip, up to global phase
        for a, b in zip(V, angles_to_unitaries(np.concatenate([unitary_to_angles(v) for v in V]))):
            assert abs(abs(np.vdot(a, b)) - 2) < 1e-12

    for name, fn in [("T12 K=2^n exact", t12), ("T13 rotated product", t13),
                     ("T14 ring graph state", t14), ("T15 vs cluster_expansion.py", t15),
                     ("T16 grad vs FD", t16)]:
        check(name, fn)
    return results


if __name__ == "__main__":
    t0 = time.time()
    run_tests()
    print(f"{time.time() - t0:.1f} s")
