"""Walshness numerics for plans/01_walshness.md.

Conventions: |s>, s in {0,1}^n, Z_j|s> = (-1)^{s_j}|s>, idx(s) = sum_j s_j 2^{n-1-j}
(site 0 most significant; NetKet sigma_j = 1 - 2 s_j). State vectors are dense numpy arrays.
Run `python walshness.py` for the self-tests (T1-T11 of the plan).
"""
import os
import time
from functools import lru_cache
from itertools import combinations

import numpy as np
from numba import njit
from scipy.special import logsumexp

I2 = np.eye(2, dtype=complex)
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
HAD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
SGATE = np.diag([1, 1j])
CZ4 = np.diag([1, 1, 1, -1]).astype(complex)
AX = "xyz"


def nqubits(psi):
    return int(psi.size).bit_length() - 1


# ---------------------------------------------------------------- state ops
@njit(cache=True)
def _apply_1q_inplace(x, op, j, n):
    L = 1 << (n - 1 - j)
    a, b, c, d = op[0, 0], op[0, 1], op[1, 0], op[1, 1]
    for hi in range(1 << j):
        for i0 in range(2 * hi * L, (2 * hi + 1) * L):
            x0, x1 = x[i0], x[i0 + L]
            x[i0] = a * x0 + b * x1
            x[i0 + L] = c * x0 + d * x1


@njit(cache=True)
def _site_matrix(phi, chi, j, n):
    L = 1 << (n - 1 - j)
    m00 = m01 = m10 = m11 = 0j
    for hi in range(1 << j):
        for i0 in range(2 * hi * L, (2 * hi + 1) * L):
            p0, p1 = phi[i0].conjugate(), phi[i0 + L].conjugate()
            c0, c1 = chi[i0], chi[i0 + L]
            m00 += p0 * c0
            m01 += p0 * c1
            m10 += p1 * c0
            m11 += p1 * c1
    return np.array([[m00, m01], [m10, m11]])


def apply_1q(psi, op, j):
    x = np.array(psi, dtype=complex)
    _apply_1q_inplace(x, np.asarray(op, dtype=complex), j, nqubits(x))
    return x


def apply_2q(psi, G, i, j):
    """4x4 gate G on sites i<j (site i = first tensor factor)."""
    v = psi.reshape(2**i, 2, 2 ** (j - i - 1), 2, -1)
    return np.einsum("abcd,xcydz->xaybz", G.reshape(2, 2, 2, 2), v).reshape(-1)


def apply_gate(psi, gate):
    _, sites, M = gate
    return apply_1q(psi, M, sites[0]) if len(sites) == 1 else apply_2q(psi, M, *sites)


def apply_word(psi, word):
    for g in word:
        psi = apply_gate(psi, g)
    return psi


def apply_local(psi, Vs):
    for j, V in enumerate(Vs):
        psi = apply_1q(psi, V, j)
    return psi


def site_matrix(phi, chi, j):
    """M[a,b] = sum over other sites of conj(phi[..a..]) chi[..b..]."""
    return _site_matrix(np.asarray(phi, complex), np.asarray(chi, complex), j, nqubits(phi))


def pauli_vec(M):
    """(Tr(sigma^a M^T))_a = <phi|sigma^a|chi> for M = site_matrix(phi, chi, j)."""
    return np.einsum("aij,ij->a", PAULI, M)


# ---------------------------------------------------------------- Walsh
@lru_cache(maxsize=None)
def hamming(n):
    r = np.arange(2**n)
    return sum((r >> k) & 1 for k in range(n)).astype(float)


def fwht(psi):
    """psi_hat(r) = 2^-n sum_s (-1)^{r.s} psi(s)."""
    x = np.array(psi, dtype=complex)
    n = nqubits(x)
    for j in range(n):
        v = x.reshape(2**j, 2, -1)
        a = v[:, 0].copy()
        v[:, 0] += v[:, 1]
        v[:, 1] = a - v[:, 1]
    return x / 2**n


def walsh_spectrum(psi):
    p = np.abs(fwht(psi)) ** 2
    return p / p.sum()


def walshness_Z(psi, alpha=1.0):
    with np.errstate(divide="ignore"):
        return logsumexp(np.log(walsh_spectrum(psi)) + alpha * hamming(nqubits(psi))) / alpha


def hamming_histogram(psi):
    n = nqubits(psi)
    return np.bincount(hamming(n).astype(int), weights=walsh_spectrum(psi), minlength=n + 1)


def g_alpha(alpha):
    return np.log((1 + np.exp(alpha)) / 2) / alpha


# ---------------------------------------------------------------- Bloch / LB
def bloch_vectors(psi):
    psi = psi / np.linalg.norm(psi)
    return np.array([pauli_vec(site_matrix(psi, psi, j)).real for j in range(nqubits(psi))])


def lower_bound(psi):
    return lb_from_bloch(bloch_vectors(psi))


def lb_from_bloch(b):
    return float(np.sum((1 - np.linalg.norm(b, axis=1)) / 2))


def two_site_rdms(psi):
    """{(i,j): rho_ij (4x4, site i first)} for all i<j."""
    psi = psi / np.linalg.norm(psi)
    n = nqubits(psi)
    out = {}
    for i, j in combinations(range(n), 2):
        v = psi.reshape(2**i, 2, 2 ** (j - i - 1), 2, -1)
        out[i, j] = np.einsum("xaybz,xcydz->abcd", v, v.conj()).reshape(4, 4)
    return out


def bloch_pair(rho):
    """Bloch vectors of both sites of a two-site density matrix."""
    r = rho.reshape(2, 2, 2, 2)
    ri, rj = np.einsum("abcb->ac", r), np.einsum("abad->bd", r)
    return [np.einsum("aij,ji->a", PAULI, x).real for x in (ri, rj)]


def fingerprint(psi, extra=()):
    b = np.sort(np.linalg.norm(bloch_vectors(psi), axis=1))
    pur = np.sort([np.trace(r @ r).real for r in two_site_rdms(psi).values()])
    return tuple(np.round(np.concatenate([b, pur, extra]), 9))


# ---------------------------------------------------------------- inner solver
def A_op(nv, alpha, inverse=False):
    s = np.sinh(alpha / 2) * (-1 if not inverse else 1)
    return np.cosh(alpha / 2) * I2 + s * np.einsum("a,abc->bc", nv, PAULI)


def _descent(phi, axes, alpha, max_sweeps=200, tol=1e-10):
    """Exact single-site updates minimizing E = <phi| prod_j A(n_j) |phi>."""
    axes = np.array(axes, dtype=float)
    n = len(axes)
    phi = np.asarray(phi, complex)
    chi = apply_local(phi, [A_op(a, alpha) for a in axes])
    E = np.vdot(phi, chi).real
    for _ in range(max_sweeps):
        for j in range(n):
            # chi_{-j} = A_j^{-1} chi is never formed: <phi|.|chi_{-j}> = site_matrix(phi, chi) A_j^{-T}
            Ainv = A_op(axes[j], alpha, inverse=True)
            v = pauli_vec(_site_matrix(phi, chi, j, n) @ Ainv.T)
            assert np.abs(v.imag).max() <= 1e-8 * max(1.0, np.abs(v).max()), v
            v = v.real
            if np.linalg.norm(v) > 0:
                axes[j] = v / np.linalg.norm(v)
            _apply_1q_inplace(chi, A_op(axes[j], alpha) @ Ainv, j, n)
        E_new = np.vdot(phi, chi).real
        assert E_new > 0 and E_new <= E * (1 + 1e-12), (E, E_new)
        done = E - E_new <= tol * max(1.0, abs(E))
        E = E_new
        if done:
            break
    return E, axes


def _solve(args):
    phi, axes, schedule = args
    for a in schedule:
        E, axes = _descent(phi, axes, a)
    return E, axes


def intrinsic_walshness(phi, alpha, inits, continuation=False, pool=None):
    """min_V W^Z_alpha[V phi] by coordinate descent from each init (n,3) axis array.

    Returns dict W (best), axes (best), W_all (every restart)."""
    phi = phi / np.linalg.norm(phi)
    n = nqubits(phi)
    sched = [a for a in (0.25, 0.5, 1.0) if a < alpha] + [alpha] if continuation else [alpha]
    tasks = [(phi, x, sched) for x in inits]
    res = list(pool.map(_solve, tasks) if pool is not None else map(_solve, tasks))
    W_all = np.array([n / 2 + np.log(E) / alpha for E, _ in res])
    k = int(np.argmin(W_all))
    return dict(W=W_all[k], axes=res[k][1], W_all=W_all)


def random_axes(n, rng):
    x = rng.normal(size=(n, 3))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def bloch_axes(phi, rng):
    b = bloch_vectors(phi)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    return np.where(nb > 1e-8, b / np.maximum(nb, 1e-300), random_axes(len(b), rng))


def final_mode(phi, alpha, rng, extra=(), n_random=32, pool=None):
    n = nqubits(phi)
    inits = [bloch_axes(phi, rng), *extra] + [random_axes(n, rng) for _ in range(n_random)]
    return intrinsic_walshness(phi, alpha, inits, continuation=True, pool=pool)


def search_inits(phi, parent_axes, rng):
    return [bloch_axes(phi, rng), parent_axes, random_axes(nqubits(phi), rng)]


# ---------------------------------------------------------------- axes -> V
def axis_to_unitary(nv):
    """Minimal rotation V with V^dag X V = n.sigma."""
    nv = np.asarray(nv, float) / np.linalg.norm(nv)
    m = np.cross([1.0, 0, 0], nv)
    if np.linalg.norm(m) < 1e-12:
        return I2.copy() if nv[0] > 0 else 1j * PAULI[2]
    th = np.arctan2(np.linalg.norm(m), nv[0])  # = arccos(n_x), but accurate near n = +-x
    m /= np.linalg.norm(m)
    W = np.cos(th / 2) * I2 - 1j * np.sin(th / 2) * np.einsum("a,abc->bc", m, PAULI)
    V = W.conj().T
    assert np.allclose(V.conj().T @ PAULI[0] @ V, np.einsum("a,abc->bc", nv, PAULI), atol=1e-12)
    return V


def axes_to_unitaries(axes):
    return [axis_to_unitary(a) for a in axes]


def snapping_error(axes):
    """max_j (1 - max_a |n_ja|): 0 iff every axis is a +-Pauli axis."""
    return float(np.max(1 - np.abs(axes).max(axis=1)))


# ---------------------------------------------------------------- gates
def projector(a):
    return (I2 - PAULI[AX.index(a)]) / 2


def G_proj(a, b):
    """G^{ab} = I - 2 Pi^a (x) Pi^b."""
    return np.eye(4) - 2 * np.kron(projector(a), projector(b))


_L = {"z": I2, "x": HAD, "y": HAD @ SGATE.conj()}


def G_std(a, b):
    L = np.kron(_L[a], _L[b])
    return L.conj().T @ CZ4 @ L


_STIM = {"z": "", "x": "X", "y": "Y"}


def stim_name(a, b):
    return {"zz": "CZ", "zx": "CX", "zy": "CY"}.get(a + b, f"{_STIM[a]}C{b.upper()}")


def G_gates(n):
    """All 9 binom(n,2) generalized CZ gates as (stim_label, (i,j), 4x4)."""
    return [(stim_name(a, b), (i, j), G_proj(a, b))
            for i, j in combinations(range(n), 2) for a in AX for b in AX]


def HSCZ_gates(n):
    return ([("H", (i,), HAD) for i in range(n)] + [("S", (i,), SGATE) for i in range(n)]
            + [("CZ", (i, j), CZ4) for i, j in combinations(range(n), 2)])


def word_str(word):
    return " ".join(f"{g[0]}{list(g[1])}" for g in word)


# ---------------------------------------------------------------- beam search
def make_pool(n_workers=28):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    return ProcessPoolExecutor(n_workers, mp_context=mp.get_context("spawn"))


def _eval_children(args):
    """Apply each gate to parent phi and evaluate the objective."""
    phi, gates, mode, alpha, parent_axes, seed = args
    rng = np.random.default_rng(seed)
    out = []
    for g in gates:
        c = apply_gate(phi, g)
        if mode == "Z":
            out.append((walshness_Z(c, alpha), None))
        else:
            r = intrinsic_walshness(c, alpha, search_inits(c, parent_axes, rng))
            out.append((r["W"], r["axes"]))
    return out


def _child_lb(parent_lb, bloch, rdms, g):
    if len(g[1]) == 1:
        return parent_lb  # LB is invariant under local unitaries
    i, j = g[1]
    bi, bj = bloch_pair(g[2] @ rdms[i, j] @ g[2].conj().T)
    return parent_lb + (np.linalg.norm(bloch[i]) + np.linalg.norm(bloch[j])
                        - np.linalg.norm(bi) - np.linalg.norm(bj)) / 2


def beam_search(psi, gates, mode, alpha=1.0, B=4, K=16, D_max=None, patience=2, pool=None,
                seed=0, init_axes=None, verbose=True, ce_K=None, init_Vs=(), ce_random=1):
    """Beam search over gate words C = g_k...g_1 acting on psi.

    mode "intrinsic": J(C) = W_alpha[C psi] (min over local V, the cliff->local basis);
    mode "Z":         J(C) = W^Z_alpha[C psi] (fixed basis, the local->cliff basis);
    mode "CE":        J(C) = min_V I_CE[V C psi; ce_K] (general SU(2) V, ce_basis.py). The LB
                      is no bound here: children are screened by I_CE[V_parent g phi_parent];
                      inits per child: parent V, identity, `ce_random` random.
    K = children per parent kept after LB (or CE) screening (None: all).
    Returns dict word, phi (= C psi), J, axes (Vs for "CE"), history."""
    rng = np.random.default_rng(seed)
    psi = psi / np.linalg.norm(psi)
    n = nqubits(psi)
    D_max = D_max or n
    if mode == "CE":
        import ce_basis as ce
        r = ce.ce_final(psi, ce_K, rng, extra=init_Vs, pool=pool)
        root = dict(word=[], phi=psi, J=r["I"], axes=r["Vs"], I_restarts=r["I_all"])
    elif mode == "Z":
        root = dict(word=[], phi=psi, J=walshness_Z(psi, alpha), axes=None)
    else:
        extra = [] if init_axes is None else [init_axes]
        r = final_mode(psi, alpha, rng, extra=extra, pool=pool)
        root = dict(word=[], phi=psi, J=r["W"], axes=r["axes"])
    root["LB"] = lower_bound(psi)
    best, beam, stall = root, [root], 0
    history = [dict(depth=0, J=best["J"], LB=best["LB"], t=0.0)]
    t0 = time.time()
    for depth in range(1, D_max + 1):
        tasks, meta = [], []
        for parent in beam:
            bloch, rdms = bloch_vectors(parent["phi"]), two_site_rdms(parent["phi"])
            last = parent["word"][-1][:2] if parent["word"] else None
            cand = [(_child_lb(parent["LB"], bloch, rdms, g), g) for g in gates if g[:2] != last]
            if mode == "CE":
                chunks = [cand[k:k + 50] for k in range(0, len(cand), 50)]
                scr = [x for r in pool.map(ce.screen, [(parent["phi"], parent["axes"],
                                                        [g for _, g in ch], ce_K) for ch in chunks])
                       for x in r]
                cand = [c for _, c in sorted(zip(scr, cand), key=lambda x: x[0])][:K]
                for lb, g in cand:
                    for a in ce.search_init_angles(parent["axes"], n, rng, ce_random):
                        tasks.append((parent["phi"], g, a, ce_K))
                        meta.append((parent, [(lb, g)]))
                continue
            cand = sorted([c for c in cand if c[0] < best["J"] - 1e-12], key=lambda c: c[0])[:K]
            if mode == "Z":  # cheap objective: chunk the children
                for k in range(0, len(cand), 24):
                    chunk = cand[k:k + 24]
                    tasks.append((parent["phi"], [g for _, g in chunk], mode, alpha, None, 0))
                    meta.append((parent, chunk))
            else:
                for lb, g in cand:
                    tasks.append((parent["phi"], [g], mode, alpha, parent["axes"],
                                  int(rng.integers(2**31))))
                    meta.append((parent, [(lb, g)]))
        if mode == "CE":  # one task per (child, init); keep the best init per child
            results = {}
            for (parent, [(lb, g)]), (J, x) in zip(meta, pool.map(ce.eval_child, tasks)):
                key = (id(parent), g[:2])
                if key not in results or J < results[key][0]:
                    results[key] = (J, x, parent, lb, g)
            children = [dict(word=p["word"] + [g], parent=p, LB=lb, J=J,
                             axes=ce.angles_to_unitaries(x)) for J, x, p, lb, g in results.values()]
        else:
            results = pool.map(_eval_children, tasks) if pool is not None else map(_eval_children, tasks)
            children = [dict(word=parent["word"] + [g], parent=parent, LB=lb, J=J, axes=ax)
                        for (parent, chunk), res in zip(meta, results)
                        for (lb, g), (J, ax) in zip(chunk, res)]
        if not children:
            break
        children.sort(key=lambda c: (c["J"], c["LB"], len(c["word"])))
        beam, seen = [], set()
        for c in children:
            c["phi"] = apply_gate(c.pop("parent")["phi"], c["word"][-1])
            fp = fingerprint(c["phi"], extra=[c["J"]] if mode == "Z" else [])
            if fp in seen:
                continue
            seen.add(fp)
            beam.append(c)
            if len(beam) == B:
                break
        if beam[0]["J"] < best["J"] - 1e-9:
            best, stall = beam[0], 0
            if mode == "intrinsic":
                r = final_mode(best["phi"], alpha, rng, extra=[best["axes"]], pool=pool)
                if r["W"] < best["J"]:
                    best["J"], best["axes"] = r["W"], r["axes"]
            elif mode == "CE":
                r = ce.ce_final(best["phi"], ce_K, rng, extra=[best["axes"]], pool=pool)
                best["I_restarts"] = r["I_all"]
                if r["I"] < best["J"]:
                    best["J"], best["axes"] = r["I"], r["Vs"]
        else:
            stall += 1
        history.append(dict(depth=depth, J=best["J"], beamJ=beam[0]["J"], LB=best["LB"],
                            t=time.time() - t0, n_children=len(children)))
        if verbose:
            print(f"depth {depth:2d}: beam J={beam[0]['J']:.6f} best J={best['J']:.6f} "
                  f"LB={best['LB']:.6f} children={len(children)} t={time.time() - t0:.0f}s",
                  flush=True)
        if best["J"] < 1e-8 or stall >= patience:
            break
    out = dict(word=best["word"], phi=best["phi"], J=best["J"], LB=best["LB"], history=history)
    if mode == "CE":
        out.update(Vs=best["axes"], I_restarts=best["I_restarts"])
    else:
        out["axes"] = best["axes"]
    return out


# ---------------------------------------------------------------- Hamiltonians
def terms_from_netket(op):
    out = {}
    for s, w in zip(op.operators, op.weights):
        out[str(s)] = out.get(str(s), 0) + complex(w)
    return out


def to_netket(terms, hilbert):
    import netket as nk
    return nk.operator.PauliStrings(hilbert, list(terms), [complex(w) for w in terms.values()])


def _merge(pairs, cut=1e-12):
    out = {}
    for s, w in pairs:
        out[s] = out.get(s, 0) + w
    return {s: w for s, w in out.items() if abs(w) >= cut}


def clifford_conjugate(terms, word):
    """P -> C P C^dag for C = g_k...g_1 (word in time order). Returns (terms, n_before_merge)."""
    import stim
    circ = stim.Circuit()
    for name, sites, _ in word:
        circ.append(name, list(sites))
    n = len(next(iter(terms)))
    circ.append("I", [n - 1])  # pin the qubit count
    tab = stim.Tableau.from_circuit(circ)
    pairs = []
    for s, w in terms.items():
        q = tab(stim.PauliString(s.replace("I", "_")))
        pairs.append(("".join("IXYZ"[k] for k in q), w * q.sign))
    return _merge(pairs), len(pairs)


def local_conjugate(terms, Vs):
    """sigma^a_j -> V_j sigma^a_j V_j^dag = sum_b R_ba sigma^b. Returns (terms, n_before_merge)."""
    R = [np.einsum("bij,jk,akl,li->ba", PAULI, V, PAULI, V.conj().T).real / 2 for V in Vs]
    pairs = []
    for s, w in terms.items():
        partial = [("", w)]
        for j, c in enumerate(s):
            if c == "I":
                partial = [(p + "I", x) for p, x in partial]
                continue
            col = R[j][:, "XYZ".index(c)]
            partial = [(p + "XYZ"[b], x * col[b]) for p, x in partial for b in range(3)
                       if abs(col[b]) > 1e-12]
        pairs += partial
    return _merge(pairs), len(pairs)


def term_stats(terms, n_before=None):
    w = np.array([abs(x) for x in terms.values()])
    return dict(n_before=n_before if n_before is not None else len(terms), n_terms=len(terms),
                n_gt_1e8=int((w > 1e-8).sum()), n_gt_1e4=int((w > 1e-4).sum()),
                max_weight=max(sum(c != "I" for c in s) for s in terms))


def pauli_expect(terms, psi):
    out = 0
    for s, w in terms.items():
        x = psi
        for j, c in enumerate(s):
            if c != "I":
                x = apply_1q(x, PAULI["XYZ".index(c)], j)
        out += w * np.vdot(psi, x)
    return out / np.vdot(psi, psi).real


# ---------------------------------------------------------------- RBM
def train_rbm(terms, outfile, targets=(), cpus=None, seed=1234, n_iter=7000, n_samples=2**9,
              lr=0.002, diag_shift=3e-4, rbm_alpha=5, metadata=None):
    """RBM + MetropolisLocal + VMC_SR on the Pauli sum `terms` (settings of Kitaev.ipynb).

    Writes the RuntimeLog + metadata (+ final infidelity to span(targets)) to outfile (json).
    `cpus` pins this process before JAX starts (use from a fresh worker process)."""
    if cpus is not None:
        os.sched_setaffinity(0, cpus)
    import json
    import netket as nk
    import netket.experimental as nkx  # VMC_SR lives here in NetKet 3.19

    hi = nk.hilbert.Spin(0.5, len(next(iter(terms))))
    vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi),
                        nk.models.RBM(alpha=rbm_alpha, param_dtype=np.complex64),
                        n_samples=n_samples, seed=seed, sampler_seed=seed)
    drv = nkx.driver.VMC_SR(to_netket(terms, hi), nk.optimizer.Sgd(learning_rate=lr),
                            variational_state=vs, diag_shift=diag_shift)
    log = nk.logging.RuntimeLog()
    t0 = time.time()
    drv.run(n_iter=n_iter, out=log, show_progress=False)
    phi = np.asarray(vs.to_array(normalize=True))
    # computed by hand (NetKet's InfidelityOperator is not used, see plan Sec. 8)
    infid = 1 - sum(abs(np.vdot(t / np.linalg.norm(t), phi)) ** 2 for t in targets)
    log.serialize(outfile)
    with open(outfile) as f:
        d = json.load(f)
    d.update(metadata or {}, final_infidelity=float(infid), wall_time=time.time() - t0, seed=seed,
             n_iter=n_iter, n_samples=n_samples, learning_rate=lr, diag_shift=diag_shift,
             RBMAlpha=rbm_alpha)
    with open(outfile, "w") as f:
        json.dump(d, f)
    return outfile


def dense(terms):
    """Dense matrix of a Pauli sum (tests only)."""
    P = dict(I=I2, X=PAULI[0], Y=PAULI[1], Z=PAULI[2])
    out = 0
    for s, w in terms.items():
        M = np.ones((1, 1))
        for c in s:
            M = np.kron(M, P[c])
        out = out + w * M
    return out


# ---------------------------------------------------------------- tests
def _ghz(n):
    psi = np.zeros(2**n, complex)
    psi[0] = psi[-1] = 2**-0.5
    return psi


def _product(vecs):
    out = np.ones(1)
    for v in vecs:
        out = np.kron(out, v)
    return out


def _rand_state(n, rng):
    x = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return x / np.linalg.norm(x)


def _rand_word(gates, k, rng):
    return [gates[i] for i in rng.integers(len(gates), size=k)]


def run_tests(verbose=True):
    import stim
    rng = np.random.default_rng(1)
    results = {}

    def check(name, fn):
        fn()
        results[name] = "pass"
        if verbose:
            print(f"{name}: pass", flush=True)

    def t1():
        import netket as nk
        for n in (3, 5):
            e = np.zeros(2**n)
            e[2 ** (n - 1)] = 1
            assert np.allclose(apply_1q(e, PAULI[2], 0), -e)
            hi = nk.hilbert.Spin(0.5, n)
            s = (1 - np.asarray(hi.all_states())) // 2
            assert np.array_equal(s @ (2 ** np.arange(n - 1, -1, -1)), np.arange(2**n))
            t = {"ZIX" + "I" * (n - 3): 0.7, "Y" * n: 0.3}
            assert np.allclose(to_netket(t, hi).to_dense(), dense(t))

    def t2():
        n = 6
        plus, minus = np.array([1, 1]) / 2**0.5, np.array([1, -1]) / 2**0.5
        for a in (1.0, 2.0):
            assert np.isclose(walshness_Z(_product([minus] * n), a), n)
            assert abs(walshness_Z(_product([plus] * n), a)) < 1e-12
            assert np.isclose(walshness_Z(_product([[1, 0]] * n), a), n * g_alpha(a))

    def t4():
        n = 6
        for _ in range(3):
            vecs = [axis_to_unitary(a).conj().T @ np.array([1, 1]) / 2**0.5
                    for a in random_axes(n, rng)]
            phi = _product(vecs)
            r = final_mode(phi, 1.0, rng, n_random=4)
            assert r["W"] < 1e-8, r["W"]

    def t5():
        for n in (4, 6, 8):
            phi = _ghz(n)
            assert np.isclose(lower_bound(phi), n / 2)
            r = final_mode(phi, 1.0, rng)
            assert abs(r["W"] - n / 2) < 1e-6, (n, r["W"])
            assert n * g_alpha(1.0) >= r["W"] >= lower_bound(phi) - 1e-12

    def t7():
        phi = _rand_state(6, rng)
        for a in (1.0, 2.0):
            r = final_mode(phi, a, rng, n_random=4)
            assert abs(walshness_Z(apply_local(phi, axes_to_unitaries(r["axes"])), a) - r["W"]) < 1e-8
            assert lower_bound(phi) - 1e-12 <= r["W"] <= 6 * g_alpha(a)
        assert np.allclose(axis_to_unitary([-1, 0, 0]), 1j * PAULI[2])
        assert np.allclose(axis_to_unitary([1, 0, 0]), I2)
        axis_to_unitary([-1, 1e-15, 3e-8])  # near -x: arccos(n_x) would fail the 1e-12 check

    def t9():
        for a in AX:
            for b in AX:
                G = G_proj(a, b)
                assert np.allclose(G, G_std(a, b))
                assert np.allclose(G @ G, np.eye(4))
                Ust = stim.Tableau.from_named_gate(stim_name(a, b)).to_unitary_matrix(endian="big")
                assert np.allclose(G, Ust), (a, b)
        assert np.allclose(G_proj("z", "x"), np.eye(4)[[0, 1, 3, 2]])  # CNOT(control 0, target 1)
        # stim conjugation direction and endianness: CNOT maps X_0 -> X_0 X_1, Z_1 -> Z_0 Z_1
        cx = [("CX", (0, 1), G_proj("z", "x"))]
        assert clifford_conjugate({"XI": 1.0}, cx)[0] == {"XX": 1.0}
        assert clifford_conjugate({"IZ": 1.0}, cx)[0] == {"ZZ": 1.0}
        e = np.zeros(4)
        e[2] = 1  # |10>
        assert np.allclose(apply_word(e, cx), np.eye(4)[3])  # -> |11>

    def t10():
        for n in (4, 6):
            r = beam_search(_ghz(n), G_gates(n), "intrinsic", B=2, K=8, verbose=False)
            assert r["J"] < 1e-8 and len(r["word"]) <= n - 1, (n, r["J"], len(r["word"]))

    def t11():
        n = 4
        strings = ["".join(rng.choice(list("IXYZ"), n)) for _ in range(12)]
        terms = _merge([(s, rng.normal()) for s in strings])
        psi = _rand_state(n, rng)
        Vs = axes_to_unitaries(random_axes(n, rng))
        for order, gates in (("VC", G_gates(n)), ("CV", HSCZ_gates(n))):
            word = _rand_word(gates, 6, rng)
            if order == "VC":
                HU = local_conjugate(clifford_conjugate(terms, word)[0], Vs)[0]
                psiU = apply_local(apply_word(psi, word), Vs)
            else:
                HU = clifford_conjugate(local_conjugate(terms, Vs)[0], word)[0]
                psiU = apply_word(apply_local(psi, Vs), word)
            H, Hu = dense(terms), dense(HU)
            assert np.allclose(np.linalg.eigvalsh(H), np.linalg.eigvalsh(Hu))
            assert np.isclose(np.vdot(psi, H @ psi), np.vdot(psiU, Hu @ psiU))

    for name, fn in [("T1 bit order", t1), ("T2 product states", t2), ("T4 inner solver", t4),
                     ("T5 GHZ", t5), ("T7 solver vs FWHT", t7), ("T9 gates/stim", t9),
                     ("T10 GHZ+Clifford", t10), ("T11 H_U", t11)]:
        check(name, fn)
    return results


if __name__ == "__main__":
    run_tests()
