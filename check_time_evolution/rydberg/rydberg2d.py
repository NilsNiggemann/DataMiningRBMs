"""Rydberg & related quench screen (PLAN_rydberg_quench.md): lattices, Pauli-string Hamiltonians,
piecewise-constant (Floquet) evolution with exact interaction-picture phase unwrapping, diagnostics.

Conventions as in quench2d.py: site i <-> bit (N-1-i), bit 0 <-> s=+1 <-> |g>, NetKet ordering.
Hamiltonians are lists of (coef, 'IXYZ...' string), so dense, sparse and NetKet versions come from one source.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, eigsh
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "quench2d"))  # quench2d.py, analysis.py, plotstyle.py
from quench2d import fwht, fit_mu, order_of_masks, NOISE

# ----------------------------------------------------------------------------- lattices
S3_ = np.sqrt(3)


def _lat(pos, L, sub=None, sub3=None, scale=None):
    """Distances in units of `scale` (default: the smallest min-image distance)."""
    pos = np.asarray(pos, float)
    N = len(pos)
    shifts = [np.zeros(2)] + [s * np.asarray(v, float) for v in L for s in (1, -1)]
    if len(L) == 2:
        shifts += [s1 * np.asarray(L[0]) + s2 * np.asarray(L[1]) for s1 in (1, -1) for s2 in (1, -1)]
    d = pos[None, :, :] - pos[:, None, :]
    R = np.min([np.linalg.norm(d + s, axis=-1) for s in shifts], axis=0)  # minimum image
    R[np.arange(N), np.arange(N)] = np.inf
    nn = [(i, j) for i in range(N) for j in range(i + 1, N) if R[i, j] < R.min() * 1.01]
    scale = R.min() if scale is None else scale
    # unit vector of the minimum-image displacement (for dipolar anisotropy)
    best = np.argmin([np.linalg.norm(d + s, axis=-1) for s in shifts], axis=0)
    D = d + np.array(shifts)[best]
    sub = np.array([(-1) ** i for i in range(N)]) if sub is None else np.asarray(sub)
    return dict(pos=pos, R=R / scale, D=D / scale, N=N, nn=nn, sub=sub, sub3=sub3)


def lattice(name):
    if name.startswith("chain") or name.startswith("ring"):
        N = int(name[5:] if name[0] == "c" else name[4:])
        return _lat([(i, 0) for i in range(N)], [(N, 0)] if name[0] == "r" else [])
    if name in ("sq4x3", "sq4x4"):
        Lx, Ly = 4, int(name[-1])
        return _lat([(x, y) for x in range(Lx) for y in range(Ly)], [(Lx, 0)] + ([(0, Ly)] if Ly == 4 else []),
                    sub=[(-1) ** (x + y) for x in range(Lx) for y in range(Ly)])
    if name in ("tri4x3", "tri4x4"):
        Lx, Ly = 4, int(name[-1])
        a1, a2 = np.array([1, 0]), np.array([0.5, S3_ / 2])
        return _lat([x * a1 + y * a2 for x in range(Lx) for y in range(Ly)], [Lx * a1, Ly * a2],
                    sub=[(-1) ** (x + y) for x in range(Lx) for y in range(Ly)],
                    sub3=[(x + 2 * y) % 3 for x in range(Lx) for y in range(Ly)])
    if name == "kag12":
        a1, a2 = np.array([2, 0]), np.array([1, S3_])
        b = [np.array([0, 0]), np.array([1, 0]), np.array([0.5, S3_ / 2])]
        return _lat([x * a1 + y * a2 + b[k] for x in range(2) for y in range(2) for k in range(3)], [2 * a1, 2 * a2],
                    sub3=[k for x in range(2) for y in range(2) for k in range(3)])
    if name == "hc12":
        a1, a2 = np.array([1.5, S3_ / 2]), np.array([0, S3_])
        return _lat([x * a1 + y * a2 + np.array([k, 0]) for x in range(2) for y in range(3) for k in range(2)],
                    [2 * a1, 3 * a2], sub=[(-1) ** k for x in range(2) for y in range(3) for k in range(2)])
    if name == "ruby12":
        # atoms on the bond midpoints of a kagome strip of 4 corner-sharing triangles (up, down, up, down), OBC
        tri = [[(0, 0), (1, 0), (0.5, S3_ / 2)], [(1, 0), (2, 0), (1.5, -S3_ / 2)],
               [(2, 0), (3, 0), (2.5, S3_ / 2)], [(3, 0), (4, 0), (3.5, -S3_ / 2)]]
        mids = [(np.add(t[a], t[b]) / 2) for t in tri for a, b in [(0, 1), (1, 2), (0, 2)]]
        return _lat(mids, [])
    if name == "tc12":
        # toric code on a 2x3 torus: qubits on the 12 links; positions = link midpoints
        Lx, Ly = 2, 3
        pos = [(x + 0.5, y) for x in range(Lx) for y in range(Ly)] + [(x, y + 0.5) for x in range(Lx) for y in range(Ly)]
        lat = _lat(pos, [(Lx, 0), (0, Ly)])
        h = lambda x, y: (x % Lx) * Ly + (y % Ly)
        v = lambda x, y: Lx * Ly + (x % Lx) * Ly + (y % Ly)
        lat["stars"] = [[h(x, y), h(x - 1, y), v(x, y), v(x, y - 1)] for x in range(Lx) for y in range(Ly)]
        lat["plaq"] = [[h(x, y), h(x, y + 1), v(x, y), v(x + 1, y)] for x in range(Lx) for y in range(Ly)]
        return lat
    raise ValueError(name)


def triangles(lat):
    nn = set(lat["nn"])
    N = lat["N"]
    return [(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)
            if {(i, j), (i, k), (j, k)} <= nn]


# ----------------------------------------------------------------------------- Pauli-string Hamiltonians
def P(N, ops):
    """Pauli string for {site: 'X'|'Y'|'Z'}."""
    s = ["I"] * N
    for i, o in ops.items():
        s[i] = o
    return "".join(s)


def pair_terms(lat, Jfun, ops=("ZZ",), cutoff=1e-12):
    """sum_{i<j} J(i,j) * (op op) for each op string in ops (e.g. ('XX','YY'))."""
    N, out = lat["N"], []
    for i in range(N):
        for j in range(i + 1, N):
            J = Jfun(i, j)
            if abs(J) > cutoff:
                out += [(J, P(N, {i: o[0], j: o[1]})) for o in ops]
    return out


def field_terms(N, h, op):
    h = np.broadcast_to(h, (N,))
    return [(h[i], P(N, {i: op})) for i in range(N) if h[i] != 0]


def rydberg_terms(lat, Omega, Delta, V):
    """Omega/2 sum X - Delta sum n + sum_{i<j} V_ij n_i n_j with n=(1-Z)/2, constants dropped.
    V: (N,N) matrix. Delta: scalar or per-site."""
    N = lat["N"]
    Delta = np.broadcast_to(Delta, (N,)).astype(float)
    hz = Delta / 2 - V.sum(1) / 4  # n_i n_j -> (ZZ - Z_i - Z_j)/4 ; -Delta n -> Delta/2 Z
    return (field_terms(N, Omega / 2, "X") + field_terms(N, hz, "Z")
            + pair_terms(lat, lambda i, j: V[i, j] / 4))


def sparse_parts(terms, N):
    """-> (diag (dim,), offdiag csr). Builds by grouping terms with equal X-mask."""
    dim = 2 ** N
    k = np.arange(dim, dtype=np.int64)
    diag = np.zeros(dim)
    groups = {}
    for c, s in terms:
        xm = sum(1 << (N - 1 - i) for i, o in enumerate(s) if o in "XY")
        zm = sum(1 << (N - 1 - i) for i, o in enumerate(s) if o in "ZY")
        ny = s.count("Y")
        val = c * (1j ** ny) * (1 - 2 * (np.bitwise_count(k & zm).astype(np.int64) & 1))
        if xm == 0:
            diag += val.real
        else:
            groups[xm] = groups.get(xm, 0) + val
    rows, cols, vals = [], [], []
    for xm, v in groups.items():
        rows.append(k ^ xm); cols.append(k); vals.append(v)
    if rows:
        v = np.concatenate(vals)
        v = v.real if np.allclose(v.imag, 0) else v
        off = sp.csr_matrix((v, (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
        off.eliminate_zeros()
    else:
        off = sp.csr_matrix((dim, dim))
    return diag, off


def nk_operator(hi, terms):
    import netket as nk
    return nk.operator.PauliStrings(hi, [s for _, s in terms], [c for c, _ in terms])


def eta(diag, off):
    """||H - diag H||_F / ||H - tr H / D||_F (basis dependent on purpose)."""
    o = np.sum(np.abs(off.data) ** 2)
    return np.sqrt(o / (o + np.sum((diag - diag.mean()) ** 2)))


# ----------------------------------------------------------------------------- states
def product_state(angles, phis=None):
    """prod_i (cos(th_i/2)|up> + e^{i phi_i} sin(th_i/2)|down>), |up> = bit 0 = Z=+1 = |g>."""
    psi = np.ones(1, complex)
    phis = np.zeros(len(angles)) if phis is None else phis
    for th, ph in zip(angles, phis):
        psi = np.kron(psi, [np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)])
    return psi


def initial_state(lat, init, theta=np.pi / 4):
    """'+x', 'th' (all tilted by theta), 'Z2th' (sublattice +1 at theta, -1 at pi-theta), '120' (xy-plane, 3 sublattices),
    'seed' (all at theta except site N//2 at pi-theta)."""
    N = lat["N"]
    if init == "+x":
        return product_state(np.full(N, np.pi / 2))
    if init == "th":
        return product_state(np.full(N, theta))
    if init == "Z2th":
        return product_state(np.where(lat["sub"] > 0, theta, np.pi - theta))
    if init == "seed":
        a = np.full(N, theta); a[N // 2] = np.pi - theta
        return product_state(a)
    if init == "120":
        return product_state(np.full(N, np.pi / 2), 2 * np.pi / 3 * np.asarray(lat["sub3"]))
    raise ValueError(init)


def ground_state(terms, N):
    d, o = sparse_parts(terms, N)
    H = o + sp.diags(d)
    v = eigsh(H, k=1, which="SA")[1][:, 0] if 2 ** N > 64 else np.linalg.eigh(H.toarray())[1][:, 0]
    return v.astype(complex) / np.linalg.norm(v)


# ----------------------------------------------------------------------------- evolution
def evolve(psi0, segs, t_save, dt_max, period=None):
    """Piecewise-constant evolution. segs = [(duration, diag, off)] (repeated cyclically if period is set; a single
    segment otherwise). Returns per save time: logabs, unwrapped
    phase arg(psi) and interaction-picture phase arg(psi~), psi~ = exp(+i int diag dt) psi.
    The phase is unwrapped exactly for the diagonal part and by step-to-step continuity for the rest."""
    N = int(np.log2(len(psi0)))
    psi = psi0.copy()
    ph = np.angle(psi0)
    ph_cl = np.zeros_like(ph)  # -int diag dt
    out_t, out = [], {"logabs": [], "ph": [], "pht": [], "psi_last": None}
    bounds = np.cumsum([0] + [s[0] for s in segs])
    T = bounds[-1] if period else np.inf
    t, isave = 0.0, 0
    t_save = np.asarray(t_save)
    A_cache = {}

    def record():
        out["logabs"].append(np.log(np.abs(psi) + 1e-300)); out["ph"].append(ph.copy()); out["pht"].append(ph - ph_cl)
        out_t.append(t)

    record(); isave = 1
    while isave < len(t_save):
        tau = t % T if period else t
        if period and T - tau < 1e-9:  # round-off just below a period boundary: zero-length steps would stall t
            tau -= T
        si = int(np.searchsorted(bounds, tau + 1e-9, side="right") - 1) if period else 0
        si = min(si, len(segs) - 1)
        seg_end = (t - tau + bounds[si + 1]) if period else np.inf
        t_next = min(t_save[isave], seg_end)
        dur, diag, off = segs[si]
        n = max(1, int(np.ceil((t_next - t) / dt_max - 1e-9)))
        h = (t_next - t) / n
        if off.nnz == 0:  # diagonal segment: exact
            tk = h * np.arange(1, n + 1)
            vs = psi[None, :] * np.exp(-1j * np.outer(tk, diag))
        else:  # all substeps from one interval call (one norm estimate); keep only the current segment's matrix
            if si not in A_cache:
                A_cache.clear(); A_cache[si] = (-1j * (off + sp.diags(diag))).tocsr()
            vs = expm_multiply(A_cache[si], psi, start=0, stop=t_next - t, num=n + 1, endpoint=True)[1:]
        steps = [(diag, v) for v in vs]
        for d, new in steps:
            ph += np.angle(new * np.conj(psi) * np.exp(1j * d * h)) - d * h
            ph_cl -= d * h
            psi = new
        t = t_next
        if abs(t - t_save[isave]) < 1e-9:
            t = float(t_save[isave])
            record(); isave += 1
    out["psi_last"] = psi
    return np.array(out_t), {k: np.array(v) for k, v in out.items() if k != "psi_last"}, psi


# ----------------------------------------------------------------------------- diagnostics
PAGE = lambda nA, nB: nA * np.log(2) - 2 ** (nA - nB) / 2 if nA <= nB else PAGE(nB, nA)


def entanglement(psi, N, eps=1e-3):
    """S1, S2 at every cut along the site ordering, and chi_eps (discarded weight <= eps) at the worst S1 cut."""
    S1, S2, chi = [], [], []
    for c in range(1, N):
        lam = np.linalg.svd(psi.reshape(2 ** c, -1), compute_uv=False) ** 2
        lam = lam / lam.sum()
        l = lam[lam > 1e-16]
        S1.append(-(l * np.log(l)).sum()); S2.append(-np.log((lam ** 2).sum()))
        tail = np.cumsum(lam[::-1])[::-1]  # tail[k] = sum_{j>=k} lam_j
        chi.append(int(np.argmax(np.append(tail, 0.0) <= eps)))
    S1, S2 = np.array(S1), np.array(S2)
    w = int(np.argmax(S1))
    cap = 2 ** min(w + 1, N - w - 1)
    return dict(S=S1[N // 2 - 1], S2=S2[N // 2 - 1], Smax=S1.max(), S2max=S2.max(), chi=chi[w], chi_frac=chi[w] / cap,
                S_rel=S1[N // 2 - 1] / PAGE(N // 2, N - N // 2))


def analyze(times, ev, N):
    """Fit proxies (amplitude, phase, interaction-picture phase), order spectra, entanglement, n_eff, minlog."""
    order = order_of_masks(N)
    pairs = np.where(order == 2)[0]
    keys = ["mu", "T", "R2", "mu2", "mu_ph", "T_ph", "mu_pht", "w_hi", "w_hi_ph", "minlog", "n_eff",
            "S", "S2", "Smax", "S2max", "chi", "chi_frac", "S_rel"]
    out = {k: [] for k in keys}
    out["W"], out["W_ph"] = [], []
    p = {k: (-10.0, 3.0) for k in ["a", "a2", "p", "pt"]}
    for k in range(len(times)):
        la = ev["logabs"][k]
        C = fwht(la)
        ph = ev["ph"][k] - ev["ph"][k].mean()
        pht = ev["pht"][k] - ev["pht"][k].mean()
        Cp, Cpt = fwht(ph), fwht(pht)
        res = {}
        for key, c, a0 in [("a", C, 1.0), ("a2", C, 2.0), ("p", Cp, 1.0), ("pt", Cpt, 1.0)]:
            alive = np.abs(c[1:]) > np.exp(NOISE)
            if order[1:][alive].max(initial=0) <= 2:  # exactly <= 2-body: no fit, mu = -inf (plan section 4)
                res[key] = (-np.inf, np.nan, np.nan); continue
            mu, T, R2, _ = fit_mu(c, a0, p[key])
            if np.isfinite(mu):
                p[key] = (mu, T)
            res[key] = (mu, T, R2)
        out["mu"].append(res["a"][0]); out["T"].append(res["a"][1]); out["R2"].append(res["a"][2])
        out["mu2"].append(res["a2"][0]); out["mu_ph"].append(res["p"][0]); out["T_ph"].append(res["p"][1])
        out["mu_pht"].append(res["pt"][0])
        W = np.bincount(order[1:], weights=C[1:] ** 2, minlength=N + 1)
        Wp = np.bincount(order[1:], weights=Cp[1:] ** 2, minlength=N + 1)
        out["W"].append(W); out["W_ph"].append(Wp)
        out["w_hi"].append(W[5:].sum() / max(W.sum(), 1e-300)); out["w_hi_ph"].append(Wp[5:].sum() / max(Wp.sum(), 1e-300))
        out["minlog"].append(2 * la.min() + N * np.log(2))
        out["n_eff"].append(int(np.sum(np.abs(np.sin(2 * Cp[pairs])) > 0.1)))
        psi = np.exp(la + 1j * ev["ph"][k])
        for kk, v in entanglement(psi / np.linalg.norm(psi), N).items():
            out[kk].append(v)
    return {k: np.array(v) for k, v in out.items()}


# ----------------------------------------------------------------------------- self-checks
if __name__ == "__main__":
    import netket as nk
    from scipy.linalg import expm
    rng = np.random.default_rng(0)

    # 1. NetKet and sparse H agree (all Pauli types, incl. Y and fields)
    lat = lattice("sq4x3"); N = lat["N"]
    terms = (pair_terms(lat, lambda i, j: rng.normal() / lat["R"][i, j] ** 3, ("XX", "YY", "ZZ"))
             + field_terms(N, rng.normal(size=N), "Z") + field_terms(N, 0.3, "X") + field_terms(N, 0.2, "Y")
             + [(0.7, P(N, {0: "X", 1: "Y", 2: "Z", 5: "X"}))])
    d, o = sparse_parts(terms, N)
    H = (o + sp.diags(d)).toarray()
    assert np.allclose(H, H.conj().T)
    assert np.allclose(nk_operator(nk.hilbert.Spin(0.5, N), terms).to_dense(), H), "NetKet/sparse mismatch"

    # 2. n_i n_j expansion: rydberg_terms equals the direct occupation-basis diagonal
    V = 1 / np.where(np.isinf(lat["R"]), np.inf, lat["R"]) ** 6
    Om, De = 0.3, 0.7
    d, o = sparse_parts(rydberg_terms(lat, Om, De, V), N)
    n = (1 - (1 - 2 * ((np.arange(2 ** N)[:, None] >> (N - 1 - np.arange(N))) & 1))) / 2
    direct = -De * n.sum(1) + 0.5 * np.einsum("ki,ij,kj->k", n, V, n)
    assert np.allclose(d - d.mean(), direct - direct.mean()), "n n expansion"
    assert np.allclose(o.toarray(), sparse_parts(field_terms(N, Om / 2, "X"), N)[1].toarray())

    # 3. Krylov (stepper) agrees with dense expm at N=12; unwrapped phase is consistent with psi
    psi0 = initial_state(lat, "th", 0.6)
    tsave = np.linspace(0, 3, 7)
    t, ev, psi = evolve(psi0, [(1.0, d, o)], tsave, dt_max=0.05)
    ref = expm(-1j * 3 * (o + sp.diags(d)).toarray()) @ psi0
    assert abs(abs(np.vdot(ref, psi)) - 1) < 1e-10
    assert np.allclose(np.exp(ev["logabs"][-1] + 1j * ev["ph"][-1]), ref, atol=1e-10)

    # 4. Omega = 0: |psi| constant, unwrapped phase = -t E_cl exactly (<= 2-body)
    d0, o0 = sparse_parts(rydberg_terms(lat, 0.0, 0.0, V), N)
    t, ev, _ = evolve(psi0, [(1.0, d0, o0)], np.linspace(0, 50, 11), dt_max=1.0)
    assert np.allclose(ev["logabs"], ev["logabs"][0])
    Cp = fwht(ev["ph"][-1] - ev["ph"][-1].mean())
    assert np.abs(Cp[order_of_masks(N) > 2]).max() < 1e-9, "Omega=0 phase must be <= 2-body"
    A = analyze(t, ev, N)
    assert np.isinf(A["mu"][-1]) and A["mu"][-1] < 0

    # 5. kicked Ising at J=b=pi/4, h=0 from |+x>... use |0>: stabilizer state -> |psi| uniform on its support
    lat = lattice("chain12"); N = 12
    dz, oz = sparse_parts(pair_terms(lat, lambda i, j: np.pi / 4 if (i, j) in lat["nn"] else 0), N)
    dx, ox = sparse_parts(field_terms(N, np.pi / 4, "X"), N)
    psi0 = np.zeros(2 ** N, complex); psi0[0] = 1
    t, ev, psi = evolve(psi0, [(1.0, dz, oz), (1.0, dx, ox)], np.arange(0, 13, 2.0), dt_max=0.05, period=2.0)
    a = np.abs(psi) ** 2
    supp = a > 1e-10
    assert np.allclose(a[supp], a[supp].max()) and np.log2(supp.sum()) % 1 == 0, "KIM Clifford point: stabilizer"
    print("ok")
