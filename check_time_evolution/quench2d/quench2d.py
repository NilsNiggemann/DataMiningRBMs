"""2D XXZ(+fields) quench on a small lattice: ED, cumulant (WHT) fits, NetKet operator builder.

Sites i = x*Ly + y. Basis: bit (N-1-i) of the index is site i, bit=1 <-> s=-1 (netket ordering, site 0 most significant).
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import eigh

def lattice(Lx, Ly, pbc_x=True, pbc_y=False, nnn=False):
    idx = lambda x, y: (x % Lx) * Ly + (y % Ly)
    nn, nn2 = set(), set()
    for x in range(Lx):
        for y in range(Ly):
            for dx, dy, store in [(1, 0, nn), (0, 1, nn), (1, 1, nn2), (1, -1, nn2)]:
                if store is nn2 and not nnn:
                    continue
                X, Y = x + dx, y + dy
                if (not pbc_x and not 0 <= X < Lx) or (not pbc_y and not 0 <= Y < Ly):
                    continue
                a, b = idx(x, y), idx(X, Y)
                if a != b:
                    store.add(tuple(sorted((a, b))))
    return sorted(nn), sorted(nn2)

def spins(N):
    k = np.arange(2 ** N)
    return 1 - 2 * ((k[:, None] >> (N - 1 - np.arange(N))) & 1)  # (dim, N), +-1

def hamiltonian(N, nn, nn2=(), J=1.0, Jxx=0.0, J2=0.0, J2xx=0.0, hx=0.0, hz=0.0):
    """H = sum_<ij> [J Z Z + Jxx (X X + Y Y)] + sum_<<ij>> [J2 Z Z + J2xx (XX+YY)] + hx sum X + hz sum Z"""
    dim = 2 ** N
    s = spins(N)
    k = np.arange(dim)
    H = np.zeros((dim, dim))
    diag = np.zeros(dim)
    for bonds, jz, jxy in [(nn, J, Jxx), (nn2, J2, J2xx)]:
        for a, b in bonds:
            diag += jz * s[:, a] * s[:, b]
            if jxy:
                m = (1 << (N - 1 - a)) | (1 << (N - 1 - b))
                anti = s[:, a] != s[:, b]   # XX+YY = 2(S+S- + h.c.) -> 2 on antialigned pairs
                H[k[anti], (k ^ m)[anti]] += 2 * jxy
    diag += hz * s.sum(1)
    H[k, k] += diag
    if hx:
        for i in range(N):
            H[k, k ^ (1 << (N - 1 - i))] += hx
    return H

def fwht(a):
    a = a.astype(float).copy()
    h = 1
    n = a.shape[-1]
    while h < n:
        a = a.reshape(*a.shape[:-1], -1, 2, h)
        x, y = a[..., 0, :].copy(), a[..., 1, :].copy()
        a[..., 0, :], a[..., 1, :] = x + y, x - y
        a = a.reshape(*a.shape[:-3], n)
        h *= 2
    return a / n

NOISE = -28.0
A0_FACTOR = 1.0  # A0 = A0_FACTOR * N_real for the primary mu (2.0 pins mu to the tail); mu2 always uses 2.0

def fd(x, mu, T, A0):
    return A0 / (np.exp(np.clip((x - mu) / T, -700, 700)) + 1)

def fit_mu(c, A0_factor=1.0, p0=(-10.0, 3.0)):
    """Rank vs log|c_A| Fermi-Dirac fit, dropping c_0 and symmetry zeros. A0 = A0_factor * N_real."""
    a = np.abs(c[1:])
    with np.errstate(divide='ignore'):
        lc = np.log(a)
    lc = np.sort(lc[lc > NOISE])[::-1]
    if len(lc) < 5:
        return -np.inf, np.nan, np.nan, len(lc)
    r = np.arange(1, len(lc) + 1)
    A0 = A0_factor * len(lc)
    f = lambda x, mu, T: fd(x, mu, T, A0)
    try:
        p, _ = curve_fit(f, lc, r, p0=p0, maxfev=20000)
        r2 = 1 - np.sum((r - f(lc, *p)) ** 2) / np.sum((r - r.mean()) ** 2)
        if not (lc.min() - 10 < p[0] < lc.max() + 10):  # fit ran away
            return np.nan, np.nan, np.nan, len(lc)
        return p[0], p[1], r2, len(lc)
    except RuntimeError:
        return np.nan, np.nan, np.nan, len(lc)

def order_of_masks(N):
    k = np.arange(2 ** N)
    return np.array([bin(x).count('1') for x in k])

def analyze(psi_t, N, times, cut=None):
    """Per time: mu,T,R2 (A0=N_real and 2N_real), mean WHT order, min log|psi|^2 - log(1/dim), half-cut EE, phase-mu."""
    dim = 2 ** N
    order = order_of_masks(N)
    logabs = np.log(np.abs(psi_t) + 1e-300)
    C = fwht(logabs)  # WHT index = bitmask in same bit convention
    phase = np.unwrap(np.angle(psi_t), axis=0)
    phase -= phase.mean(1, keepdims=True)
    Cp = fwht(phase)
    out = {k: [] for k in ['mu', 'T', 'R2', 'mu2', 'T2', 'mean_ord', 'minlog', 'S', 'mu_ph', 'T_ph', 'w_hi']}
    cut = cut or N // 2
    p0, p02, p0p = (-10., 3.), (-10., 3.), (-10., 3.)
    for k in range(len(times)):
        mu, T, R2, _ = fit_mu(C[k], A0_FACTOR, p0)
        mu2, T2, _, _ = fit_mu(C[k], 2.0, p02)
        mup, Tp, _, _ = fit_mu(Cp[k], 1.0, p0p)
        if np.isfinite(mu): p0 = (mu, T)
        if np.isfinite(mu2): p02 = (mu2, T2)
        if np.isfinite(mup): p0p = (mup, Tp)
        w = C[k, 1:] ** 2
        out['mu'].append(mu); out['T'].append(T); out['R2'].append(R2)
        out['mu2'].append(mu2); out['T2'].append(T2)
        out['mu_ph'].append(mup); out['T_ph'].append(Tp)
        out['mean_ord'].append((order[1:] * w).sum() / max(w.sum(), 1e-300))
        out['w_hi'].append(w[order[1:] > 4].sum() / max(w.sum(), 1e-300))  # weight beyond 4-body
        out['minlog'].append(2 * logabs[k].min() + np.log(dim))
        sv = np.linalg.svd(psi_t[k].reshape(2 ** cut, -1), compute_uv=False) ** 2
        sv = sv[sv > 1e-14]
        out['S'].append(-(sv * np.log(sv)).sum())
    return {k: np.array(v) for k, v in out.items()}

def evolve(H, psi0, times):
    E, V = eigh(H)
    c0 = V.T @ psi0
    return (np.exp(-1j * np.outer(times, E)) * c0) @ V.T

# Families on the 4x3 cylinder. psi0 = gs(-ZZ - h_i X); H_f = -ZZ + Jxx (XX+YY) - h_f X
LX, LY = 4, 3
def family_kwargs(fam, p, hf=1.0):
    return dict(A=dict(J=-1, hx=-p), B=dict(J=-1, Jxx=p, hx=-hf))[fam]

def initial_kwargs(h_i=4.0):
    return dict(J=-1, hx=-h_i)

def infidelity(a, b):
    return 1 - abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)

def nk_ham(hi, nn, J=0.0, Jxx=0.0, hx=0.0):
    """NetKet version of hamiltonian() (nn bonds, no nnn, no hz)."""
    from netket.operator.spin import sigmax, sigmay, sigmaz
    H = sum(hx * sigmax(hi, i) for i in range(hi.size))
    for a, b in nn:
        H = H + J * sigmaz(hi, a) * sigmaz(hi, b)
        if Jxx:
            H = H + Jxx * (sigmax(hi, a) * sigmax(hi, b) + sigmay(hi, a) * sigmay(hi, b))
    return H.to_local_operator() if hasattr(H, "to_local_operator") else H

def plus_state(N):
    return np.ones(2 ** N) / np.sqrt(2 ** N)

def ground_state(H):
    return eigh(H, subset_by_index=[0, 0])[1][:, 0]

if __name__ == '__main__':
    # self-check: FWHT inverse and XX+YY hermiticity / Heisenberg point leaves |+x> invariant
    N = 6
    nn, _ = lattice(3, 2, True, False)
    H = hamiltonian(N, nn, J=1.0, Jxx=1.0)
    assert np.allclose(H, H.T)
    p = plus_state(N)
    Hp = H @ p
    assert np.allclose(Hp, (p @ Hp) * p), 'Heisenberg point: |+x> should be eigenstate'
    v = np.random.rand(2 ** N)
    from scipy.linalg import hadamard
    assert np.allclose(fwht(v), hadamard(2 ** N) @ v / 2 ** N)
    import netket as nk
    nn, _ = lattice(LX, LY, True, False)
    hi = nk.hilbert.Spin(0.5, LX * LY)
    kw = dict(J=-1, Jxx=0.3, hx=-1.0)
    assert np.allclose(nk_ham(hi, nn, **kw).to_dense(), hamiltonian(LX * LY, nn, **kw)), 'NetKet/dense mismatch'
    print('ok')
