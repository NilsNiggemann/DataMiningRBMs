import numpy as np
import netket as nk
from netket.operator.spin import sigmax, sigmaz
from scipy.linalg import eigh


def build_TFI_hamiltonian(L, g, J=1.0, pbc=False):
    """
    Constructs the transverse-field Ising Hamiltonian on a chain:

        H = -J * sum_<i,j> Z_i Z_j - g * sum_i X_i

    Args:
        L: number of sites
        g: transverse field strength
        J: ZZ coupling strength
        pbc: periodic boundary conditions (False = open boundaries)

    Returns:
        g_graph: NetKet Chain graph
        hi: NetKet Hilbert space
        H: NetKet Hamiltonian operator
    """
    g_graph = nk.graph.Chain(length=L, pbc=pbc)
    hi = nk.hilbert.Spin(s=1 / 2, N=g_graph.n_nodes)

    H = sum(-g * sigmax(hi, i) for i in g_graph.nodes())
    for (i, j) in g_graph.edges():
        H += -J * sigmaz(hi, i) * sigmaz(hi, j)

    return g_graph, hi, H


def build_TFI_hamiltonian_2(L, g, J=1.0, J2=0.5,gz=0.0, theta = 0.0,pbc=False):
    """
    Constructs the transverse-field Ising Hamiltonian on a chain:

        H = -J * sum_<i,j> Z_i Z_j - g * sum_i X_i

    Args:
        L: number of sites
        g: transverse field strength
        J: ZZ coupling strength
        J2: next-nearest neighbor ZZ coupling strength
        gz: longitudinal field strength
        theta: angle for rotation of the spin operators
        pbc: periodic boundary conditions (False = open boundaries)

    Returns:
        g_graph: NetKet Chain graph
        hi: NetKet Hilbert space
        H: NetKet Hamiltonian operator
    """
    def sz(hi, i):
        return np.cos(theta) * sigmaz(hi, i) + np.sin(theta) * sigmax(hi, i)

    def sx(hi, i):
        return -np.sin(theta) * sigmaz(hi, i) + np.cos(theta) * sigmax(hi, i)

    g_graph = nk.graph.Chain(length=L, pbc=pbc)
    hi = nk.hilbert.Spin(s=1 / 2, N=g_graph.n_nodes)

    H = sum(-g * sx(hi, i) for i in g_graph.nodes())
    H += sum(-gz * sz(hi, i) for i in g_graph.nodes())
    for (i, j) in g_graph.edges():
        H += -J * sz(hi, i) * sz(hi, j)

    for i in range(L):
        if pbc:
            j = (i + 2) % L
            H += -J2 * sz(hi, i) * sz(hi, j)
        else:
            if i + 2 < L:
                j = i + 2
                H += -J2 * sz(hi, i) * sz(hi, j)

    return g_graph, hi, H


def ground_state(H):
    """Ground state via Lanczos ED. Returns (E0, psi0)."""
    evals, eigenvectors = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=True)
    return evals[0], eigenvectors[:, 0]


def full_diagonalization(H):
    """Full exact diagonalization of the (Hermitian) dense Hamiltonian.

    Returns (evals, evecs) with evecs[:, n] the n-th eigenvector.
    """
    H_dense = np.array(H.to_dense())
    evals, evecs = eigh(H_dense)
    return evals, evecs


def entanglement_entropy(psi, L, LA=None):
    """Von Neumann entanglement entropy of a bipartition of a chain of L spins.

    Relies on NetKet's Spin Hilbert space basis ordering, where site 0 is the
    most significant digit of the flat state index (confirmed via
    `hi.numbers_to_states`), so a plain reshape splits the chain into the
    first LA sites (A) and the remaining L-LA sites (B).

    Args:
        psi: state vector, shape (2**L,)
        L: number of sites
        LA: size of subsystem A (default: L // 2, i.e. the half-chain cut)

    Returns:
        float: S = -sum_i p_i log(p_i), p_i the Schmidt-value probabilities
    """
    if LA is None:
        LA = L // 2
    mat = psi.reshape(2 ** LA, 2 ** (L - LA))
    s = np.linalg.svd(mat, compute_uv=False)
    p = s ** 2
    p = p[p > 1e-14]
    p /= p.sum()
    return -np.sum(p * np.log(p))


def evolve_state(evals, evecs, psi0, times):
    """Exact unitary time evolution of psi0 under H, in the eigenbasis of H.

    Args:
        evals, evecs: full spectrum of H, from full_diagonalization(H)
        psi0: initial state, shape (dim,)
        times: array of times, shape (n_times,)

    Returns:
        psi_t: array of shape (n_times, dim)
    """
    c0 = evecs.conj().T @ psi0
    phases = np.exp(-1j * np.outer(times, evals))  # (n_times, dim)
    psi_t = (phases * c0) @ evecs.T
    return psi_t
