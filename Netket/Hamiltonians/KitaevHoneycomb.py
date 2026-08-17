import netket as nk
from netket.graph import Lattice
from netket.operator.spin import sigmax, sigmay, sigmaz
import numpy as np


def S_S(hi, i, j):
    return (sigmax(hi, i) * sigmax(hi, j)
            + sigmay(hi, i) * sigmay(hi, j)
            + sigmaz(hi, i) * sigmaz(hi, j))


def get_site(g, i, translation):
    return g.id_from_position(g.positions[i] + translation)


def _sigma(hi, comp, i):
    """Return the requested Pauli operator (comp in {'x','y','z'}) acting on site i."""
    if comp == 'x':
        return sigmax(hi, i)
    elif comp == 'y':
        return sigmay(hi, i)
    elif comp == 'z':
        return sigmaz(hi, i)
    else:
        raise ValueError(f"Unknown spin component: {comp}")


# The three complementary components to a given bond label, used for the
# Gamma (off-diagonal) term: for a gamma-bond, Gamma couples the OTHER two
# components.
_OTHER_TWO = {
    'x': ('y', 'z'),
    'y': ('z', 'x'),
    'z': ('x', 'y'),
}


def get_honeycomb_lattice(Lx, Ly, pbc=True):
    """
    Builds a honeycomb Lattice with a 2-site basis (sublattices A=0, B=1).

    The three nearest-neighbor bonds emanating from each A-site are
    classified as 'x', 'y', 'z' bonds, matching the standard Kitaev
    honeycomb model convention. Returns the lattice together with the
    three translation vectors (in the same order x, y, z) that map an
    A-site to its neighboring B-site across each bond type.
    """
    a1 = np.array([1.0, 0.0])
    a2 = np.array([0.5, np.sqrt(3) / 2])

    # Offset from sublattice A to sublattice B chosen so that the three
    # nearest-neighbor bond vectors (delta, delta - a1, delta - a2) all
    # have equal length.
    delta = np.array([0.5, 1.0 / (2 * np.sqrt(3))])
    site_offsets = [[0.0, 0.0], list(delta)]

    g = Lattice(
        basis_vectors=[a1, a2],
        site_offsets=site_offsets,
        pbc=pbc,
        extent=[Lx, Ly],
    )

    bond_vectors = {
        'z': delta,          # same unit cell
        'x': delta - a1,
        'y': delta - a2,
    }

    return g, bond_vectors


def get_KitaevHoneycomb_Hamiltonian(K=1.0, J=0.0, Gamma=0.0, GammaPrime=0.0,
                                     h=(0.0, 0.0, 0.0), Lx=3, Ly=3, pbc=True):
    """
    Constructs the Kitaev-Gamma-Heisenberg Hamiltonian on the honeycomb
    lattice, with an arbitrary (possibly zero) magnetic field:

        H = sum_{<i,j> in gamma-bonds} [
                K_gamma * S^gamma_i S^gamma_j                       (Kitaev)
              + J * S_i . S_j                                       (Heisenberg)
              + Gamma_gamma * (S^a_i S^b_j + S^b_i S^a_j)            (Gamma)
              + GammaPrime_gamma * (S^a_i S^g_j + S^g_i S^a_j
                                     + S^b_i S^g_j + S^g_i S^b_j)    (Gamma')
            ]
          + sum_i h . S_i                                           (Zeeman field)

    where for a gamma-bond (gamma in {x,y,z}), {a,b} are the other two
    spin components.

    Args:
        K: Kitaev coupling. Either a scalar (applied to all bond types)
           or a dict/tuple of length 3 giving (Kx, Ky, Kz) bond-dependent
           couplings.
        J: Heisenberg coupling (scalar, isotropic).
        Gamma: off-diagonal symmetric coupling. Scalar or (Gx, Gy, Gz).
        GammaPrime: second off-diagonal coupling. Scalar or (Gx', Gy', Gz').
        h: magnetic field vector (hx, hy, hz), arbitrary direction/magnitude.
        Lx, Ly: number of unit cells along each lattice direction.
        pbc: periodic boundary conditions.

    Returns:
        g: NetKet Lattice object
        hi: NetKet Hilbert space
        H: NetKet Hamiltonian operator
    """
    g, bond_vectors = get_honeycomb_lattice(Lx, Ly, pbc=pbc)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

    def _to_dict(val):
        if isinstance(val, dict):
            return val
        if isinstance(val, (tuple, list, np.ndarray)):
            return {'x': val[0], 'y': val[1], 'z': val[2]}
        return {'x': val, 'y': val, 'z': val}

    K_dict = _to_dict(K)
    Gamma_dict = _to_dict(Gamma)
    GammaPrime_dict = _to_dict(GammaPrime)

    # sigma_y is complex-valued, so the Hamiltonian must be built with
    # complex dtype from the outset (netket enforces strict dtype matching
    # on in-place addition of operators).
    H = nk.operator.LocalOperator(hi, dtype=complex)

    # Bond-dependent Kitaev, Heisenberg, Gamma, and Gamma' terms.
    # Only sublattice-A sites (subl == 0) are used as bond origins so that
    # every physical bond is added exactly once.
    for i in g.nodes():
        _, _, subl = g.basis_coords[i]
        if subl != 0:
            continue
        for gamma, t in bond_vectors.items():
            j = get_site(g, i, t)
            a, b = _OTHER_TWO[gamma]

            # Kitaev term
            H += K_dict[gamma] * _sigma(hi, gamma, i) * _sigma(hi, gamma, j)

            # Heisenberg term
            H += J * S_S(hi, i, j)

            # Gamma term (off-diagonal, symmetric)
            H += Gamma_dict[gamma] * (
                _sigma(hi, a, i) * _sigma(hi, b, j)
                + _sigma(hi, b, i) * _sigma(hi, a, j)
            )

            # Gamma' term (second off-diagonal coupling)
            if GammaPrime_dict[gamma] != 0.0:
                H += GammaPrime_dict[gamma] * (
                    _sigma(hi, a, i) * _sigma(hi, gamma, j)
                    + _sigma(hi, gamma, i) * _sigma(hi, a, j)
                    + _sigma(hi, b, i) * _sigma(hi, gamma, j)
                    + _sigma(hi, gamma, i) * _sigma(hi, b, j)
                )

    # Arbitrary-direction magnetic (Zeeman) field on every site.
    hx, hy, hz = h
    if hx != 0.0 or hy != 0.0 or hz != 0.0:
        for i in g.nodes():
            if hx != 0.0:
                H += hx * sigmax(hi, i)
            if hy != 0.0:
                H += hy * sigmay(hi, i)
            if hz != 0.0:
                H += hz * sigmaz(hi, i)

    return g, hi, H


from scipy import integrate

def kitaev_ground_state_energy(Kx, Ky, Kz):
    """
    Computes the ground state energy per unit cell for the Kitaev honeycomb model.
    """
    # Define the energy dispersion relation |epsilon(q)|
    def dispersion(q1, q2):
        cos_term = (Kx**2 + Ky**2 + Kz**2 + 
                    2 * Kx * Ky * np.cos(q1) + 
                    2 * Ky * Kz * np.cos(q2) + 
                    2 * Kz * Kx * np.cos(q1 + q2))
        # Ensure floating point stability near zero
        return 2 * np.sqrt(np.maximum(cos_term, 0))

    # Perform the 2D numerical integration over [-pi, pi]
    integral, _ = integrate.dblquad(
        dispersion, 
        -np.pi, np.pi,          # q2 limits
        lambda x: -np.pi,       # q1 lower limit
        lambda x: np.pi         # q1 upper limit
    )
    
    # Apply the prefactor -1 / (2 * (2*pi)^2)
    prefactor = -1.0 / (2 * (2 * np.pi)**2)
    return prefactor * integral

if __name__ == "__main__":
    # Pure (ferromagnetic) Kitaev model, no field, small periodic cluster.
    g, hi, H = get_KitaevHoneycomb_Hamiltonian(
        K=1.0, J=0.0, Gamma=0.0, GammaPrime=0.0,
        h=(0.0, 0.0, 0.0), Lx=2, Ly=2, pbc=True,
    )
    print(f"Honeycomb lattice: {g.n_nodes} sites")
    print(H)

    # Kitaev-Gamma model with a [111]-direction field.
    g, hi, H_field = get_KitaevHoneycomb_Hamiltonian(
        K=-1.0, J=0.0, Gamma=0.3, GammaPrime=0.0,
        h=(0.05, 0.05, 0.05), Lx=2, Ly=2, pbc=True,
    )
    print(H_field)