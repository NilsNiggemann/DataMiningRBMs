import netket as nk
from netket.graph import Lattice
from netket.operator.spin import sigmax, sigmay, sigmaz
from functools import reduce
import operator
import numpy as np


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


def _prod(ops):
    """Multiply a list of operators together with '@' (matrix product)."""
    return reduce(operator.matmul, ops)


def get_toric_code_lattice(Lx, Ly, pbc=True):
    """
    Builds the toric code layout on an Lx x Ly square lattice of vertices
    (a torus when pbc=True). Physical spin-1/2 qubits live on the EDGES of
    the lattice, not on the vertices.

    Edges are indexed as:
        horizontal edge starting at vertex i (i -> i+x): index i
        vertical edge starting at vertex i   (i -> i+y): index n_vertices + i
    so there are 2 * n_vertices edges in total.

    Returns:
        g: NetKet Lattice object over the VERTICES (kept for geometric
           reference, e.g. plotting); the physical qubits are on its edges.
        n_edges: total number of edge qubits, 2 * g.n_nodes
        stars: list of 4-tuples of edge indices, one per vertex (star operator support)
        plaquettes: list of 4-tuples of edge indices, one per face (plaquette operator support)
    """
    g = Lattice(basis_vectors=[[1, 0], [0, 1]], pbc=pbc, extent=[Lx, Ly])
    n_vertices = g.n_nodes

    def eh(i):
        """Horizontal edge from vertex i to its +x neighbor."""
        return i

    def ev(i):
        """Vertical edge from vertex i to its +y neighbor."""
        return n_vertices + i

    stars = []
    plaquettes = []
    for i in g.nodes():
        left = get_site(g, i, [-1, 0])
        down = get_site(g, i, [0, -1])
        right = get_site(g, i, [1, 0])
        up = get_site(g, i, [0, 1])

        # Star operator at vertex i: the 4 edges touching it.
        stars.append((eh(i), eh(left), ev(i), ev(down)))

        # Plaquette operator for the face with lower-left corner i:
        # bottom edge, top edge, left edge, right edge.
        plaquettes.append((eh(i), eh(up), ev(i), ev(right)))

    n_edges = 2 * n_vertices
    return g, n_edges, stars, plaquettes


def get_ToricCode_Hamiltonian(Je=1.0, Jm=1.0, h=(0.0, 0.0, 0.0), Lx=3, Ly=3, pbc=True):
    """
    Constructs the toric code Hamiltonian on a square-lattice torus, with an
    arbitrary (possibly zero) transverse magnetic field on the edge qubits:

        H = - Je * sum_s A_s - Jm * sum_p B_p - sum_e h . sigma_e

    where:
        A_s = product of sigma^x over the 4 edges touching vertex s (star operator)
        B_p = product of sigma^z over the 4 edges bounding plaquette p (plaquette operator)
        h . sigma_e = hx*sigma^x_e + hy*sigma^y_e + hz*sigma^z_e

    With Je, Jm > 0 and h = 0, the ground state is the stabilizer toric
    code state with oA_s = B_p = +1 everywhere (4-fld topologically
    degenerate on the torus). A field component along z commutes with all
    B_p but not with the A_s stabilizers (and vice versa for a field along
    x), and is the term conventionally used to drive the confinement /
    Ising-like transition out of the topological phase.

    Args:
        Je: coupling for the vertex (star) stabilizers.
        Jm: coupling for the plaquette stabilizers.
        h: transverse field vector (hx, hy, hz) applied to every edge qubit.
        Lx, Ly: number of vertices along each lattice direction.
        pbc: periodic boundary conditions (set True for the actual torus
             construction the toric code needs; open boundaries break the
             stabilizer structure at the edges).

    Returns:
        g: NetKet Lattice object over the vertices (geometric reference)
        hi: NetKet Hilbert space (spins live on the lattice's edges)
        H: NetKet Hamiltonian operator
    """
    g, n_edges, stars, plaquettes = get_toric_code_lattice(Lx, Ly, pbc=pbc)
    hi = nk.hilbert.Spin(s=1 / 2, N=n_edges)

    # sigma_y is complex-valued, so build with complex dtype from the start
    # to avoid dtype-mismatch errors on in-place addition.
    H = nk.operator.LocalOperator(hi, dtype=complex)

    for star in stars:
        H += -Je * _prod([sigmax(hi, e) for e in star])

    for plaq in plaquettes:
        H += -Jm * _prod([sigmaz(hi, e) for e in plaq])

    hx, hy, hz = h
    if hx != 0.0 or hy != 0.0 or hz != 0.0:
        for e in range(n_edges):
            if hx != 0.0:
                H += -hx * sigmax(hi, e)
            if hy != 0.0:
                H += -hy * sigmay(hi, e)
            if hz != 0.0:
                H += -hz * sigmaz(hi, e)

    return g, hi, H

def e_toric_code(Je, Jm,Lx,Ly,pbc=True):
    g, n_edges, stars, plaquettes = get_toric_code_lattice(Lx, Ly, pbc=pbc)
    res = -Je * len(stars) - Jm * len(plaquettes)
    return res

if __name__ == "__main__":
    # Pure toric code, no field, on a small torus.
    g, hi, H = get_ToricCode_Hamiltonian(Je=1.0, Jm=1.0, h=(0.0, 0.0, 0.0), Lx=2, Ly=2, pbc=True)
    print(f"Vertices: {g.n_nodes}, edge qubits: {hi.size}")
    print(H)

    # Toric code with a transverse field along z, driving the phase transition.
    g, hi, H_field = get_ToricCode_Hamiltonian(Je=1.0, Jm=1.0, h=(0.0, 0.0, 0.3), Lx=2, Ly=2, pbc=True)
    print(H_field)