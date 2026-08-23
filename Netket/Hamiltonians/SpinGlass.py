
import netket as nk
from netket.graph import Lattice
from netket.operator.spin import sigmax, sigmaz

def get_SpinGlass(Jij):
    # Jij is a matrix of size (L,L)
    # Spin based Hilbert Space
    g = nk.graph.Chain(length=Jij.shape[0], pbc=True)
    hi = nk.hilbert.Spin(s = 1/2, N = g.n_nodes)
    # implement the spin glass Hamiltonian as \sum_{<i,j>} J_{ij} S_i S_j, where J_{ij} = - sum_n x^n_i x^n_j, and x^n_i is the i-th element of the nth solution vector
    H = 0
    for i in range(Jij.shape[0]):
            for j in range(i+1, Jij.shape[0]):
                H += Jij[i, j] * sigmaz(hi, i) @ sigmaz(hi, j)


    return g, hi, H