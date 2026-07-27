import netket as nk
from netket.graph import Lattice
from netket.operator.spin import sigmax, sigmay, sigmaz
import jax.numpy as jnp
import numpy as np

def get_TFI_Hamiltonian(L,Lambda,pbc=False):
    """
    Constructs the ferromagnetic rotated TFI model Hamiltonian on a chain.
    Strength of the interaction is set to 1, and the transverse field strength is Lambda.

    H = - sum_{<i,j>} (- S^z_i S^z_j  + Lambda S^x_i)

    Args:
        L: Length of the chain
        Lambda: Strength of the transverse field
        pbc: Whether to use periodic boundary conditions (default: False)

    Returns:
        g: NetKet Lattice object
        hi: NetKet Hilbert space
        H: NetKet Hamiltonian operator
    """
    g = nk.graph.Chain(length=L,pbc=pbc)
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
    
    H = sum(- Lambda * sigmax(hi,i) for i in g.nodes())

    for (i,j) in g.edges():
        # Right neighbor
        H += -sigmaz(hi,i)@sigmaz(hi,j)
        
    return g, hi, H