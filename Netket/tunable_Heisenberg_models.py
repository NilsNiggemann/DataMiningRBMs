import netket as nk
from netket.graph import Lattice
from netket.operator.spin import sigmax, sigmay, sigmaz
import json
import jax.numpy as jnp
import numpy as np

MODEL_CAPACITY = 1000 # number of models per index
# 0034 would mean the 0th model (square lattice) and the 34th Hamiltonian in that model

def deconstruct_run_index(run_idx):
    model_idx = run_idx // MODEL_CAPACITY # get the first digit of the run index which corresponds to the model index
    J2_idx = run_idx % MODEL_CAPACITY  # rest of the digits correspond to the Hamiltonian index
    return model_idx, J2_idx

def get_J2(model_idx_or_run_idx):
    """
    Maps model index to J2 value for tunable frustration models.
    For example, model_idx 0 -> J2 = 0.0, model_idx 1 -> J2 = 0.1, ..., model_idx 10 -> J2 = 1.0
    """
    return model_idx_or_run_idx % MODEL_CAPACITY * 0.05

def get_model_name(run_idx):
    model_idx, _ = deconstruct_run_index(run_idx)
    if model_idx == 0:
        return "J1-J2 Square"
    elif model_idx == 1:
        return "J1-J2 Triangular"
    elif model_idx == 2:
        return "J1-J2 Kagome"
    else:
        raise ValueError(f"Unknown model index: {model_idx}")
    
def get_J1J2_Hamiltonian(run_idx, Lx='auto', Ly='auto', pbc=True):
    model_idx, J2_idx = deconstruct_run_index(run_idx)
    J2 = get_J2(J2_idx)
    if model_idx == 0:
        Lx = Lx if Lx != 'auto' else 3
        Ly = Ly if Ly != 'auto' else 4
        g, hi, H = get_J1J2Square_Hamiltonian(J1=1.0, J2=J2, Lx=Lx, Ly=Ly, pbc=pbc)
    elif model_idx == 1:
        Lx = Lx if Lx != 'auto' else 3
        Ly = Ly if Ly != 'auto' else 4
        g, hi, H = get_J1J2triangular_Hamiltonian(J1=1.0, J2=J2, Lx=Lx, Ly=Ly, pbc=pbc)
    elif model_idx == 2:
        Lx = Lx if Lx != 'auto' else 2
        Ly = Ly if Ly != 'auto' else 2
        g, hi, H = get_J1J2kagome_Hamiltonian(J1=1.0, J2=J2, Lx=Lx, Ly=Ly, pbc=pbc)
    return g, hi, H

def S_S(hi, i, j):
    return (sigmax(hi, i) * sigmax(hi, j) + sigmay(hi, i) * sigmay(hi, j) + sigmaz(hi, i) * sigmaz(hi, j))

def get_site(g,i,translation):
    return g.id_from_position(g.positions[i] + translation)

def get_J1J2Square_Hamiltonian(J1, J2, Lx, Ly, pbc = True):
    """
    Constructs the J1-J2 Heisenberg Hamiltonian on a square lattice.
    Returns:
        g: NetKet Lattice object
        hi: NetKet Hilbert space
        H: NetKet Hamiltonian operator
    """
    g = Lattice(basis_vectors=[[1, 0], [0, 1]], pbc=pbc, extent=[Lx, Ly])
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
    H = 0
    for (i,j) in g.edges():
        # Right neighbor
        H += J1 * S_S(hi, i, j)
    for i in g.nodes():
        # Diagonal next nearest neighbors
        NNs = np.where(g.distances()[i] == 2)
        for j in NNs[0]:
            H += J2 * S_S(hi, i, j)
    return g, hi, H

def get_J1J2triangular_Hamiltonian(J1, J2, Lx, Ly, pbc = True):
    """
    Constructs a Heisenberg Hamiltonian on a triangular lattice. J2 is a tunable frustrated bond, see https://doi.org/10.1038/s41467-020-15402-w
    Returns:
        g: NetKet Lattice object
        hi: NetKet Hilbert space
        H: NetKet Hamiltonian operator
    """

    a1 = np.array([1, 0])
    a2 = np.array([0.5, np.sqrt(3)/2])

    J1_bonds = [a1, a2]
    J2bonds = [np.array([0.5, -np.sqrt(3)/2])]
    
    g = Lattice(
        basis_vectors=[a1, a2],
        pbc=pbc,
        extent=[Lx, Ly]
    )
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
    H = 0
    for i in g.nodes():
        for t in J1_bonds:
            j = get_site(g, i, t)
            H += J1 * S_S(hi, i, j)
        for t in J2bonds:
            j = get_site(g, i, t)
            H += J2 * S_S(hi, i, j)

    return g, hi, H


def get_J1J2kagome_Hamiltonian(J1, J2, Lx, Ly, pbc = True):
    """
    Constructs a Heisenberg Hamiltonian on a kagome lattice. J2 is a tunable frustrated bond, see https://doi.org/10.1038/s41467-020-15402-w
    Returns:
        g: NetKet Lattice object
        hi: NetKet Hilbert space
        H: NetKet Hamiltonian operator
    """
    # Kagome lattice: 3-site basis, triangular Bravais lattice
    a1 = [1, 0]
    a2 = [0.5, np.sqrt(3)/2]
    site_offsets = [[0, 0], [1/2, 0], [1/4, np.sqrt(3)/4]]

    g = Lattice(
        basis_vectors=[a1, a2],
        site_offsets=site_offsets,
        pbc=pbc,
        extent=[Lx, Ly]
    )


    J1_bonds = [a1, a2]
    J2bonds = [0.5*np.array([0.5, -np.sqrt(3)/2])]
    
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
    H = 0
    
    for i in g.nodes():
        for t in J1_bonds:
            j = get_site(g, i, t)
            H += J1 * S_S(hi, i, j)

        _,_,subl = g.basis_coords[i]
        if subl == 0: #0 type sites do not have a J2 bond
            continue
        for t in J2bonds:
            j = get_site(g, i, t)
            H += J2 * S_S(hi, i, j)

    return g, hi, H
