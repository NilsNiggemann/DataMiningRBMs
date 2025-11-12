from jax import jit
from jax import numpy as jnp
import numpy as np
import itertools

@jit
def fwht(x):
    """In-place Fast Walsh–Hadamard Transform (returns transformed vector).
    Supports complex inputs (uses complex128 internally).
    Length of x must be a power of two.
    """
    x = x.astype(jnp.complex128)
    n = x.shape[0]
    h = 1
    while h < n:
        x = x.reshape(-1, 2 * h)
        left = x[:, :h]
        right = x[:, h:2 * h]
        x = jnp.concatenate([left + right, left - right], axis=1)
        x = x.reshape(n)
        h *= 2
    return x

# Helper that maps your hilbert ordering to FWHT ordering, runs FWHT and returns coeffs in cluster-matrix column order
def fwht_coeffs_in_cluster_col_order(logpsi, hilbert):
    """Compute coefficients c solving H c = logpsi using FWHT.
   Usage: fwht_coeffs_in_cluster_col_order(logpsi, hilbert)
   where logpsi is a vector of log wavefunction amplitudes in the canonical netket ordering.   
    """
    n_sites = hilbert.size
    hstates = np.array(hilbert.all_states())  # shape (n_states, n_sites) with values +1/-1

    # Map spins s in {+1,-1} -> bits b in {0,1} with convention b=1 when s==-1
    b = ((1 - hstates) // 2).astype(np.int64)
    powers = (1 << np.arange(n_sites)).astype(np.int64)
    indices = (b * powers).sum(axis=1)
    perm = np.argsort(indices)  # mapping FWHT index -> row

    psi_arr = jnp.array(logpsi)
    psi_by_index = psi_arr[perm]
    n = psi_by_index.shape[0]
    coeffs_by_index = fwht(psi_by_index) / float(n)  # indexed by subset mask
    coeffs_by_index_np = np.array(coeffs_by_index)

    # Build mask list in the SAME ORDER as optim_cluster_expansion_extreme columns
    masks = [0]
    for cluster_size in range(1, n_sites + 1):
        for comb in itertools.combinations(range(n_sites), cluster_size):
            mask = 0
            for bpos in comb:
                mask |= (1 << bpos)
            masks.append(mask)
    masks = np.array(masks, dtype=np.int64)  # length n

    # coeffs in column order: take coeffs_by_index[mask] for each column
    coeffs_col_order = coeffs_by_index_np[masks]
    return coeffs_col_order
