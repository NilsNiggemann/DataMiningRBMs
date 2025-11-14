from jax import jit
from jax import numpy as jnp
import jax
import numpy as np
import itertools
from functools import lru_cache, partial
import netket as nk
from scipy.special import comb



def return_parity(bitstring):
    '''
    bitstring is a cluster, ex s1s2, or s1s2s3s4 etc
    type of bitstring: jnp.array, dtype=jnp.int8
    returns +1 if even parity, -1 if odd parity
    '''
    par = jnp.prod(bitstring,axis=-1,dtype=jnp.int8) #Since these are spin states +1 and -1
    return par 

def naive_cluster_expansion_mat(hilbert):
    '''Compute cluster expansion matrix up to a given maximum cluster size.
    for now only works for spin-1/2 systems
    '''
    # n_sites = hilbert.n_sites
    n_sites = hilbert.size
    hstates = hilbert.all_states()
    matsize = 2**n_sites
    mat = jnp.ones((matsize, matsize),dtype=jnp.int8)
    for state_idx, state in enumerate(hstates):
        start_idx = 1 #First column is all ones, so start from second column
        for cluster_size in jnp.arange(1, n_sites + 1):
            clusters = jnp.array(list(itertools.combinations(state, cluster_size)))
            rowvals = return_parity(clusters)
            mat = mat.at[state_idx, start_idx: start_idx + int(comb(n_sites, cluster_size))].set(rowvals)
            start_idx += int(comb(int(n_sites), cluster_size))
    return mat

##########################################################

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

##########################################################################
############################# COMPRESSION ################################
##########################################################################


@lru_cache(maxsize=8)
def _build_fwht_meta_by_n(n_sites):
    """Build and cache masks/perm tuples for a given n_sites.

    Returns: (masks_tuple, inv_perm_tuple, n)
    """
    # build masks (column order) as Python ints
    masks = [0]
    for cluster_size in range(1, n_sites + 1):
        for comb in itertools.combinations(range(n_sites), cluster_size):
            mask = 0
            for bpos in comb:
                mask |= (1 << bpos)
            masks.append(int(mask))

    # compute perm/inv_perm using a temporary hilbert (same ordering as your fwht helper)
    _hilb = nk.hilbert.Spin(0.5, N=n_sites)
    hstates = np.array(_hilb.all_states())
    b = ((1 - hstates) // 2).astype(np.int64)
    powers = (1 << np.arange(n_sites)).astype(np.int64)
    indices = (b * powers).sum(axis=1)
    perm = np.argsort(indices)
    inv_perm = np.argsort(perm)

    masks_tuple = tuple(masks)
    inv_perm_tuple = tuple(int(x) for x in inv_perm.tolist())
    n = int(2 ** n_sites)
    return masks_tuple, inv_perm_tuple, n


def prepare_fwht_meta_cached(hilbert):
    """User-facing helper: returns cached meta for the hilbert's n_sites."""
    return _build_fwht_meta_by_n(int(hilbert.size))


# Jitted selection and reconstruction routines (expect Python tuples for masks/inv_perm)
# make k (num_kept) a static arg to avoid tracer hashing inside argpartition
@partial(jit, static_argnums=(1,))
def _get_topk_indices_jit(coefs, k):
    # k is expected to be a Python int (static)
    k = max(int(k), 1)
    absvals = jnp.abs(coefs)
    part = jnp.argpartition(-absvals, k - 1)[:k]
    ordered = part[jnp.argsort(-absvals[part])]
    return ordered


# Mark masks_tuple, inv_perm_tuple and n as static to avoid hashing traced objects
@partial(jit, static_argnums=(1, 2, 3))
def _reconstruct_from_trunc_jit(trunc, masks_tuple, inv_perm_tuple, n):
    # convert static tuples to jnp arrays (these will be constants in compiled function)
    masks_jnp = jnp.array(masks_tuple, dtype=jnp.int32)
    inv_perm_jnp = jnp.array(inv_perm_tuple, dtype=jnp.int32)

    coeffs_by_index = jnp.zeros(n, dtype=jnp.complex128)
    coeffs_by_index = coeffs_by_index.at[masks_jnp].set(trunc)
    logpsi_by_index = fwht(coeffs_by_index)
    logpsi = logpsi_by_index[inv_perm_jnp]
    psi = jnp.exp(logpsi)
    return psi


# Mark num_kept, masks_tuple, inv_perm_tuple and n as static arguments
@partial(jit, static_argnums=(1, 2, 3, 4))
def compress_and_reconstruct_cached_jit(full_coeffs, num_kept, masks_tuple, inv_perm_tuple, n):
    # num_kept is static (Python int); pass it to the static _get_topk_indices_jit
    top_idx = _get_topk_indices_jit(full_coeffs, num_kept)
    trunc = jnp.zeros_like(full_coeffs).at[top_idx].set(full_coeffs[top_idx])
    psi = _reconstruct_from_trunc_jit(trunc, masks_tuple, inv_perm_tuple, n)
    return psi


# Convenience Python wrapper that uses the cache and returns a NumPy array
def compress_and_reconstruct_cached(full_coeffs, num_kept, hilbert):
    masks_tuple, inv_perm_tuple, n = prepare_fwht_meta_cached(hilbert)
    full_coeffs_jnp = jnp.array(full_coeffs, dtype=jnp.complex128)
    # ensure num_kept is a Python int when passed
    psi_jnp = compress_and_reconstruct_cached_jit(full_coeffs_jnp, int(num_kept), masks_tuple, inv_perm_tuple, n)
    return np.array(jax.device_get(psi_jnp))


