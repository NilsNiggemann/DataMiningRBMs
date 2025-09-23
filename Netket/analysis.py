import numpy as np
import h5py
import pandas as pd

import numpy as np
import h5py
import pandas as pd

def load_outputs_to_dataframe(file_list):
    """
    Reads a list of HDF5 output files and returns a DataFrame with columns:
    'psi', 'psi_0', 'Delta_E' (energy difference between variational and exact).

    Parameters:
        file_list (list of str): List of HDF5 file paths.

    Returns:
        pd.DataFrame: DataFrame with columns ['psi', 'psi_0', 'Delta_E'].
    """
    data = []
    for fname in file_list:
        with h5py.File(fname, "r") as f:
            psi = f["psi"][:]
            psi_0 = f["psi_0"][:]
            en_var = f["en_var"][()] if "en_var" in f else None
            exact_ground_energy = f["exact_ground_energy"][()]
            if en_var is not None:
                delta_e = np.abs(en_var - exact_ground_energy/exact_ground_energy)
            else:
                delta_e = None
            data.append({
                "psi": psi,
                "psi_0": psi_0,
                "Delta_E": delta_e
            })
    return pd.DataFrame(data)

def mean_sign(psi):
    """
    Compute the average sign of a vector psi.

    Parameters:
        psi (array-like): Input vector (can be complex or real).

    Returns:
        float: The average sign value.
    """
    psi = np.asarray(psi)
    norm = np.sum(np.abs(psi))
    if norm == 0:
        return 0.0
    sign = np.sign(psi)
    return np.sum(sign * np.abs(psi)) / norm

def mean_phase(psi):
    """
    Compute the mean phase of a vector psi.

    Parameters:
        psi (array-like): Input vector (can be complex or real).

    Returns:
        float: The mean phase in radians (between -pi and pi).
    """
    psi = np.asarray(psi)
    if np.all(psi == 0):
        return 0.0
    phases = np.angle(psi)
    weights = np.abs(psi)
    if np.sum(weights) == 0:
        return 0.0
    return np.angle(np.sum(weights * np.exp(1j * phases)) / np.sum(weights))

def ipr(psi):
    """
    Compute the Inverse Participation Ratio (IPR) of a vector psi.

    Parameters:
        psi (array-like): Input vector (can be complex or real).

    Returns:
        float: The IPR value.
    """
    psi = np.asarray(psi)
    norm = np.sum(np.abs(psi)**2)
    if norm == 0:
        return 0.0
    prob = np.abs(psi)**2 / norm
    return np.sum(prob**2)

def pca_spectrum_from_state(psi):
    """
    Compute PCA eigenspectrum of local Sz features
    from a given state vector psi.

    Parameters
    ----------
    psi : np.ndarray
        State vector of shape (2**L,), assumed normalized.

    Returns
    -------
    eigvals : np.ndarray
        Eigenvalues of the covariance matrix (sorted descending).
    eigvecs : np.ndarray
        Corresponding eigenvectors (columns).
    cov : np.ndarray
        The covariance matrix itself.
    """
    # number of sites
    L = int(np.log2(len(psi)))
    if 2**L != len(psi):
        raise ValueError("psi length must be a power of 2")

    # probabilities
    probs = np.abs(psi)**2

    # precompute all configurations' Sz vectors
    # each config represented by bitstring of length L
    sz_configs = np.zeros((len(psi), L))
    for idx in range(len(psi)):
        bits = [(idx >> j) & 1 for j in range(L)]
        # map 0 -> +1, 1 -> -1 (spin up/down in Sz basis)
        sz_configs[idx] = [1 - 2*b for b in bits]

    # mean <Sz_i>
    mean_sz = probs @ sz_configs  # shape (L,)

    # correlations <Sz_i Sz_j>
    corr = sz_configs.T @ (probs[:, None] * sz_configs)  # shape (L, L)

    # covariance
    cov = corr - np.outer(mean_sz, mean_sz)

    # eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort in descending order
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    return eigvals, eigvecs, cov

def pca_entropy(psi):
    """
    Compute the PCA entropy (SPCA) of a vector psi.

    Parameters:
        psi (array-like): Input vector (can be complex or real).

    Returns:
        float: The PCA entropy value.
    """
    lambdas, _, _ = pca_spectrum_from_state(psi)[0]

    normalized_lambdas = lambdas / np.sum(lambdas)

    normalized_lambdas = normalized_lambdas[normalized_lambdas > 0]  # Avoid log(0)
    k = len(normalized_lambdas)
    SPCA = - 1/np.log(k) * np.sum(normalized_lambdas * np.log(normalized_lambdas))
    return SPCA


def renyi_entropy(psi, alpha=2):
    """
    Compute the Renyi entropy of a vector psi for a given order alpha.

    Parameters:
        psi (array-like): Input vector (can be complex or real).
        alpha (float): Order of the Renyi entropy (alpha > 0, alpha != 1).

    Returns:
        float: The Renyi entropy value.
    """
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    psi = np.asarray(psi)
    norm = np.sum(np.abs(psi)**2)
    if norm == 0:
        return 0.0
    prob = np.abs(psi)**2 / norm
    prob = prob[prob > 0]  # Avoid log(0) and zero division
    return 1.0 / (1.0 - alpha) * np.log(np.sum(prob**alpha))

def infidelity(psi, psi_0):
    """
    Compute the infidelity between two state vectors psi and psi_0.

    Parameters:
        psi (array-like): First state vector (can be complex or real).
        psi_0 (array-like): Second state vector (can be complex or real).

    Returns:
        float: The infidelity value (1 - |<psi_0|psi>|^2).
    """
    psi = np.asarray(psi)
    psi_0 = np.asarray(psi_0)
    # Normalize both vectors
    norm_psi = np.linalg.norm(psi)
    norm_psi_0 = np.linalg.norm(psi_0)
    if norm_psi == 0 or norm_psi_0 == 0:
        return np.nan
    psi = psi / norm_psi
    psi_0 = psi_0 / norm_psi_0
    fidelity = np.abs(np.vdot(psi_0, psi))**2
    return 1.0 - fidelity