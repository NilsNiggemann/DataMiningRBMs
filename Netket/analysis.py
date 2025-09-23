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

def pca_entropy(psi):
    """
    Compute the PCA entropy (SPCA) of a vector psi.

    Parameters:
        psi (array-like): Input vector (can be complex or real).

    Returns:
        float: The PCA entropy value.
    """
    psi = np.asarray(psi)
    norm = np.sum(np.abs(psi)**2)
    if norm == 0:
        return 0.0
    prob = np.abs(psi)**2 / norm
    prob = prob[prob > 0]  # Avoid log(0)
    return -np.sum(prob * np.log(prob))

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
        return 1.0
    psi = psi / norm_psi
    psi_0 = psi_0 / norm_psi_0
    fidelity = np.abs(np.vdot(psi_0, psi))**2
    return 1.0 - fidelity