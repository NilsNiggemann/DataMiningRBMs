import numpy as np
import h5py
import pandas as pd

import numpy as np
import h5py
import pandas as pd
import os
from scipy.stats import pearsonr

def _arr_to_num(arr):
    if np.isscalar(arr):
        return arr
    elif np.ndim(arr) != 0 and len(arr) == 1:
        return arr[0]
    else:
        raise ValueError("Input is not a scalar or single-element array.")
    
def get_h5_files(path,suffix=".h5"):
    h5_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(suffix)]
    return h5_files

def read_folder_to_dataframe(folder, suffix=".h5"):
    """
    Reads all HDF5 files in a folder and returns a DataFrame with columns:
    'psi', 'psi_0', 'Delta_E' (energy difference between variational and exact).

    Parameters:
        folder (str): Path to the folder containing HDF5 files.
        suffix (str): Suffix of the files to read (default is ".h5").

    Returns:
        pd.DataFrame: DataFrame with columns ['psi', 'psi_0', 'Delta_E'].
    """

    file_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(suffix)]
    return load_outputs_to_dataframe(file_list)

def load_outputs_to_dataframe(file_list,attach_attributes=True,load_eigenstates=True):
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
            try:
                exact_energies = f["exact_energies"][:] 
            except KeyError:
                exact_energies = None

            if load_eigenstates:
                try: 
                    exact_eigenstates = f["exact_eigenstates"][:]
                except KeyError:
                    exact_eigenstates = None
            else:
                exact_eigenstates = None

            exact_ground_energy = _arr_to_num(f["exact_ground_energy"][()])
            infid = infidelity(psi, psi_0)
            if en_var is not None:
                delta_e = np.abs((en_var - exact_ground_energy)/exact_ground_energy)
            else:
                delta_e = None

            val_dict ={
                "psi": psi,
                "psi_0": psi_0,
                "Delta_E": delta_e,
                "E_exact": exact_ground_energy,
                "E_var": en_var,
                "infidelity": infid,
                "exact_energies": exact_energies,
                "exact_eigenstates": exact_eigenstates,
                "file" : fname
            }

            if attach_attributes:
                attrs = dict(f.attrs)
                val_dict.update(attrs)

            data.append(val_dict)

    df = pd.DataFrame(data)
    return df

def _load_single_h5_file(fname, attach_attributes=True, load_eigenstates=True):
    """
    Helper function to load a single HDF5 file. Used by load_outputs_to_dataframe_mult_thread.
    
    Parameters:
        fname (str): Path to HDF5 file.
        attach_attributes (bool): Whether to attach HDF5 file attributes to the output dict.
        load_eigenstates (bool): Whether to load exact eigenstates.
    
    Returns:
        dict: Dictionary with loaded data.
    """
    with h5py.File(fname, "r") as f:
        psi = f["psi"][:]
        psi_0 = f["psi_0"][:]
        en_var = f["en_var"][()] if "en_var" in f else None
        try:
            exact_energies = f["exact_energies"][:] 
        except KeyError:
            exact_energies = None

        if load_eigenstates:
            try: 
                exact_eigenstates = f["exact_eigenstates"][:]
            except KeyError:
                exact_eigenstates = None
        else:
            exact_eigenstates = None

        exact_ground_energy = _arr_to_num(f["exact_ground_energy"][()])
        infid = infidelity(psi, psi_0)
        if en_var is not None:
            delta_e = np.abs((en_var - exact_ground_energy)/exact_ground_energy)
        else:
            delta_e = None

        val_dict ={
            "psi": psi,
            "psi_0": psi_0,
            "Delta_E": delta_e,
            "E_exact": exact_ground_energy,
            "E_var": en_var,
            "infidelity": infid,
            "exact_energies": exact_energies,
            "exact_eigenstates": exact_eigenstates,
            "file" : fname
        }

        if attach_attributes:
            attrs = dict(f.attrs)
            val_dict.update(attrs)

        return val_dict

def load_outputs_to_dataframe_mult_thread(file_list, attach_attributes=True, load_eigenstates=True, num_workers=4):
    """
    Reads a list of HDF5 output files in parallel using multithreading and returns a DataFrame.
    Useful for speeding up I/O when loading many files.

    Parameters:
        file_list (list of str): List of HDF5 file paths.
        attach_attributes (bool): Whether to attach HDF5 file attributes to the output dict.
        load_eigenstates (bool): Whether to load exact eigenstates.
        num_workers (int): Number of worker threads to use (default=4).

    Returns:
        pd.DataFrame: DataFrame with columns ['psi', 'psi_0', 'Delta_E', ...].
    """
    from concurrent.futures import ThreadPoolExecutor
    
    data = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(_load_single_h5_file, fname, attach_attributes, load_eigenstates)
            for fname in file_list
        ]
        for future in futures:
            data.append(future.result())

    df = pd.DataFrame(data)
    return df

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
        return np.nan
    phases = np.angle(psi)
    return np.mean(phases)

def std_phase(psi):
    """
    Compute the standard deviation of the phase of a vector psi.

    Parameters:
        psi (array-like): Input vector (can be complex or real).

    Returns:
        float: The standard deviation of the phase.
    """
    psi = np.asarray(psi)
    if np.all(psi == 0):
        return np.nan
    phases = np.angle(psi)
    return np.std(phases)

def ipr(psi):
    """
    Compute the Inverse Participation Ratio (IPR) of a vector psi.

    Parameters:
        psi (array-like): Input vector (can be complex or real).

    Returns:
        float: The IPR value.
    """
    psi = np.asarray(psi)
    return np.sum(np.abs(psi)**4)

def log_ipr(psi):
    return np.log(ipr(psi))


def compute_pearson_correlation(series1, series2):
    correlation, p_value = pearsonr(series1, series2)
    return correlation

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
    lambdas, _, _ = pca_spectrum_from_state(psi)

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
    if alpha ==1:
        raise ValueError("alpha must be != 1")
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

def uniform_state_overlap(psi):
    uniform_psi = np.ones(len(psi)) / np.sqrt(len(psi))
    overlap = np.abs(np.vdot(uniform_psi, psi))**2
    return overlap

def read_h5_attributes(filename):
    """
    Reads and returns all attributes from the root of an HDF5 file.
    """
    with h5py.File(filename, "r") as f:
        attrs = dict(f.attrs)
    return attrs

def attach_hypotheses_fields(df, hypotheses):
    """
    Computes new columns for the dataframe using the provided hypotheses functions.
    Each function in hypotheses is applied to the 'psi' column of the dataframe.
    """
    for name, func in hypotheses.items():
        df[name] = df["psi_0"].apply(func)
    return df

def mean_amplitude(psi):
    """
    Compute the mean amplitude of a vector psi.

    Parameters:
        psi (array-like): Input vector (can be complex or real).

    Returns:
        float: The mean amplitude.
    """
    psi = np.asarray(psi)
    return np.mean(np.abs(psi))

def uniform_state_overlap(psi):
    uniform_psi = np.ones(len(psi)) / np.sqrt(len(psi))
    overlap = np.abs(np.vdot(uniform_psi, psi))**2
    return overlap