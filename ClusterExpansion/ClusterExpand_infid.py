# %%
import numpy as np 
import sys, os
sys.path.append('../Netket/')
import analysis
from analysis import std_phase, ipr, pca_entropy, renyi_entropy, mean_amplitude, uniform_state_overlap, infidelity

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import netket as nk
from cluster_expansion import *

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


import argparse

parser = argparse.ArgumentParser(description="Cluster expansion infidelity analysis")
parser.add_argument("file_start", type=int, help="Start index of files")
parser.add_argument("num_files", type=int, help="Number of files to process")
args = parser.parse_args()

file_start = args.file_start
numFiles = args.num_files


hypotheses = {
    "std_phase" : std_phase,
    "IPR" : ipr,
    "SPCA" : pca_entropy,
    "Renyi_2" : renyi_entropy,
    "uniform_state_overlap" : uniform_state_overlap,
    "mean_amplitude" : mean_amplitude,
}

data_root = '..'
print(f"running for files {file_start} to {file_start+numFiles-1}")

h5_files_opt = np.sort([os.path.join(f"{data_root}/data/data_optimal_basis_rbm", f) for f in os.listdir(f'{data_root}/data/data_optimal_basis_rbm') if f.endswith('.h5')])[file_start:file_start+numFiles]

df_opt = analysis.load_outputs_to_dataframe(h5_files_opt, load_eigenstates=False)
df_opt = analysis.attach_hypotheses_fields(df_opt, hypotheses)
df_opt["idx"] = df_opt["file"].apply(lambda x: int(os.path.basename(x).split('_')[2]))
print(len(df_opt))



n_sites_test = 16
hilb_test = nk.hilbert.Spin(0.5, n_sites_test)
compr_idx_list = sorted(np.array(list(set(np.logspace(1, 16, 100, base=2, dtype=int)))))  

def compute_infidelity_matrices(df, hilb, compr_idx_list):
    n_rows = len(df)
    n_compr = len(compr_idx_list)
    infidels_exact_opt_mat = np.zeros((n_rows, n_compr))
    infidels_RBM_opt_mat = np.zeros((n_rows, n_compr))
    for i, row in df.iterrows():
        psi_test_exact = np.array(row['psi_0'])
        psi_test_RBM = np.array(row['psi'])
        cluster_coeffs_test_exact = fwht_coeffs_in_cluster_col_order(psi_test_exact, hilb)
        cluster_coeffs_test_RBM = fwht_coeffs_in_cluster_col_order(psi_test_RBM, hilb)
        prepare_fwht_meta_cached(hilb)
        for j, compr_idx in enumerate(compr_idx_list):
            psi_rec_exact = compress_and_reconstruct_cached(cluster_coeffs_test_exact, compr_idx, hilb)
            psi_rec_RBM = compress_and_reconstruct_cached(cluster_coeffs_test_RBM, compr_idx, hilb)
            infidels_exact_opt_mat[i, j] = infidelity(psi_rec_exact, psi_test_exact)
            infidels_RBM_opt_mat[i, j] = infidelity(psi_rec_RBM, psi_test_exact)
    return infidels_exact_opt_mat, infidels_RBM_opt_mat

infidels_exact_opt_mat, infidels_RBM_opt_mat = compute_infidelity_matrices(df_opt, hilb_test, compr_idx_list)

# %%
import h5py
outfile = f'../data/cluster_expansion_analysis_exp/expansion_infidelities_{file_start}_{file_start+numFiles-1}.h5'

import os
os.makedirs(os.path.dirname(outfile), exist_ok=True)
with h5py.File(outfile, 'w') as f:
    f.create_dataset('infidels_exact_opt_mat', data=infidels_exact_opt_mat)
    f.create_dataset('infidels_RBM_opt_mat', data=infidels_RBM_opt_mat)
    f.create_dataset('compr_idx_list', data=compr_idx_list)
    f.create_dataset('idxs', data=df_opt.idx.values)
