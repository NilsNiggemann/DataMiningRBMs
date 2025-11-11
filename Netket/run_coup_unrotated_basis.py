from optimization import *
import optimization
import sys
import os
from scipy.optimize import minimize
# from rotation import get_U_single, apply_local_rotation_to_state, get_rotation_objective
sys.path.append("../python")

from optimization import generate_params, optimize_rbm, write_output, construct_hamiltonian_bonds_rotated
import rotation
import parseCouplings


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('index', type=int, help='The H_idx value to process')
parser.add_argument('--output_folder', type=str, default="../data/data_unrotated_basis_rbm")
args = parser.parse_args()


# output_folder = "../data/data_optimal_basis_rbm"
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

input_file = "../python/couplings69.csv"
Jijs, hs, bondss = parseCouplings.parseCouplings(input_file)

H_idx = args.index  # Use the command line argument

Jij = Jijs[H_idx]
h = hs[H_idx]
bonds = bondss[H_idx]

H_ran = construct_hamiltonian_bonds(Jijs[H_idx], hs[H_idx], bondss[H_idx])
H = H_ran

exact_ground_energy, exact_ground_state = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=True)
exact_ground_state = exact_ground_state[:,0]

H_rot = construct_hamiltonian_bonds(Jijs[H_idx], hs[H_idx], bondss[H_idx])
params = generate_params(
        alpha=1,
        seed=1234,
        learning_rate=1e-3,
        n_iter=1000,
        show_progress=False,
        diag_shift=1e-4,
        out=f"{output_folder}/rbm_optimization_{H_idx}",
        input_file=os.path.basename(input_file)
    )
try:
    out = optimize_rbm(H_rot, params)
    write_output(H_rot, out, params)
    print("done", flush=True)
except Exception:
    print("failed", flush=True)