from optimization import *
import optimization
import sys
import os
from scipy.optimize import minimize
from rotation import get_U_single, apply_local_rotation_to_state, get_rotation_objective
sys.path.append("../python")

from optimization import generate_params, optimize_rbm, write_output, construct_hamiltonian_bonds_rotated
import rotation
import parseCouplings
sys.path.append("../Netket")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('index', type=int, help='The H_idx value to process')
parser.add_argument('--output_folder', type=str, default="../data/data_optimal_basis_rbm")
args = parser.parse_args()


output_folder = "../data/data_optimal_basis_rbm"
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


try:
    result = minimize(get_rotation_objective(exact_ground_state), x0=[0,0,0], bounds=[(-np.pi, np.pi)]*3)
    alpha_opt, beta_opt, gamma_opt = result.x
except Exception:
    print("Rotation optimization failed, using identity rotation.", flush=True)
    alpha_opt, beta_opt, gamma_opt = 0.0, 0.0, 0.0

H_rot = construct_hamiltonian_bonds_unitary(Jijs[H_idx], hs[H_idx], bondss[H_idx], get_U_single(alpha_opt, beta_opt, gamma_opt))
print(f"Optimizing Hamiltonian {H_idx} with rotation angles: alpha={alpha_opt}, beta={beta_opt}, gamma={gamma_opt}", flush=True)
params = generate_params(
        alpha=1,
        seed=1234,
        learning_rate=1e-3,
        n_iter=1000,
        show_progress=False,
        diag_shift=1e-4,
        alpha_opt=alpha_opt,
        beta_opt=beta_opt,
        gamma_opt=gamma_opt,
        out=f"{output_folder}/rbm_optimization_{H_idx}",
        input_file=os.path.basename(input_file)
    )
try:
    out = optimize_rbm(H_rot, params)
    write_output(H_rot, out, params)
    print("done", flush=True)
except Exception:
    print("failed", flush=True)