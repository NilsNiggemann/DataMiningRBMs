from optimization import *
import sys
import os
from scipy.optimize import minimize
import tunable_Heisenberg_models as thm

sys.path.append("../python")
from optimization import generate_params, optimize_rbm, write_output


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('index', type=int, help='The run_idx value to process')
parser.add_argument('--output_folder', type=str, default="../data/data_optimal_basis_rbm")
args = parser.parse_args()


# output_folder = "../data/data_optimal_basis_rbm"
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

run_idx = args.index  # Use the command line argument

g, hi, H = thm.get_J1J2_Hamiltonian(run_idx)
#explicitly hermitize H
H = (H + H.getH()) / 2

# exact_ground_energy, exact_ground_state = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=True)
# exact_ground_state = exact_ground_state[:,0]

model_name = thm.get_model_name(run_idx)
J2_value = thm.get_J2(run_idx)
Lx, Ly = g.extent

print(f"Optimizing Hamiltonian {run_idx} ", flush=True)
params = generate_params(
        alpha=1,
        seed=1234,
        learning_rate=1e-3,
        n_iter=1000,
        show_progress=False,
        diag_shift=1e-4,
        run_idx=run_idx,
        Lx = Lx,
        Ly = Ly,
        model_name=model_name,
        J1=1.0,
        J2=J2_value,
        out=f"{output_folder}/rbm_optimization_{run_idx}",
)
print(f"Parameters: {params}", flush=True)
try:
    out = optimize_rbm(H, params)
    write_output(H, out, params, k_states_save=3)
    print("done", flush=True)
except Exception:
    print("failed", flush=True)