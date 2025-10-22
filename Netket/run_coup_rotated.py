from optimization import *
import optimization
import sys
import os
sys.path.append("../python")

from optimization import generate_params, optimize_rbm, write_output, construct_hamiltonian_bonds_rotated
import rotation
import parseCouplings
sys.path.append("../Netket")

output_folder = "../data/data_random_rotated_rbm"
os.makedirs(output_folder, exist_ok=True)

input_file = "../python/couplings69.csv"
Jijs, hs, bondss = parseCouplings.parseCouplings(input_file)

H_idxs = [50,30,10]
n_samples = 100

for H_idx in H_idxs:
    Jij = Jijs[H_idx]
    h = hs[H_idx]
    bonds = bondss[H_idx]

    angle_range = [0, 2 * np.pi]
    alphas = np.random.uniform(*angle_range, n_samples)
    betas = np.random.uniform(*angle_range, n_samples)
    gammas = np.random.uniform(*angle_range, n_samples)

    for idx in range(n_samples):
        alpha, beta, gamma = alphas[idx], betas[idx], gammas[idx]
        H_rot = construct_hamiltonian_bonds_rotated(Jij, h, bonds, alpha, beta, gamma)
        print(f"Optimizing Hamiltonian {H_idx} with rotation angles: alpha={alpha}, beta={beta}, gamma={gamma}", flush=True)
        params = generate_params(
            alpha=1,
            seed=1234,
            learning_rate=8e-3,
            n_iter=500,
            show_progress=False,
            diag_shift=0.03,
            rot_alpha=alpha,
            rot_beta=beta,
            rot_gamma=gamma,
            optuna=False,
            out=f"{output_folder}/rbm_optimization_{H_idx}",
            input_file=os.path.basename(input_file)
        )

        out = optimize_rbm(H_rot, params)
        write_output(H_rot, out, params)
        print("done", flush=True)