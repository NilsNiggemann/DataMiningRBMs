from optimization import *
import optimization
import sys
import importlib
importlib.reload(optimization)
sys.path.append("../python")
import parseCouplings
sys.path.append("../Netket")

from optimization import generate_params, optimize_rbm, write_output
import os

input_file = "../python/couplings69.csv"
# Set up the 2D lattice for TFI
g = nk.graph.Hypercube(length=4, n_dim=2, pbc=True)

# Load couplings
Jij, h, bonds = parseCouplings.parseCouplings(input_file)
indices = np.random.choice(len(h), size=100, replace=False)

for idx_num, i in enumerate(indices):
    H_ran = construct_hamiltonian_bonds(Jij[i], 0*h[i], bonds[i])
    H = H_ran

    exact_ground_energy, exact_ground_state = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=True)
    print(f"Hamiltonian {idx_num+1}: index={i}, Exact ground state energy: {exact_ground_energy[0]}")

    params = generate_params(
        alpha=1,
        seed=1234,
        learning_rate=3e-2,
        n_iter=1000,
        show_progress=False,
        out=f"../data/data_rand_h0/rbm_optimization_{i}",
    )
    output_file = params["out"] + ".log"
    if os.path.exists(output_file):
        print(f"Skipping Hamiltonian {idx_num+1}: index={i}, output file already exists.")
        continue

    out = optimize_rbm(H, params)
    write_output(H, out, params)