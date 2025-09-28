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

def TFI2D(g, J=1.0, h=1.0):
    hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    H = -J * sum(nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, j) for (i, j) in g.edges())
    H += -h * sum(nk.operator.spin.sigmax(hi, i) for i in g.nodes())
    return H

# Load couplings
Jij, h, bonds = parseCouplings.parseCouplings(input_file)
indices = np.random.choice(len(h), size=100, replace=False)

for idx_num, i in enumerate(indices):
    epsilon = np.random.uniform(0.01, 0.3)
    H_ran = construct_hamiltonian_bonds(Jij[i], h[i], bonds[i])
    H2 = TFI2D(g, J=1.0, h=0.5)
    H = epsilon * H_ran + H2

    exact_ground_energy, exact_ground_state = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=True)
    print(f"Hamiltonian {idx_num+1}: index={i}, epsilon={epsilon:.3f}, Exact ground state energy: {exact_ground_energy[0]}")

    params = generate_params(
        alpha=1,
        seed=1234,
        learning_rate=3e-2,
        n_iter=1000,
        show_progress=True,
        out=f"data_rand/rbm_optimization_{i}_eps_{epsilon:.3f}",
        epsilon=epsilon,
    )
    output_file = f"data_rand/rbm_optimization_{i}_eps_{epsilon:.3f}.npz"
    if os.path.exists(output_file):
        print(f"Skipping Hamiltonian {idx_num+1}: index={i}, output file already exists.")
        continue

    out = optimize_rbm(H, params)
    write_output(H, out, params)