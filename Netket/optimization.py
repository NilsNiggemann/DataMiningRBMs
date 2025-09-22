import numpy as np
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
def construct_hamiltonian(Jijalphabeta, h):
    N = h.shape[1]
    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    pauli = [sigmax, sigmay, sigmaz]

    # Interaction terms
    interaction_terms = [
        Jijalphabeta[alpha, beta, i, j] * pauli[alpha](hilbert,i) * pauli[beta](hilbert,j)
        for i in range(N)
        for j in range(i,N)
        for alpha in range(3)
        for beta in range(3)
        if np.abs(Jijalphabeta[alpha, beta, i, j]) > 1e-12
    ]

    # Local field terms
    field_terms = [
        h[alpha, i] * pauli[alpha](hilbert, i)
        for i in range(N)
        for alpha in range(3)
        if np.abs(h[alpha, i]) > 1e-12
    ]

    ha = sum(interaction_terms, nk.operator.LocalOperator(hilbert)) + sum(field_terms, nk.operator.LocalOperator(hilbert))
    ha = 0.5*(ha + ha.H)  # Ensure Hermiticity
    return ha

def optimize_rbm(H, params):
    # Hilbert space from Hamiltonian
    hilbert = H.hilbert
    alpha = params.get("alpha", 1)  # Hidden unit density

    # Define the RBM model
    model = nk.models.RBM(alpha=alpha, use_visible_bias=True, use_hidden_bias=True)

    # Extract kwargs from params with defaults
    learning_rate = params.get("learning_rate", 0.01)
    n_iter = params.get("n_iter", 200)

    # Variational state
    vstate = nk.vqs.FullSumState(hilbert, model)

    # Optimizer
    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)

    # Stochastic Reconfiguration
    diag_shift = params.get("diag_shift", 0.01)
    sr = nk.optimizer.SR(diag_shift=diag_shift)

    # VMC driver
    driver = nk.VMC(hamiltonian=H, optimizer=optimizer, variational_state=vstate, preconditioner=sr)

    show_progress = params.get("show_progress", False)

    # Run optimization and collect energy per step
    log = driver.run(n_iter=n_iter, out=None, show_progress=show_progress)

    out = {
        "vstate": vstate,
        "log": log
    }
    return out
