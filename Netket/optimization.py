import numpy as np
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
import h5py
import json

import jax.numpy as jnp

def construct_hamiltonian_bonds(Jijalphabeta, h, bonds):
    N = h.shape[0]
    hilbert = nk.hilbert.Spin(s=0.5, N=N)
    pauli = [nk.operator.spin.sigmax, nk.operator.spin.sigmay, nk.operator.spin.sigmaz]

    interaction_terms = [
        Jijalphabeta[bond, alpha, beta] * pauli[alpha](hilbert, i) * pauli[beta](hilbert, j)
        for (bond, (i, j)) in enumerate(bonds)
        for alpha in range(3)
        for beta in range(3)
        if np.abs(Jijalphabeta[bond, alpha, beta]) > 1e-12
    ]

    field_terms = [
        h[i, alpha] * pauli[alpha](hilbert, i)
        for i in range(N)
        for alpha in range(3)
        if np.abs(h[i, alpha]) > 1e-12
    ]

    ha = sum(interaction_terms, nk.operator.LocalOperator(hilbert)) + sum(field_terms, nk.operator.LocalOperator(hilbert))
    return ha

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

DEFAULT_PARAMS = {
        "alpha": 1,  # Hidden unit density
        "learning_rate": 0.01,
        "n_iter": 200,
        "diag_shift": 0.01,
        "show_progress": False,
        "out": None,  # Output filename prefix
        "symmetries": None,  # List or array of symmetry operations (e.g., permutation matrices), or None if not used
    }

def optimize_rbm(H, params):
    # Hilbert space from Hamiltonian
    # Use DEFAULT_PARAMS as base, override with params
    merged_params = {**DEFAULT_PARAMS, **params}
    alpha = merged_params["alpha"]
    learning_rate = merged_params["learning_rate"]
    n_iter = merged_params["n_iter"]
    diag_shift = merged_params["diag_shift"]
    show_progress = merged_params["show_progress"]
    out = merged_params["out"]

    hilbert = H.hilbert
    symmetries = merged_params["symmetries"]

    if symmetries is not None:
        model = nk.models.RBMSymm(alpha=alpha, symmetries=symmetries, use_visible_bias=True, use_hidden_bias=True)
    else:
        model = nk.models.RBM(alpha=alpha, use_visible_bias=True, use_hidden_bias=True)

    vstate = nk.vqs.FullSumState(hilbert, model)

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)

    sr = nk.optimizer.SR(diag_shift=diag_shift)
    
    driver = nk.VMC(hamiltonian=H, optimizer=optimizer, variational_state=vstate, preconditioner=sr)

    driver.run(n_iter=n_iter, out=out, show_progress=show_progress)
    return vstate

def optimize_inf_rbm(H, params):
    # Use DEFAULT_PARAMS as base, override with params
    merged_params = {**DEFAULT_PARAMS, **params}
    alpha = merged_params["alpha"]
    learning_rate = merged_params["learning_rate"]
    n_iter = merged_params["n_iter"]
    diag_shift = merged_params["diag_shift"]
    show_progress = merged_params["show_progress"]
    out = merged_params["out"]

    hilbert = H.hilbert
    model = nk.models.RBM(alpha=alpha, use_visible_bias=True, use_hidden_bias=True)

    sa = nk.sampler.MetropolisLocal(hilbert=hilbert, n_chains_per_rank=16)

    e_gs, v_gs = nk.exact.lanczos_ed(H, compute_eigenvectors=True)

    vs_target = nk.vqs.MCState(
        sampler=sa,
        model=nk.models.LogStateVector(hilbert, param_dtype=jnp.float64),
        n_samples=100,
        variables={"params": {"logstate": jnp.log(v_gs.astype(jnp.complex128)).squeeze()}},
    )

    vs = nk.vqs.MCState(
        sampler=sa,
        model=model,
        n_samples=100,
    )
    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)

    sr = nk.optimizer.SR(diag_shift=diag_shift)
    
    import netket.experimental as nkx
    driver = nkx.driver.Infidelity_SR(
        target_state=vs_target,
        optimizer=optimizer,
        diag_shift=diag_shift,
        variational_state=vs,
        operator=None,
        # preconditioner=sr
    )
    driver.run(n_iter=n_iter, out=out, show_progress=show_progress)
    return vs

# Limit for parameter string length in filenames to avoid excessively long file names.
MAX_PARAM_STRING_LENGTH = 30
def generate_filename(params):
    filename = params.get("out", "output")
    for key, value in params.items():
        val_str = str(value)
        if key != "out" and len(val_str) <= MAX_PARAM_STRING_LENGTH:
            formatted_value = val_str.replace(".", "_")
            filename += f"_{key}_{formatted_value}"
    return filename

def generate_params(**kwargs):
    params = {
        **kwargs
    }
    filename = generate_filename(params)
    params["out"] = filename
    return params

def maximally_positive_sign(psi):
    avg_sign = np.mean(np.sign(psi))
    if avg_sign < 0:
        print("Flipping the sign of the ground state wavefunction to ensure positive overlap.")
        psi *= -1
    return psi

def write_output(H, vstate, params):
    psi = maximally_positive_sign(vstate.to_array())
    outfile = params.get("out", "output") + ".h5"
    try:
        logfile = params.get("out", "output") + ".log"
        data = json.load(open(logfile))
        en_var_steps = data["Energy"]["Mean"]["real"]
        en_var = en_var_steps[-1]
    except Exception as e:
        print(f"Could not read log file {logfile}: {e}")
        en_var_steps = None
        en_var = None
    
    exact_ground_energy, psi_0 = nk.exact.lanczos_ed(H, k=1, compute_eigenvectors=True)
    psi_0 = maximally_positive_sign(psi_0[:, 0])  # Get the ground state vector
    
    with h5py.File(outfile, "w") as f:
        f.create_dataset("psi", data=psi)
        f.create_dataset("en_var_steps", data=en_var_steps if en_var_steps is not None else False)
        f.create_dataset("en_var", data=en_var if en_var is not None else False)
        f.create_dataset("exact_ground_energy", data=exact_ground_energy[0])
        f.create_dataset("psi_0", data=psi_0)

        for (key, value) in params.items():
            if isinstance(value, (int, float, str)):
                f.attrs[key] = value
            else:
                f.attrs[key] = str(value)


