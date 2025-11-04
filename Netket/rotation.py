import numpy as np
from scipy.linalg import expm
def rotation_matrix_rpy(roll, pitch, yaw):
    """
    Compute the 3D rotation matrix from roll, pitch, yaw (Euler angles).
    
    Args:
        roll  (float): rotation angle around x-axis, in radians
        pitch (float): rotation angle around y-axis, in radians
        yaw   (float): rotation angle around z-axis, in radians
    
    Returns:
        R (3x3 numpy.ndarray): rotation matrix
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])
    
    return R



def get_U_single(alpha, beta, gamma):
    # Use sx, sy, sz from notebook
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    return expm(-1j*gamma*sz) @ expm(-1j*beta*sy) @ expm(-1j*alpha*sx)

def apply_local_rotation_to_state(psi, U_single, N, d=2):
    # psi: vector shape (d**N,)
    # U_single: (d,d)
    # we assume last index is site 0 (fastest varying). Change reshape order if needed.
    psi_t = psi.reshape((d,)*N)                    # tensor shape (d,d,...,d)
    # apply U to each site using tensordot, looping over sites
    for site in range(N):
        # move axis 'site' to front, apply U, move back
        psi_t = np.moveaxis(psi_t, site, 0)       # bring current site to axis 0
        psi_t = np.tensordot(U_single, psi_t, axes=[1,0])  # result shape (d, ... )
        psi_t = np.moveaxis(psi_t, 0, site)       # put axis back
    return psi_t.reshape(-1)

def get_rotation_objective(psi_initial, psi_target = None):
    if psi_target is None:
        psi_target = np.ones_like(psi_initial)/np.sqrt(len(psi_initial))
    N = int(np.log2(len(psi_target)))
    def objective(params):
        alpha, beta, gamma = params
        U1 = get_U_single(alpha, beta, gamma)
        # U1 = rotation_unitary_rpy(alpha, beta, gamma)
        # Rotate psi_max_ipr_arr and compare to psi_min_ipr_arr
        psi_rot = apply_local_rotation_to_state(psi_initial, U1, N, d=2)
        # Use negative overlap as loss
        overlap = np.abs(np.vdot(psi_target, psi_rot)) / (np.linalg.norm(psi_rot) * np.linalg.norm(psi_target))
        return -overlap
    return objective