import numpy as np
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