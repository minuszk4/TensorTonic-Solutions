import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.array(T)
    points = np.array(points)

    single_point = False
    if points.ndim == 1:
        points = points.reshape(1, -1)
        single_point = True

    if points.shape[1] != 3:
        raise ValueError("Points should have shape (3,) or (N, 3)")

    # Convert to homogeneous
    ph = np.hstack((points, np.ones((points.shape[0], 1))))

    # Apply transform
    pth = (T @ ph.T).T  # (N,4)

    # Convert back from homogeneous
    result = pth[:, :3] / pth[:, 3][:, None]

    if single_point:
        return result[0]   # shape (3,)

    return result         # shape (N,3)