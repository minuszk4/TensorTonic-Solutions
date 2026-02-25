import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    If matrix is singular, return None
    """
    try:
        A = np.array(A, dtype=float)
        n = A.shape[0]
        A_inv = np.zeros_like(A)
        I = np.eye(n)

        for i in range(n):
            A_inv[:, i] = np.linalg.solve(A, I[:, i])

        return A_inv

    except np.linalg.LinAlgError:
        return None