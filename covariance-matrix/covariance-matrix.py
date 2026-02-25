import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.array(X, dtype=float)
    if X.ndim != 2:
        return None
    n, d = X.shape
    if n < 2:
        return None
    mean = np.mean(X, axis=0)
    C = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cov = np.sum((X[:, i] - mean[i]) *
                         (X[:, j] - mean[j])) / (n - 1)
            C[i, j] = cov
    return C