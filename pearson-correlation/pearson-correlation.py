import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """

    X = np.array(X, dtype=float)
    
    if X.ndim != 2:
        return None

    n, d = X.shape

    if n < 2:
        return None

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)

    R = np.zeros((d, d))

    for i in range(d):
        for j in range(d):

            if std[i] == 0 or std[j] == 0:
                R[i, j] = np.nan
            else:
                cov = np.sum((X[:, i] - mean[i]) *
                             (X[:, j] - mean[j])) / (n - 1)

                R[i, j] = cov / (std[i] * std[j])

    return R