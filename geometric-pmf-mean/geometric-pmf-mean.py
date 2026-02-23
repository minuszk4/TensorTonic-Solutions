import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    k = np.asarray(k, dtype=float)
    p = np.asarray(p, dtype=float)

    pmf = (1 - p) ** (k - 1) * p
    mean = 1 / p

    return pmf, mean