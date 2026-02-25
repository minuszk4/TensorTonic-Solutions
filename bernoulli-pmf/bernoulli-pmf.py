import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    Returns (pmf, mean, var)
    """

    if not (0 <= p <= 1):
        return None

    x = np.array(x)

    if not np.all((x == 0) | (x == 1)):
        return None

    pmf = np.where(x == 1, p, 1 - p)

    mean = float(p)
    var = float(p * (1 - p))

    return pmf, mean, var