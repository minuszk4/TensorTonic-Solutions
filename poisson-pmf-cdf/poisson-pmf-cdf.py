import numpy as np
import math
def poisson_pmf_cdf(lam, k):
    """
    Compute Poisson PMF and CDF.
    """
    # Write code here
    pmf = (lam ** k) * np.exp(-lam) / math.factorial(k)
    cdf = sum((lam ** i) * np.exp(-lam) / math.factorial(i) for i in range(k + 1))
    return pmf, cdf