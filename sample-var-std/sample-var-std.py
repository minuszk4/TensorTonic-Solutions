import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    s = np.var(x, ddof=1)
    std = np.sqrt(s)
    return s, std