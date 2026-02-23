import numpy as np
def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    # Write code here
    W = np.asarray(W, dtype=float)
    limit = np.sqrt(6 / fan_in)
    W = W * (2 * limit) - limit
    return W.tolist()