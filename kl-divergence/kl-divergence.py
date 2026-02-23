import numpy as np

def kl_divergence(p, q, eps=1e-12):
    """
    Compute KL Divergence D_KL(P || Q).
    """
    # Write code here
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    D_KL = np.sum(p * np.log((p + eps) / (q + eps)))
    return D_KL