import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # Write code here
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)
    s = np.dot(Z1, Z2.T) / temperature
    exp_s = np.exp(s - np.max(s, axis=1, keepdims=True))
    loss = -np.log(exp_s[np.arange(len(Z1)), np.arange(len(Z1))] / np.sum(exp_s, axis=1))
    return np.mean(loss)