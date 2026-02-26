import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    if x.ndim == 2:
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_normalized = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_normalized + beta
        return out
    elif x.ndim == 4:
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)
        x_normalized = (x - mean) / np.sqrt(var + eps)
        out = gamma[:, None, None] * x_normalized + beta[:, None, None]
        return out
    else:
        raise ValueError("Input must be either (N,D) or (N,C,H,W)")