import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if h_prev.ndim == 1:
        h_prev = h_prev.reshape(1, -1)
    W_z, U_z, b_z = params['Wz'], params['Uz'], params['bz']
    W_r, U_r, b_r = params['Wr'], params['Ur'], params['br']
    W_h, U_h, b_h = params['Wh'], params['Uh'], params['bh']
    z = _sigmoid(x @ W_z + h_prev @ U_z + b_z)
    r = _sigmoid(x @ W_r + h_prev @ U_r + b_r)
    h_tilde = np.tanh(x @ W_h + (r * h_prev) @ U_h + b_h)
    h_next = (1 - z) * h_prev + z * h_tilde             
    return h_next.squeeze() if h_next.shape[0] == 1 else h_next
