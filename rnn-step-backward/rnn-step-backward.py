import numpy as np

def rnn_step_backward(dh, cache):
    """
    dh: (H,)
    cache: [x_t, h_prev, h_t, W, U, b]

    Returns:
        dx_t: (D,)
        dh_prev: (H,)
        dW: (H, D)
        dU: (H, H)
        db: (H,)
    """
    x_t, h_prev, h_t, W, U, b = cache

    x_t = np.asarray(x_t, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    h_t = np.asarray(h_t, dtype=float)
    W = np.asarray(W, dtype=float)
    U = np.asarray(U, dtype=float)
    dh = np.asarray(dh, dtype=float)

    # 1. tanh derivative
    dz = dh * (1 - h_t ** 2)

    # 2. gradients
    dx_t = W.T @ dz
    dh_prev = U.T @ dz

    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    db = dz

    return dx_t, dh_prev, dW, dU, db