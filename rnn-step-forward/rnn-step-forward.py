import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    x_t = np.asarray(x_t, dtype=float)
    h_prev = np.asarray(h_prev, dtype=float)
    Wx = np.asarray(Wx, dtype=float)
    Wh = np.asarray(Wh, dtype=float)
    b = np.asarray(b, dtype=float)

    h_t = np.tanh(x_t @ Wx + h_prev @ Wh + b)

    return h_t