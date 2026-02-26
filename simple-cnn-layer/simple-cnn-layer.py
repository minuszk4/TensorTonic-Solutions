import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    x = np.asarray(x)
    W = np.asarray(W)
    b = np.asarray(b)

    C_out, C_in, K_h, K_w = W.shape
    N, C_in_x, H_x, W_x = x.shape

    H_out = H_x - K_h + 1
    W_out = W_x - K_w + 1

    out = np.zeros((N, C_out, H_out, W_out))

    for n in range(N):
        for c_out in range(C_out):
            for h in range(H_out):
                for w in range(W_out):
                    out[n, c_out, h, w] = np.sum(
                        x[n, :, h:h+K_h, w:w+K_w] * W[c_out]
                    ) + b[c_out]

    return out