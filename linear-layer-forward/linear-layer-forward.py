import numpy as np

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here
    X = np.array(X, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    output = X @ W + b
    return output.tolist()