import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    Perform one AdaGrad update step.
    
    Returns:
        new_w, new_G
    """
    # Ensure float dtype (avoid casting errors)
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    G = np.asarray(G, dtype=float)

    # Step 1: accumulate squared gradients
    new_G = G + g**2

    # Step 2: parameter update
    new_w = w - (lr / (np.sqrt(new_G + eps))) * g

    return new_w, new_G