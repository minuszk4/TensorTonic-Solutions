import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g)

    if max_norm <= 0:
        return g

    total_norm = np.linalg.norm(g)

    if total_norm == 0:
        return g

    # Apply clipping if needed
    if total_norm > max_norm:
        scale = max_norm / total_norm
        g = g * scale

    return g