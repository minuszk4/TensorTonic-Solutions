import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    anchor = np.asarray(anchor, dtype=float)
    positive = np.asarray(positive, dtype=float)
    negative = np.asarray(negative, dtype=float)

    if anchor.ndim == 1:
        d_ap = np.sum((anchor - positive) ** 2)
        d_an = np.sum((anchor - negative) ** 2)
        return float(max(0.0, d_ap - d_an + margin))

    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)

    losses = np.maximum(0, d_ap - d_an + margin)

    return float(np.mean(losses))