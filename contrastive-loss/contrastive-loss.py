import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a = np.array(a)
    b = np.array(b)
    y = np.array(y)
    distances = np.linalg.norm(a - b, axis=-1)
    loss_similar = y * distances**2
    loss_dissimilar = (1 - y) * np.maximum(0, margin - distances)**2
    loss = loss_similar + loss_dissimilar

    if reduction == "mean":
        return np.mean(loss)
    elif reduction == "sum":
        return np.sum(loss)
    else:
        raise ValueError("Invalid reduction type. Use 'mean' or 'sum'.")