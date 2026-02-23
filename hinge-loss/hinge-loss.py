import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    y_true: 1D array of {-1,+1}
    y_score: 1D array of real scores, same shape as y_true
    reduction: "mean" or "sum"
    Return: float
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    losses = np.maximum(0, margin - y_true * y_score)
    if reduction == "mean":
        return np.mean(losses)
    elif reduction == "sum":
        return np.sum(losses)
    else:
        raise ValueError("Invalid reduction type. Use 'mean' or 'sum'.")