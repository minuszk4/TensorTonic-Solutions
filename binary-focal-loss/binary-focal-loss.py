import numpy as np
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    pt = np.where(targets == 1, predictions, 1 - predictions)
    loss = -alpha * (1 - pt) ** gamma * np.log(pt + 1e-12)
    return np.mean(loss)