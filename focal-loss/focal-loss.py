import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    p: predicted probabilities (shape N,)
    y: ground truth labels (0 or 1, shape N,)
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    Fl = (
        - y * (1 - p) ** gamma * np.log(p )
        - (1 - y) * p ** gamma * np.log(1 - p )
    )

    return float(np.mean(Fl))