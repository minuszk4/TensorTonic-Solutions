import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    y_true: shape (N,) — class indices
    y_pred: shape (N, C) — predicted probabilities
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=float)

    N = y_true.shape[0]

    # lấy xác suất đúng lớp
    correct_probs = y_pred[np.arange(N), y_true]

    loss = -np.mean(np.log(correct_probs))

    return float(loss)