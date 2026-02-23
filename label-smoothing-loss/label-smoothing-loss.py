import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    predictions = np.asarray(predictions, dtype=float)


    K = predictions.shape[0]

    q = np.full(K, epsilon / K)
    q[target] += (1 - epsilon)

    # Cross entropy
    loss = -np.sum(q * np.log(predictions + 1e-12))

    return loss