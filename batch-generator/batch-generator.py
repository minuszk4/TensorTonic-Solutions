import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    if rng is None:
        rng = np.random.default_rng()
    n_samples = len(X)
    X = np.asarray(X)
    y = np.asarray(y)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = start + batch_size
        if drop_last and end > n_samples:
            break
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]
    
