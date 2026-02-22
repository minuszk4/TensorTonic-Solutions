import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x, dtype=float)
    
    if p == 0.0:
        return x.copy(), np.ones_like(x)
    
    if rng is None:
        rng = np.random.default_rng()
    
    mask = rng.random(x.shape) < (1 - p)
    
    dropout_pattern = mask.astype(float) / (1 - p)
    
    output = x * dropout_pattern
    return output, dropout_pattern