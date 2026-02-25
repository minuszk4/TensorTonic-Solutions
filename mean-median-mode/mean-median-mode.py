import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.array(x, dtype=float)
    return np.mean(x), np.median(x), Counter(x).most_common(1)[0][0]