import numpy as np

def binning(values, num_bins):
    values = np.asarray(values, dtype=float)

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")

    if values.size == 0:
        return []

    min_val = np.min(values)
    max_val = np.max(values)

    if max_val == min_val:
        return [0] * len(values)

    width = (max_val - min_val) / num_bins

    bins = np.floor((values - min_val) / width).astype(int)

    # Fix right edge case
    bins[bins == num_bins] = num_bins - 1

    return bins.tolist()