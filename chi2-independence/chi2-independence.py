import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    # Write code here
    C = np.array(C, dtype=float)
    if C.ndim != 2:
        return None
    row_sums = np.sum(C, axis=1)
    col_sums = np.sum(C, axis=0)
    total = np.sum(C)
    expected = np.outer(row_sums, col_sums) / total
    chi2_stat = np.sum((C - expected) ** 2 / expected)
    return chi2_stat, expected