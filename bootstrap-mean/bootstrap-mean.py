import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns: (boot_means, lower, upper)
    """
    # Write code here
    if rng is None:
        rng = np.random.default_rng()
    x = np.array(x, dtype=float)
    n = len(x)
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(x, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2* 100)
    return boot_means, lower, upper