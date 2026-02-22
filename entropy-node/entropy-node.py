import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)
    if len(y) == 0:
        return 0.0

    _,counts =np.unique(y, return_counts=True)
    prob = counts / len(y)

    entropy = -np.sum(np.where(prob > 0, prob * np.log2(prob), 0))
    return entropy