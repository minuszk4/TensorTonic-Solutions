import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Returns: updated value function V_new
    """
    # Write code here
    V = np.asarray(V, dtype=float)
    V_new = V.copy()
    err = r + gamma * V[s_next] - V[s]
    V_new[s] += alpha * err
    return V_new.tolist()