import numpy as np

def q_learning_update(Q, s, a, r, s_next, alpha, gamma):
    """
    Returns: updated Q-table Q_new
    """
    Q = np.array(Q, dtype=np.float64)
    max_next_q = np.max(Q[s_next])
    Q[s, a] = Q[s, a] + alpha * (r + gamma * max_next_q - Q[s, a])
    return Q