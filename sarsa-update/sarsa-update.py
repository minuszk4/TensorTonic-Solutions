import numpy as np
def sarsa_update(q_table, state, action, reward, next_state, next_action, alpha, gamma):
    """
    Perform one SARSA update and return the updated Q-table.
    """
    # Write code here
    q_table = np.asarray(q_table, dtype=float)
    q_next = q_table[next_state, next_action]
    q_table[state, action] += alpha * (reward + gamma * q_next - q_table[state, action])
    return q_table.tolist()