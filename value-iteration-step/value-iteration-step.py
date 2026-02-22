import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    values = np.array(values, dtype=float)
    transitions = np.array(transitions, dtype=float)
    rewards = np.array(rewards, dtype=float)

    # Compute expected future value:
    # For each (s, a): sum_s' T(s,a,s') * V(s')
    expected_future = np.sum(transitions * values, axis=2)

    # Compute Q(s,a)
    Q = rewards + gamma * expected_future

    # Take max over actions
    new_values = np.max(Q, axis=1)

    return new_values.tolist()