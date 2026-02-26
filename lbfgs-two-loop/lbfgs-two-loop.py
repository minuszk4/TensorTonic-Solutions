def _dot(a, b):
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def lbfgs_direction(grad, s_list, y_list):
    """
    Compute the L-BFGS search direction using the two-loop recursion.
    """
    # Write code here
    q = grad
    alpha = []
    for s, y in zip(reversed(s_list), reversed(y_list)):
        rho = 1.0 / _dot(y, s)
        a = rho * _dot(s, q)
        alpha.append(a)
        q = [q_i - a * y_i for q_i, y_i in zip(q, y)]
    if s_list:
        gamma = _dot(s_list[-1], y_list[-1]) / _dot(y_list[-1], y_list[-1])
    else:
        gamma = 1.0
    r = [gamma * q_i for q_i in q]
    for s, y, a in zip(s_list, y_list, reversed(alpha)):
        rho = 1.0 / _dot(y, s)
        b = rho * _dot(y, r)
        r = [r_i + s_i * (a - b) for r_i, s_i in zip(r, s)]
    return [-r_i for r_i in r]