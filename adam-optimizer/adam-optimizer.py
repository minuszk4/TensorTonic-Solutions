import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m = np.array(m, dtype=np.float64)
    v = np.array(v, dtype=np.float64)
    param = np.array(param, dtype=np.float64)
    grad = np.array(grad, dtype=np.float64)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    mt =m /(1 - beta1 ** t)
    vt = v/(1 - beta2 ** t)
    param  = param - lr * mt / (np.sqrt(vt) + eps)
    return param, m, v