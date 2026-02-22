import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    w = np.array(w, dtype=np.float64)
    g = np.array(g, dtype=np.float64)
    s = np.array(s, dtype=np.float64)
    s = beta*s + (1-beta)*g**2
    w = w- lr*g/(np.sqrt(s)+eps)
    return w, s