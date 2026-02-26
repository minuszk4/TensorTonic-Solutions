import numpy as np

def maxpool_forward(X, pool_size, stride):

    X = np.asarray(X)
    
    H, W = X.shape
    p = pool_size
    s = stride
    
    H_out = (H - p) // s + 1
    W_out = (W - p) // s + 1
    
    out = []
    
    for i in range(H_out):
        row = []
        for j in range(W_out):
            h_start = i * s
            w_start = j * s
            
            window = X[h_start:h_start+p,
                       w_start:w_start+p]
            
            row.append(int(np.max(window)))  
        
        out.append(row)
    
    return out