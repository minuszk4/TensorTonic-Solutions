import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    pe = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(d_model):
            exponent = 2 * (i // 2) / d_model
            angle = pos / (base ** exponent)
            
            if i % 2 == 0:
                pe[pos, i] = np.sin(angle)
            else:
                pe[pos, i] = np.cos(angle)
                
    return pe