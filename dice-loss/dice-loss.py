import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    intersection = np.sum(p * y)    
    union = np.sum(p) + np.sum(y)
    dice_coeff = (2 * intersection + eps) / (union + eps)
    return 1 - dice_coeff