import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    Return None for invalid input.
    """

    try:
        matrix = np.array(matrix, dtype=float)

        if matrix.ndim != 2:
            return None

        if axis not in (None, 0, 1):
            return None

        if norm_type not in ('l1', 'l2', 'max'):
            return None

        if norm_type == 'l1':
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        elif norm_type == 'l2':
            norm = np.sqrt(np.sum(matrix ** 2, axis=axis, keepdims=True))
        else:  # max norm
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)

        norm[norm == 0] = 1

        return matrix / norm

    except:
        return None