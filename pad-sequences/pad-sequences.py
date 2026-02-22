import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)

    padded_seqs = np.full((len(seqs), max_len), pad_value, dtype=np.int32)

    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        padded_seqs[i, :length] = seq[:length]

    return padded_seqs