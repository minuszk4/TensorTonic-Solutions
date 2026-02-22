import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    vector = np.zeros(len(vocab), dtype=int)
    token_counts = {}
    for token in tokens:
        if token in vocab:
            token_counts[token] = token_counts.get(token, 0) + 1
    for i, word in enumerate(vocab):
        vector[i] = token_counts.get(word, 0)
    return vector