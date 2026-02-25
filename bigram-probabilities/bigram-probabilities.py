def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """

    bigram_counts = {}
    unigram_counts = {}

    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]

        bigram_counts[(w1, w2)] = bigram_counts.get((w1, w2), 0) + 1
        unigram_counts[w1] = unigram_counts.get(w1, 0) + 1

    vocab = set(tokens)
    V = len(vocab)

    probs = {}

    for w1 in vocab:
        for w2 in vocab:
            c = bigram_counts.get((w1, w2), 0)
            denom = unigram_counts.get(w1, 0) + V
            probs[(w1, w2)] = (c + 1) / denom

    return bigram_counts, probs