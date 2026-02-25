import math
def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    
    P = 1.0
    for n in range(1, max_n + 1):
        candidate_ngrams = {}
        reference_ngrams = {}

        for i in range(len(candidate) - n + 1):
            ngram = tuple(candidate[i:i + n])
            candidate_ngrams[ngram] = candidate_ngrams.get(ngram, 0) + 1

        for i in range(len(reference) - n + 1):
            ngram = tuple(reference[i:i + n])
            reference_ngrams[ngram] = reference_ngrams.get(ngram, 0) + 1

        overlap = sum(min(candidate_ngrams.get(ngram, 0), reference_ngrams.get(ngram, 0)) for ngram in candidate_ngrams)
        total = sum(candidate_ngrams.values())

        if total > 0:
            P *= overlap / total
        else:
            P *= 0
    c = len(candidate)
    r = len(reference)
    if c > r:
        BP = 1
    else:
        BP = math.exp(1 - r / c) if c > 0 else 0
    return BP * P ** (1 / max_n)