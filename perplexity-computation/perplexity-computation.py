import numpy as np
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    for prob_dist, token in zip(prob_distributions, actual_tokens):
        if token < 0 or token >= len(prob_dist):
            raise ValueError(f"Token index {token} is out of bounds for the probability distribution.")
        if prob_dist[token] <= 0:
            raise ValueError(f"Probability for token index {token} must be greater than zero.")
    log_prob_sum = 0.0
    for prob_dist, token in zip(prob_distributions, actual_tokens):
        log_prob_sum += np.log(prob_dist[token])
    avg_log_prob = log_prob_sum / len(actual_tokens)
    perplexity = np.exp(-avg_log_prob)  
    return perplexity