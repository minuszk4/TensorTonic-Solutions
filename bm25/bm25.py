import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    # Write code here
    idf = {}
    doc_freq = Counter()
    for doc in docs:
        unique_tokens = set(doc)
        for token in unique_tokens:
            doc_freq[token] += 1
    total_docs = len(docs)  
    for token, freq in doc_freq.items():
        idf[token] = math.log((total_docs - freq + 0.5) / (freq + 0.5) + 1)
    scores = []
    avg_doc_len = sum(len(doc) for doc in docs) / total_docs if total_docs > 0 else 0
    for doc in docs:
        score = 0.0
        doc_len = len(doc)
        for token in query_tokens:
            tf = doc.count(token)
            if tf > 0:
                score += idf.get(token, 0) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        scores.append(score)
    return np.array(scores)