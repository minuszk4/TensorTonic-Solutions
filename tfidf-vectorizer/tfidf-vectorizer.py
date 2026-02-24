import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    tfidf_matrix = []
    vocabulary = set()
    doc_count = len(documents)
    doc_freq = Counter()
    for doc in documents:
        word_count = Counter(doc.split())
        tfidf_matrix.append(word_count)
        vocabulary.update(word_count.keys())
        for word in word_count.keys():
            doc_freq[word] += 1
    vocabulary = sorted(vocabulary)
    tfidf_matrix_final = []
    for word_count in tfidf_matrix:
        tfidf_vector = []
        for word in vocabulary:
            tf = word_count.get(word, 0)/sum(word_count.values()) if sum(word_count.values()) > 0 else 0.0
            idf = math.log(doc_count / (doc_freq[word]))
            tfidf_vector.append(tf * idf)
        tfidf_matrix_final.append(tfidf_vector)
    return np.array(tfidf_matrix_final), vocabulary