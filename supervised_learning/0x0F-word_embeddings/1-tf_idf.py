#!/usr/bin/env python3
"""
1-tf_idf.py
"""
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """function that produces an array of embeddings from a corpus"""

    # tf-idf: term frequency–inverse document frequency
    # The tf–idf value increases proportionally to the number of times
    # a word appears in the document and is offset by the number of documents
    # in the corpus that contains the word, which helps to adjust for the fact
    # that some words appear more frequently in general.

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)

    embeddings = X.toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
