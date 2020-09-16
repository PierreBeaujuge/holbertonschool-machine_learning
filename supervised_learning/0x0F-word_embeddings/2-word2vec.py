#!/usr/bin/env python3
"""
2-word2vec.py
"""
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5,
                   negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """function that creates and trains a gensim word2vec model"""

    model = Word2Vec(sentences, size=size, window=window,
                     min_count=min_count, negative=negative,
                     sg=cbow, seed=seed, workers=workers, iter=iterations)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.iter)

    return model
