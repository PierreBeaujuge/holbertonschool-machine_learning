#!/usr/bin/env python3
"""
3-gensim_to_keras.py
"""
from gensim.models import Word2Vec
# Install keras with: pip install --user keras==2.2.5
# for this task to work


def gensim_to_keras(model):
    """function that converts a gensim word2vec model
    to a trainable keras layer"""

    return model.wv.get_keras_embedding(train_embeddings=True)
