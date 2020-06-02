#!/usr/bin/env python3
"""
Train a model, with validaiton data
"""
import tensorflow as tf


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data)
    return history
