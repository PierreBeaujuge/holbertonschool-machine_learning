#!/usr/bin/env python3
"""
Predict
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """function that makes a prediction using a neural network"""
    result = network.predict(
        x=data,
        batch_size=None,
        verbose=verbose
    )
    return result
