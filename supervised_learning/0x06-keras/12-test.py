#!/usr/bin/env python3
"""
Test
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """function that tests a neural network"""
    result = network.evaluate(
        x=data,
        y=labels,
        batch_size=None,
        verbose=verbose
    )
    return result
