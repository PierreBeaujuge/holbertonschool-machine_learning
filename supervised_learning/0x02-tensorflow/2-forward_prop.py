#!/usr/bin/env python3
"""
Forward Propagation
"""
import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """function that creates the forward propagation graph for the nn"""
    for i in range(len(layer_sizes)):
        if i == 0:
            y_pred = x
        y_pred = create_layer(y_pred, layer_sizes[i], activations[i])
    return y_pred
