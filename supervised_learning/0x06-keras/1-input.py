#!/usr/bin/env python3
"""
Input - use the Input class
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    inputs = K.layers.Input(shape=(nx,))
    for i in range(len(layers)):
        if i == 0:
            outputs = inputs
        outputs = K.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(outputs)
        if i != len(layers) - 1:
            outputs = K.layers.Dropout(1 - keep_prob)(outputs)
    network = K.models.Model(inputs=inputs, outputs=outputs)
    return network
