#!/usr/bin/env python3
"""
Sequential - use the Sequential class
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    network = K.models.Sequential()
    for i in range(len(layers)):
        if i == 0:
            network.add(K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha),
                input_shape=(nx,)))
        else:
            network.add(K.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=K.regularizers.l2(lambtha)))
        if i != len(layers) - 1:
            network.add(K.layers.Dropout(1 - keep_prob))
    return network
