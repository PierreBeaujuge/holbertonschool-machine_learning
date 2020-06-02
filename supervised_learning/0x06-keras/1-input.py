#!/usr/bin/env python3
"""
Input - use the Input class
"""
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    inputs = tf.keras.layers.Input(shape=(nx,))
    for i in range(len(layers)):
        if i == 0:
            outputs = inputs
        outputs = tf.keras.layers.Dense(
            layers[i], activation=activations[i],
            kernel_regularizer=tf.keras.regularizers.l2(lambtha))(outputs)
        if i != len(layers) - 1:
            outputs = tf.keras.layers.Dropout(1 - keep_prob)(outputs)
    network = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return network
