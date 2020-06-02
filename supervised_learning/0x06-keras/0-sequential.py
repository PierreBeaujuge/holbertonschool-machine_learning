#!/usr/bin/env python3
"""
Sequential - use the Sequential class
"""
import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function that builds a neural network with the Keras library"""
    network = tf.keras.models.Sequential()
    for i in range(len(layers)):
        if i == 0:
            network.add(tf.keras.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=tf.keras.regularizers.l2(lambtha),
                input_shape=(nx,)))
        else:
            network.add(tf.keras.layers.Dense(
                layers[i], activation=activations[i],
                kernel_regularizer=tf.keras.regularizers.l2(lambtha)))
        if i != len(layers) - 1:
            network.add(tf.keras.layers.Dropout(1 - keep_prob))
    return network
