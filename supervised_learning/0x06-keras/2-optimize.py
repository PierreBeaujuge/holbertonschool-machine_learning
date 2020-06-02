#!/usr/bin/env python3
"""
Optimize
"""
import tensorflow as tf


def optimize_model(network, alpha, beta1, beta2):
    """function that sets up an Adam optimizer"""
    optimizer = tf.keras.optimizers.Adam(lr=alpha,
                                         beta_1=beta1, beta_2=beta2)
    loss = 'categorical_crossentropy'
    network.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return None
