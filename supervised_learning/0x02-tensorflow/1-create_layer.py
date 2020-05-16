#!/usr/bin/env python3
"""
Layers
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """function that sets up a layer of n nodes"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name='layer')
    return layer(prev)
