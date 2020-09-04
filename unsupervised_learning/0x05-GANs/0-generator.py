#!/usr/bin/env python3
"""
0-generator.py
"""
import tensorflow as tf


def generator(Z):
    """generator function"""

    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        layer = tf.layers.Dense(units=128,
                                activation=tf.nn.relu,
                                name='layer_1')
        outputs = layer(Z)
        layer = tf.layers.Dense(units=784,
                                activation=tf.nn.sigmoid,
                                name='layer_2')
        X = layer(outputs)

    return X
