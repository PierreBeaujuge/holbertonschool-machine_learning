#!/usr/bin/env python3
"""
1-discriminator.py
"""
import tensorflow as tf


def discriminator(X):
    """discriminator function"""

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        layer = tf.layers.Dense(units=128,
                                activation=tf.nn.relu,
                                name='layer_1')
        outputs = layer(X)
        layer = tf.layers.Dense(units=1,
                                activation=tf.nn.sigmoid,
                                name='layer_2')
        Y = layer(outputs)

    return Y
