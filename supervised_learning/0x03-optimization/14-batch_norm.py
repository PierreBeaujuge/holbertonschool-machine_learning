#!/usr/bin/env python3
"""
Batch Normalization Upgraded
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """function that creates a batch normalization layer in tensorflow"""
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=None,
                            kernel_initializer=initializer,
                            name='layer')

    m, v = tf.nn.moments(layer(prev), axes=[0])
    beta = tf.Variable(
        tf.zeros(shape=(1, n), dtype=tf.float32),
        trainable=True, name='beta'
        )
    gamma = tf.Variable(
        tf.ones(shape=(1, n), dtype=tf.float32),
        trainable=True, name='gamma'
        )
    epsilon = 1e-08

    Z_b_norm = tf.nn.batch_normalization(
        x=layer(prev), mean=m, variance=v, offset=beta, scale=gamma,
        variance_epsilon=epsilon, name=None
    )
    if activation:
        return activation(Z_b_norm)
    return Z_b_norm
