#!/usr/bin/env python3
"""
Adam Upgraded
"""
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """function that implements Adam gradient descent in tensorflow"""
    return tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon,
        use_locking=False, name='Adam'
    ).minimize(loss)
