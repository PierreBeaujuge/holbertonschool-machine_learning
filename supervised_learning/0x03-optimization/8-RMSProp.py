#!/usr/bin/env python3
"""
RMSProp Upgraded
"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """function that implements RMSProp gradient descent in tensorflow"""
    return tf.train.RMSPropOptimizer(
        learning_rate=alpha, decay=beta2, momentum=0.0, epsilon=epsilon,
        use_locking=False, centered=False, name='RMSProp'
    ).minimize(loss)
