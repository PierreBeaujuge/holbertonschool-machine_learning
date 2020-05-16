#!/usr/bin/env python3
"""
Accuracy
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """function that calculates the accuracy of a prediction"""
    label = tf.argmax(y, axis=1)
    pred = tf.argmax(y_pred, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, label), tf.float32))
    return accuracy
