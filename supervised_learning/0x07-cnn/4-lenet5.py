#!/usr/bin/env python3
"""
LeNet-5 in Tensorflow
"""
import tensorflow as tf


def lenet5(x, y):
    """function that builds a modified version of LeNet-5"""
    initializer = tf.contrib.layers.variance_scaling_initializer()
    layer = tf.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding='same',
                             kernel_initializer=initializer,
                             activation=tf.nn.relu)
    output = layer(x)
    layer = tf.layers.MaxPooling2D(pool_size=2,
                                   strides=2)
    output = layer(output)
    layer = tf.layers.Conv2D(filters=16,
                             kernel_size=5,
                             padding='valid',
                             kernel_initializer=initializer,
                             activation=tf.nn.relu)
    output = layer(output)
    layer = tf.layers.MaxPooling2D(pool_size=2,
                                   strides=2)
    output = layer(output)
    layer = tf.layers.Flatten()
    output = layer(output)
    layer = tf.layers.Dense(units=120,
                            activation=tf.nn.relu,
                            kernel_initializer=initializer)
    output = layer(output)
    layer = tf.layers.Dense(units=84,
                            activation=tf.nn.relu,
                            kernel_initializer=initializer)
    output = layer(output)
    layer = tf.layers.Dense(units=10,
                            kernel_initializer=initializer)
    output = layer(output)

    # define loss from unactivated ouput (logits)
    loss = tf.losses.softmax_cross_entropy(y, output)

    # define an Adam optimizer with default learning rate
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # activate the output with softmax
    y_pred = tf.nn.softmax(output)

    # evaluate the accuracy of the model
    acc = accuracy(y, y_pred)

    return y_pred, train_op, loss, acc


def accuracy(y, y_pred):
    """evaluate the accuracy of the model"""
    label = tf.argmax(y, axis=1)
    pred = tf.argmax(y_pred, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(label, pred), tf.float32))
    return acc
