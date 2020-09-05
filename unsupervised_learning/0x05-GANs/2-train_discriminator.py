#!/usr/bin/env python3
"""
2-train_discriminator.py
"""
import tensorflow as tf
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_discriminator(Z, X):
    """function that trains the discriminator"""

    # Sample generated images from Z
    X_gen = generator(Z)
    # Pass X_gen on to the discriminator, compute Y_gen
    Y_gen = discriminator(X_gen)

    # Pass X (original images) to the discriminator
    Y = discriminator(X)

    # Evaluate the discriminator loss (negative minimax loss)
    loss = -tf.reduce_mean(tf.log(Y) + tf.log(1 - Y_gen))

    # Trainable variables to update to minimize loss
    # print("trainable_variables:", tf.trainable_variables())
    var_list = [v for v in tf.trainable_variables()
                if 'discriminator' in v.name]

    # Setup the optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss, var_list=var_list)

    return loss, train_op
