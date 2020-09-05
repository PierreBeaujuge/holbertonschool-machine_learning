#!/usr/bin/env python3
"""
3-train_generator.py
"""
import tensorflow as tf
generator = __import__('0-generator').generator
discriminator = __import__('1-discriminator').discriminator


def train_generator(Z):
    """function that trains the generator"""

    # Sample generated images from Z
    X_gen = generator(Z)
    # Pass X_gen on to the discriminator, compute Y_gen
    Y_gen = discriminator(X_gen)

    # Evaluate the generator loss
    loss = -tf.reduce_mean(tf.log(Y_gen))

    # Trainable variables to update to minimize loss
    # print("trainable_variables:", tf.trainable_variables())
    var_list = [v for v in tf.trainable_variables()
                if 'generator' in v.name]

    # Setup the optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss, var_list=var_list)

    return loss, train_op
