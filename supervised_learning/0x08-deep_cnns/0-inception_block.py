#!/usr/bin/env python3
"""
Inception Block
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    function that builds an inception block
    as described in Going Deeper with Convolutions (2014)
    """
    initializer = K.initializers.he_normal()
    F1_layer = K.layers.Conv2D(filters=filters[0],
                               kernel_size=1,
                               padding='same',
                               kernel_initializer=initializer,
                               activation='relu')
    F1_output = F1_layer(A_prev)
    F3R_layer = K.layers.Conv2D(filters=filters[1],
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=initializer,
                                activation='relu')
    F3R_output = F3R_layer(A_prev)
    F3_layer = K.layers.Conv2D(filters=filters[2],
                               kernel_size=3,
                               padding='same',
                               kernel_initializer=initializer,
                               activation='relu')
    F3_output = F3_layer(F3R_output)
    F5R_layer = K.layers.Conv2D(filters=filters[3],
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=initializer,
                                activation='relu')
    F5R_output = F5R_layer(A_prev)
    F5_layer = K.layers.Conv2D(filters=filters[4],
                               kernel_size=5,
                               padding='same',
                               kernel_initializer=initializer,
                               activation='relu')
    F5_output = F5_layer(F5R_output)
    Pool_layer = K.layers.MaxPool2D(pool_size=3,
                                    padding='same',
                                    strides=1)
    Pool_output = Pool_layer(A_prev)
    FPP_layer = K.layers.Conv2D(filters=filters[5],
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=initializer,
                                activation='relu')
    FPP_output = FPP_layer(Pool_output)

    # concatenate the outputs of the branches
    output = K.layers.concatenate(
        [F1_output, F3_output, F5_output, FPP_output])

    return output
