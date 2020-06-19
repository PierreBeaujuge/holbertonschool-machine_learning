#!/usr/bin/env python3
"""
Identity Block
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    function that builds an identity block
    as described in Deep Residual Learning for Image Recognition (2015)
    """
    initializer = K.initializers.he_normal()

    F11_layer = K.layers.Conv2D(filters=filters[0],
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=initializer,
                                activation=None)
    F11_output = F11_layer(A_prev)
    F11_norm = K.layers.BatchNormalization()
    F11_output = F11_norm(F11_output)
    F11_activ = K.layers.Activation('relu')
    F11_output = F11_activ(F11_output)

    F3_layer = K.layers.Conv2D(filters=filters[1],
                               kernel_size=3,
                               padding='same',
                               kernel_initializer=initializer,
                               activation=None)
    F3_output = F3_layer(F11_output)
    F3_norm = K.layers.BatchNormalization()
    F3_output = F3_norm(F3_output)
    F3_activ = K.layers.Activation('relu')
    F3_output = F3_activ(F3_output)

    F12_layer = K.layers.Conv2D(filters=filters[2],
                                kernel_size=1,
                                padding='same',
                                kernel_initializer=initializer,
                                activation=None)
    F12_output = F12_layer(F3_output)
    F12_norm = K.layers.BatchNormalization()
    F12_output = F12_norm(F12_output)

    # add input (residual connection) and output
    output = K.layers.Add()([A_prev, F12_output])
    # activate the combined output
    output = K.layers.Activation('relu')(output)

    return output
