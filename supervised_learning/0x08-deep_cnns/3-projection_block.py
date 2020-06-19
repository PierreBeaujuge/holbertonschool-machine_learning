#!/usr/bin/env python3
"""
Projection Block
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    function that builds a projection block
    as described in Deep Residual Learning for Image Recognition (2015)
    """
    initializer = K.initializers.he_normal()

    F11_layer = K.layers.Conv2D(filters=filters[0],
                                kernel_size=1,
                                padding='same',
                                strides=s,
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

    F12_bypass_layer = K.layers.Conv2D(filters=filters[2],
                                       kernel_size=1,
                                       padding='same',
                                       strides=s,
                                       kernel_initializer=initializer,
                                       activation=None)
    F12_bypass = F12_bypass_layer(A_prev)
    bypass_norm = K.layers.BatchNormalization()
    F12_bypass = bypass_norm(F12_bypass)

    # add input (bypass connection) and output
    output = K.layers.Add()([F12_output, F12_bypass])
    # activate the combined output
    output = K.layers.Activation('relu')(output)

    return output
