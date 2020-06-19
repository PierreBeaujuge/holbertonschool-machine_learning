#!/usr/bin/env python3
"""
Dense Block
"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    function that builds a transition layer
    as described in Densely Connected Convolutional Networks
    """
    initializer = K.initializers.he_normal()

    l1_norm = K.layers.BatchNormalization()
    l1_output = l1_norm(X)
    l1_activ = K.layers.Activation('relu')
    l1_output = l1_activ(l1_output)
    l1_layer = K.layers.Conv2D(filters=int(nb_filters*compression),
                               kernel_size=1,
                               padding='same',
                               kernel_initializer=initializer,
                               activation=None)
    l1_output = l1_layer(l1_output)

    avg_pool = K.layers.AvgPool2D(pool_size=2,
                                  padding='same',
                                  strides=None)
    X = avg_pool(l1_output)

    return X, X.shape[-1]
    # return X, int(nb_filters*compression)
