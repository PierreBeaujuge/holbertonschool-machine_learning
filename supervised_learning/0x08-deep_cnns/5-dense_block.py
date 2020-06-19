#!/usr/bin/env python3
"""
Dense Block
"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    function that builds a dense block
    as described in Densely Connected Convolutional Networks
    """
    initializer = K.initializers.he_normal()

    for i in range(layers):

        l1_norm = K.layers.BatchNormalization()
        l1_output = l1_norm(X)
        l1_activ = K.layers.Activation('relu')
        l1_output = l1_activ(l1_output)
        l1_layer = K.layers.Conv2D(filters=4*growth_rate,
                                   kernel_size=1,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   activation=None)
        l1_output = l1_layer(l1_output)

        l2_norm = K.layers.BatchNormalization()
        l2_output = l2_norm(l1_output)
        l2_activ = K.layers.Activation('relu')
        l2_output = l2_activ(l2_output)
        l2_layer = K.layers.Conv2D(filters=growth_rate,
                                   kernel_size=3,
                                   padding='same',
                                   kernel_initializer=initializer,
                                   activation=None)
        l2_output = l2_layer(l2_output)

        # channel-wise concatenation
        # concatenate the outputs of the branches (input & output)
        X = K.layers.concatenate([X, l2_output])

        # # infer the number of channels in the output (after concat)
        # nb_filters += growth_rate

    # # activate the combined output
    # output = K.layers.Activation('relu')(output)

    return X, X.shape[-1]
    # this return also works:
    # return X, nb_filters
