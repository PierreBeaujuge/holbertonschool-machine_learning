#!/usr/bin/env python3
"""
Inception Network
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    function that builds an inception network
    as described in Going Deeper with Convolutions (2014)
    """
    initializer = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))

    layer_1 = K.layers.Conv2D(filters=64,
                              kernel_size=7,
                              padding='same',
                              strides=2,
                              kernel_initializer=initializer,
                              activation='relu')
    output_1 = layer_1(X)
    layer_2 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)
    output_2 = layer_2(output_1)
    layer_3R = K.layers.Conv2D(filters=64,
                               kernel_size=1,
                               padding='same',
                               strides=1,
                               kernel_initializer=initializer,
                               activation='relu')
    output_3R = layer_3R(output_2)
    layer_3 = K.layers.Conv2D(filters=192,
                              kernel_size=3,
                              padding='same',
                              strides=1,
                              kernel_initializer=initializer,
                              activation='relu')
    output_3 = layer_3(output_3R)
    layer_4 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)
    output_4 = layer_4(output_3)
    output_5 = inception_block(output_4, [64, 96, 128, 16, 32, 32])
    output_6 = inception_block(output_5, [128, 128, 192, 32, 96, 64])
    layer_7 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)
    output_7 = layer_7(output_6)
    output_8 = inception_block(output_7, [192, 96, 208, 16, 48, 64])
    output_9 = inception_block(output_8, [160, 112, 224, 24, 64, 64])
    output_10 = inception_block(output_9, [128, 128, 256, 24, 64, 64])
    output_11 = inception_block(output_10, [112, 144, 288, 32, 64, 64])
    output_12 = inception_block(output_11, [256, 160, 320, 32, 128, 128])
    layer_13 = K.layers.MaxPool2D(pool_size=3,
                                  padding='same',
                                  strides=2)
    output_13 = layer_13(output_12)
    output_14 = inception_block(output_13, [256, 160, 320, 32, 128, 128])
    output_15 = inception_block(output_14, [384, 192, 384, 48, 128, 128])
    layer_16 = K.layers.AvgPool2D(pool_size=7,
                                  padding='same',
                                  strides=None)
    output_16 = layer_16(output_15)

    layer_17 = K.layers.Dropout(0.4)
    output_17 = layer_17(output_16)

    # output_17 is now 1 x 1: no need to flatten the data
    # layer_18 = K.layers.Flatten()
    # output = layer_18(output_17)

    # here pass 'softmax' activation to the model
    # prior to compiling/training the model (not recommended)
    layer_18 = K.layers.Dense(units=1000,
                              activation='softmax',
                              kernel_initializer=initializer,
                              kernel_regularizer=K.regularizers.l2())
    output_18 = layer_18(output_17)

    # instantiate a model from the Model class
    model = K.models.Model(inputs=X, outputs=output_18)

    return model
