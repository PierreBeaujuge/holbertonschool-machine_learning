#!/usr/bin/env python3
"""
ResNet-50
"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    function that builds a ResNet-50 network
    as described in Deep Residual Learning for Image Recognition (2015)
    """
    initializer = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))

    layer_1 = K.layers.Conv2D(filters=64,
                              kernel_size=7,
                              padding='same',
                              strides=2,
                              kernel_initializer=initializer,
                              activation=None)
    output_1 = layer_1(X)
    norm_1 = K.layers.BatchNormalization()
    output_1 = norm_1(output_1)
    activ_1 = K.layers.Activation('relu')
    output_1 = activ_1(output_1)

    layer_2 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)
    output_2 = layer_2(output_1)
    output_2 = projection_block(output_2, [64, 64, 256], s=1)
    output_2 = identity_block(output_2, [64, 64, 256])
    output_2 = identity_block(output_2, [64, 64, 256])

    output_3 = projection_block(output_2, [128, 128, 512], s=2)
    output_3 = identity_block(output_3, [128, 128, 512])
    output_3 = identity_block(output_3, [128, 128, 512])
    output_3 = identity_block(output_3, [128, 128, 512])

    output_4 = projection_block(output_3, [256, 256, 1024], s=2)
    output_4 = identity_block(output_4, [256, 256, 1024])
    output_4 = identity_block(output_4, [256, 256, 1024])
    output_4 = identity_block(output_4, [256, 256, 1024])
    output_4 = identity_block(output_4, [256, 256, 1024])
    output_4 = identity_block(output_4, [256, 256, 1024])

    output_5 = projection_block(output_4, [512, 512, 2048], s=2)
    output_5 = identity_block(output_5, [512, 512, 2048])
    output_5 = identity_block(output_5, [512, 512, 2048])

    avg_pool = K.layers.AvgPool2D(pool_size=7,
                                  padding='same',
                                  strides=None)
    output_6 = avg_pool(output_5)

    # AvgPool2D reduced data to 1 x 1: no need to flatten here
    # flatten = K.layers.Flatten()
    # output_6 = flatten(output_6)

    # here pass 'softmax' activation to the model
    # prior to compiling/training the model (not recommended)
    softmax = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2())
    output_7 = softmax(output_6)

    # instantiate a model from the Model class
    model = K.models.Model(inputs=X, outputs=output_7)

    return model
