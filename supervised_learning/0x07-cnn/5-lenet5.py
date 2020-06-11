#!/usr/bin/env python3
"""
LeNet-5 in Keras
"""
import tensorflow.keras as K


def lenet5(X):
    """function that builds a modified version of LeNet-5"""
    initializer = K.initializers.he_normal()
    layer = K.layers.Conv2D(filters=6,
                            kernel_size=5,
                            padding='same',
                            kernel_initializer=initializer,
                            activation='relu')
    output = layer(X)
    layer = K.layers.MaxPool2D(pool_size=2,
                               strides=2)
    output = layer(output)
    layer = K.layers.Conv2D(filters=16,
                            kernel_size=5,
                            padding='valid',
                            kernel_initializer=initializer,
                            activation='relu')
    output = layer(output)
    layer = K.layers.MaxPool2D(pool_size=2,
                               strides=2)
    output = layer(output)
    layer = K.layers.Flatten()
    output = layer(output)
    layer = K.layers.Dense(units=120,
                           activation='relu',
                           kernel_initializer=initializer)
    output = layer(output)
    layer = K.layers.Dense(units=84,
                           activation='relu',
                           kernel_initializer=initializer)
    output = layer(output)
    # here pass 'softmax' activation to the model
    # prior to compiling/training the model (not recommended)
    layer = K.layers.Dense(units=10,
                           activation='softmax',
                           kernel_initializer=initializer)
    output = layer(output)

    # instantiate a model from the Model class
    model = K.models.Model(inputs=X, outputs=output)

    # compile the model
    # here, define loss from activated ouput
    # by default, tf.keras.losses.categorical_crossentropy assumes that
    # y_pred encodes a probability distribution (from_logits=False)
    # However - Using from_logits=True is more numerically stable
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
