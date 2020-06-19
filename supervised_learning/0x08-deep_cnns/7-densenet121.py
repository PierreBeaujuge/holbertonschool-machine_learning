#!/usr/bin/env python3
"""
DenseNet-121
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    function that builds a DenseNet-121 network
    as described in Densely Connected Convolutional Networks
    """
    initializer = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))

    norm_1 = K.layers.BatchNormalization()
    output_1 = norm_1(X)
    activ_1 = K.layers.Activation('relu')
    output_1 = activ_1(output_1)
    layer_1 = K.layers.Conv2D(filters=64,
                              kernel_size=7,
                              padding='same',
                              strides=2,
                              kernel_initializer=initializer,
                              activation=None)
    output_1 = layer_1(output_1)

    layer_2 = K.layers.MaxPool2D(pool_size=3,
                                 padding='same',
                                 strides=2)
    output_2 = layer_2(output_1)

    db1_output = dense_block(output_2, output_2.shape[-1], growth_rate, 6)
    tl1_output = transition_layer(
        db1_output[0], int(db1_output[1]), compression)

    db2_output = dense_block(tl1_output[0], tl1_output[1], growth_rate, 12)
    tl2_output = transition_layer(
        db2_output[0], int(db2_output[1]), compression)

    db3_output = dense_block(tl2_output[0], tl2_output[1], growth_rate, 24)
    tl3_output = transition_layer(
        db3_output[0], int(db3_output[1]), compression)

    db4_output = dense_block(tl3_output[0], tl3_output[1], growth_rate, 16)

    layer_3 = K.layers.AvgPool2D(pool_size=7,
                                 padding='same',
                                 strides=None)
    output_3 = layer_3(db4_output[0])

    # no need to flatten here (for some reason)
    # flatten = K.layers.Flatten()
    # output_3 = flatten(output_3)

    # here pass 'softmax' activation to the model
    # prior to compiling/training the model (not recommended)
    softmax = K.layers.Dense(units=1000,
                             activation='softmax',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2())
    output_4 = softmax(output_3)

    # instantiate a model from the Model class
    model = K.models.Model(inputs=X, outputs=output_4)

    # # compile the model
    # # here, define loss from activated ouput
    # # by default, tf.keras.losses.categorical_crossentropy assumes that
    # # y_pred encodes a probability distribution (from_logits=False)
    # # However - Using from_logits=True is more numerically stable
    # model.compile(optimizer=K.optimizers.Adam(),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    return model
