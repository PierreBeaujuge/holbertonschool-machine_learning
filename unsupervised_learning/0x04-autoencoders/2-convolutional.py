#!/usr/bin/env python3
"""
2-convolutional.py
"""
import tensorflow.keras as keras
K = keras


def autoencoder(input_dims, filters, latent_dims):
    """function that instantiates a convolutional autoencoder instance"""

    # Define the encoder model
    encoder_inputs = K.Input(shape=input_dims)
    for i in range(len(filters)):
        layer = K.layers.Conv2D(filters=filters[i],
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding='same',
                                activation='relu')
        if i == 0:
            outputs = layer(encoder_inputs)
        else:
            outputs = layer(outputs)
        layer = K.layers.MaxPool2D(pool_size=(2, 2),
                                   strides=None,
                                   padding='same')
        outputs = layer(outputs)
    encoder = K.models.Model(inputs=encoder_inputs, outputs=outputs)

    # Define the decoder model
    decoder_inputs = K.Input(shape=latent_dims)
    for i in range(len(filters) - 1, -1, -1):
        if i != 0:
            layer = K.layers.Conv2D(filters=filters[i],
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same',
                                    activation='relu')
            if i == len(filters) - 1:
                outputs = layer(decoder_inputs)
            else:
                outputs = layer(outputs)
        else:
            layer = K.layers.Conv2D(filters=filters[i],
                                    kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='valid',
                                    activation='relu')
            outputs = layer(outputs)
        layer = K.layers.UpSampling2D(size=(2, 2))
        outputs = layer(outputs)
    layer = K.layers.Conv2D(filters=input_dims[-1],
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='sigmoid')
    outputs = layer(outputs)
    decoder = K.models.Model(inputs=decoder_inputs, outputs=outputs)

    # Define the autoencoder
    outputs = encoder(encoder_inputs)
    outputs = decoder(outputs)
    auto = K.models.Model(inputs=encoder_inputs, outputs=outputs)

    # Print the model summaries
    # encoder.summary()
    # decoder.summary()
    # auto.summary()

    # Compile the autoencoder
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
