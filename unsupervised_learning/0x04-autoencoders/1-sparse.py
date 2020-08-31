#!/usr/bin/env python3
"""
1-sparse.py
"""
import tensorflow.keras as keras
K = keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """function that instantiates a sparse autoencoder instance"""

    # Define the encoder model
    encoder_inputs = K.Input(shape=(input_dims,))
    for i in range(len(hidden_layers)):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        if i == 0:
            outputs = layer(encoder_inputs)
        else:
            outputs = layer(outputs)
    layer = K.layers.Dense(units=latent_dims, activation='relu',
                           activity_regularizer=K.regularizers.l1(lambtha))
    outputs = layer(outputs)
    encoder = K.models.Model(inputs=encoder_inputs, outputs=outputs)

    # Define the decoder model
    decoder_inputs = K.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers) - 1, -1, -1):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        if i == len(hidden_layers) - 1:
            outputs = layer(decoder_inputs)
        else:
            outputs = layer(outputs)
    layer = K.layers.Dense(units=input_dims, activation='sigmoid')
    outputs = layer(outputs)
    decoder = K.models.Model(inputs=decoder_inputs, outputs=outputs)

    # Define the autoencoder
    outputs = encoder(encoder_inputs)
    outputs = decoder(outputs)
    auto = K.models.Model(inputs=encoder_inputs, outputs=outputs)

    # Compile the autoencoder
    auto.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, auto
