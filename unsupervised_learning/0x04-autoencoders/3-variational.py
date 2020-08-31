#!/usr/bin/env python3
"""
3-variational.py
"""
import tensorflow.keras as keras
K = keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """function that instantiates a VAE instance"""

    # Define the encoder model
    encoder_inputs = K.Input(shape=(input_dims,))
    # inputs = K.Input(shape=input_dims)
    for i in range(len(hidden_layers)):
        layer = K.layers.Dense(units=hidden_layers[i], activation='relu')
        if i == 0:
            outputs = layer(encoder_inputs)
        else:
            outputs = layer(outputs)
    # Reparameterization trick
    layer = K.layers.Dense(units=latent_dims)
    mean = layer(outputs)
    layer = K.layers.Dense(units=latent_dims)
    logvar = layer(outputs)

    def sample(alist):
        """sample z"""
        mean, logvar = alist
        eps = K.backend.random_normal(shape=K.backend.shape(mean))
        z = mean + K.backend.exp(0.5 * logvar) * eps
        return z

    z = K.layers.Lambda(sample, output_shape=(latent_dims,))([mean, logvar])
    encoder = K.models.Model(inputs=encoder_inputs, outputs=[z, mean, logvar])

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

    # Print the model summaries
    # encoder.summary()
    # decoder.summary()
    # auto.summary()

    def compute_loss(inputs, outputs):
        """cost function"""
        loss = K.backend.binary_crossentropy(inputs, outputs)
        loss = K.backend.sum(loss, axis=1)
        KL_divergence = -0.5 * K.backend.sum(1 + logvar
                                             - K.backend.square(mean)
                                             - K.backend.exp(logvar),
                                             axis=-1)
        return loss + KL_divergence

    # Compile the autoencoder
    auto.compile(optimizer='Adam', loss=compute_loss)

    return encoder, decoder, auto
