#!/usr/bin/env python3
"""
Train a model, with early stopping
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""
    early_stop = None
    if validation_data and early_stopping:
        # The patience parameter is the number of epochs
        # upon which improvement should be checked
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=[early_stop])
    return history
