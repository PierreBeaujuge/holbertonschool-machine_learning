#!/usr/bin/env python3
"""
Train a model, with learning rate decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""

    early_stop = None
    lr_decay = None

    if validation_data and early_stopping:
        # The patience parameter is the number of epochs
        # upon which improvement should be checked
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
    if validation_data and learning_rate_decay:

        def schedule(epoch):
            """
            function that takes an epoch index as input (integer,
            indexed from 0) and returns a new learning rate as output (float)
            """
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(
            schedule=schedule, verbose=1)

    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=[early_stop, lr_decay])
    return history
