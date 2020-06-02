#!/usr/bin/env python3
"""
Train a model, with save the best iteration
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""

    callbacks = []

    if validation_data and early_stopping:
        # The patience parameter is the number of epochs
        # upon which improvement should be checked
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
        callbacks.append(early_stop)

    if validation_data and learning_rate_decay:

        def schedule(epoch):
            """
            function that takes an epoch index as input (integer,
            indexed from 0) and returns a new learning rate as output (float)
            """
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(
            schedule=schedule, verbose=1)
        callbacks.append(lr_decay)

    if save_best:
        # Create a callback that saves the entire model:
        # architecture, weights, and training configuration
        # continually saving the model both during and at the end of training
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath, monitor='val_loss', verbose=0,
            save_best_only=True, save_weights_only=False,
            mode='auto', period=1)
        callbacks.append(checkpoint)

    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)

    return history
