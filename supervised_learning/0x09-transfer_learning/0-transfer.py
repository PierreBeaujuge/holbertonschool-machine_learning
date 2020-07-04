#!/usr/bin/env python3
"""
Transfer Learning

The following script can be run on Google Colab
with the following libraries:

!python3 --version
Python 3.6.9
print(tf.__version__)
2.2.0
print(K.__version__)
2.3.0-tf
print(np.__version__)
1.18.5
print(matplotlib.__version__)
3.2.2

Alternative library versioning may incur errors
due to compatibility issues between deprecated
libraries on Google Colab
"""
import tensorflow as tf
import tensorflow.keras as K
import numpy as np


def preprocess_data(X, Y):
    """function that pre-processes the data"""
    X = K.applications.densenet.preprocess_input(X)
    Y = K.utils.to_categorical(Y)
    return X, Y


if __name__ == '__main__':

    # load the Cifar10 dataset, 50,000 training images and 10,000 test images
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # preprocess the data using the application's preprocess_input method
    # and convert the labels to one-hot encodings
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # instantiate a pre-trained model from the Keras API
    input_tensor = K.Input(shape=(32, 32, 3))

    # upsampling helps improve the validation accuracy to some extent
    # (insufficient here):
    # output = K.layers.UpSampling2D(size=(2, 2),
    #                                interpolation='nearest')(input_tensor)

    # another approach: resize images to the image size upon which the network
    # was pre-trained:
    resized_images = K.layers.Lambda(
        lambda image: tf.image.resize(image, (224, 224)))(input_tensor)

    base_model = K.applications.DenseNet201(include_top=False,
                                            weights='imagenet',
                                            input_tensor=resized_images,
                                            input_shape=(224, 224, 3),
                                            pooling='max',
                                            classes=1000)
    output = base_model.layers[-1].output
    base_model = K.models.Model(inputs=input_tensor, outputs=output)

    # extract the bottleneck features (output feature maps)
    # from the pre-trained network (here, base-model)
    train_datagen = K.preprocessing.image.ImageDataGenerator()
    train_generator = train_datagen.flow(x_train,
                                         y_train,
                                         batch_size=32,
                                         shuffle=False)
    features_train = base_model.predict(train_generator)

    # repeat the same operation with the test data (here used for validation)
    val_datagen = K.preprocessing.image.ImageDataGenerator()
    val_generator = val_datagen.flow(x_test,
                                     y_test,
                                     batch_size=32,
                                     shuffle=False)
    features_valid = base_model.predict(val_generator)

    # create a densely-connected head classifier

    # weights are initialized as per the he et al. method
    initializer = K.initializers.he_normal()
    input_tensor = K.Input(shape=features_train.shape[1])

    layer_256 = K.layers.Dense(units=256,
                               activation='elu',
                               kernel_initializer=initializer,
                               kernel_regularizer=K.regularizers.l2())
    output = layer_256(input_tensor)
    dropout = K.layers.Dropout(0.5)
    output = dropout(output)

    softmax = K.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer=initializer,
                             kernel_regularizer=K.regularizers.l2())
    output = softmax(output)

    model = K.models.Model(inputs=input_tensor, outputs=output)

    # compile the densely-connected head classifier
    model.compile(optimizer=K.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # reduce learning rate when val_accuracy has stopped improving
    lr_reduce = K.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                              factor=0.6,
                                              patience=2,
                                              verbose=1,
                                              mode='max',
                                              min_lr=1e-7)

    # stop training when val_accuracy has stopped improving
    early_stop = K.callbacks.EarlyStopping(monitor='val_accuracy',
                                           patience=3,
                                           verbose=1,
                                           mode='max')

    # callback to save the Keras model and (best) weights obtained
    # on an epoch basis
    checkpoint = K.callbacks.ModelCheckpoint('cifar10.h5',
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_weights_only=False,
                                             save_best_only=True,
                                             mode='max',
                                             save_freq='epoch')

    # train the densely-connected head classifier
    history = model.fit(features_train, y_train,
                        batch_size=32,
                        epochs=20,
                        verbose=1,
                        callbacks=[lr_reduce, early_stop, checkpoint],
                        validation_data=(features_valid, y_test),
                        shuffle=True)
