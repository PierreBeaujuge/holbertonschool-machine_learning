#!/usr/bin/env python3
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.keras as K
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import os
train_gan = __import__('5-train_GAN').train_gan


if __name__ == "__main__":

    # This method for loading the MNIST dataset is deprecated
    # mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

    # load the mnist dataset, 60,000 training images; using numpy
    lib = np.load('../data/MNIST.npz')
    x_train = lib['X_train']
    m = x_train.shape[0]
    x_train = x_train.reshape(m, -1)
    # print("x_train.shape:", x_train.shape)
    # print("x_train:", x_train)

    # # load the mnist dataset, 60,000 training images; using keras
    # x_train = K.datasets.mnist.load_data()[0][0]
    # # print("x_train.shape:", x_train.shape)
    # # print("x_train:", x_train)
    # # preprocess the data using the application's preprocess_input method
    # x_train = K.applications.densenet.preprocess_input(x_train)
    # m = x_train.shape[0]
    # x_train = x_train.reshape(m, -1)
    # # print("x_train.shape:", x_train.shape)
    # # print("x_train:", x_train)

    tf.set_random_seed(0)
    save_path = train_gan(x_train, 2000, 32, 100, save_path='./')
    print("Return path to the model: {}".format(save_path))
