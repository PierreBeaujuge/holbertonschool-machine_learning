#!/usr/bin/env python3
import tensorflow as tf
train_discriminator = __import__('2-train_discriminator').train_discriminator


if __name__ == "__main__":

    Z = tf.placeholder(tf.float32, shape=(5, 100), name='Z')
    X = tf.placeholder(tf.float32, shape=(5, 784), name='X')

    loss, train = train_discriminator(Z, X)
    print(loss)
    print(train)
