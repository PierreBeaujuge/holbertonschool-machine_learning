#!/usr/bin/env python3
import tensorflow as tf
discriminator = __import__('1-discriminator').discriminator


if __name__ == "__main__":

    X = tf.placeholder(tf.float32, shape=(None, 784), name='X')
    Y = discriminator(X)

    print(Y)
