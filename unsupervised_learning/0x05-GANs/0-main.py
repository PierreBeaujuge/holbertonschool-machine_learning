#!/usr/bin/env python3
import tensorflow as tf
generator = __import__('0-generator').generator


if __name__ == "__main__":

    Z = tf.placeholder(tf.float32, shape=(None, 100), name='Z')
    X = generator(Z)

    print(X)
