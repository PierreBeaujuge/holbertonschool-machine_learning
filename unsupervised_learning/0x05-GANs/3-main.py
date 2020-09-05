#!/usr/bin/env python3
import tensorflow as tf
train_generator = __import__('3-train_generator').train_generator


if __name__ == "__main__":

    Z = tf.placeholder(tf.float32, shape=(5, 100), name='Z')
    loss, train = train_generator(Z)

    print(loss)
    print(train)
