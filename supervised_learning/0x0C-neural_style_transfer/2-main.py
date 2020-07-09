#!/usr/bin/env python3

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

NST = __import__('2-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    np.random.seed(0)
    nst = NST(style_image, content_image)
    input_layer = tf.constant(np.random.randn(1, 28, 30, 3), dtype=tf.float32)
    gram_matrix = nst.gram_matrix(input_layer)
    print(gram_matrix)
