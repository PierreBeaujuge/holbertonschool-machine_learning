#!/usr/bin/env python3

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

NST = __import__('5-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    np.random.seed(0)
    nst = NST(style_image, content_image)
    vgg19 = tf.keras.applications.vgg19
    preprocecced = vgg19.preprocess_input(nst.content_image * 255)
    style_outputs = nst.model(preprocecced)[:-1]
    style_cost = nst.style_cost(style_outputs)
    print(style_cost)
