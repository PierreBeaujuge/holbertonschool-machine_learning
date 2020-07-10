#!/usr/bin/env python3

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

NST = __import__('6-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    np.random.seed(0)
    nst = NST(style_image, content_image)
    generated_image = np.random.uniform(size=nst.content_image.shape)
    generated_image = generated_image.astype('float32')
    vgg19 = tf.keras.applications.vgg19
    preprocecced = vgg19.preprocess_input(generated_image * 255)
    outputs = nst.model(preprocecced)
    content_output = outputs[-1][0]
    content_cost = nst.content_cost(content_output)
    print(content_cost)
