#!/usr/bin/env python3

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

NST = __import__('4-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    nst = NST(style_image, content_image)
    np.random.seed(0)
    vgg19 = tf.keras.applications.vgg19
    preprocecced = vgg19.preprocess_input(nst.content_image * 255)
    outputs = nst.model(preprocecced)
    layer_style_cost = nst.layer_style_cost(outputs[0],
                                            nst.gram_style_features[0])
    print(layer_style_cost)
