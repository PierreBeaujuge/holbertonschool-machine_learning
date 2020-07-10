#!/usr/bin/env python3

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

NST = __import__('7-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    np.random.seed(0)
    nst = NST(style_image, content_image)
    generated_image = np.random.uniform(size=nst.content_image.shape)
    generated_image = tf.cast(generated_image, tf.float32)
    J_total, J_content, J_style = nst.total_cost(generated_image)
    print(J_total)
    print(J_content)
    print(J_style)
