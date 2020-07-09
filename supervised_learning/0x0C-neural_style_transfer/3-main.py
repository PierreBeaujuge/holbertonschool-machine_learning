#!/usr/bin/env python3

import matplotlib.image as mpimg

NST = __import__('3-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    nst = NST(style_image, content_image)
    print(nst.gram_style_features)
    print(nst.content_feature)
