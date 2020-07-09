#!/usr/bin/env python3
"""
NST - Initialize
"""
import numpy as np
import tensorflow as tf


class NST:
    """class used to perform tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """define and initialize variables"""

        # After eager execution is enabled, operations are executed as they are
        # defined and Tensor objects hold concrete values, which can be
        # accessed as numpy.ndarray`s through the numpy() method.
        tf.enable_eager_execution()

        err_1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(err_1)
        if style_image.ndim != 3 or style_image.shape[-1] != 3:
            raise TypeError(err_1)
        err_2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(content_image, np.ndarray):
            raise TypeError(err_2)
        if content_image.ndim != 3 or content_image.shape[-1] != 3:
            raise TypeError(err_2)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Preprocessed style image
        self.style_image = self.scale_image(style_image)
        # Preprocessed content image
        self.content_image = self.scale_image(content_image)
        # Weight for content cost
        self.alpha = alpha
        # Weight for style cost
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        function that rescales an image such that its pixels values are
        between 0 and 1 and its largest side is 512 pixels
        """

        err = "image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(image, np.ndarray):
            raise TypeError(err)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise TypeError(err)

        # print("image:", type(image[0][0][0])) <-np.int, [0..255]

        # Impose image rescaling such that largest side is 512 pixels
        max_dim = 512
        # print("max_dim:", type(max_dim)) <-int
        long_dim = max(image.shape[:-1])
        # print("long_dim:", type(long_dim)) <-int
        scale = max_dim / long_dim
        # print("scale:", scale, type(scale)) <-float

        # Infer new_shape using the scale factor
        # print("image.shape[:-1]:", image.shape[:-1],
        #      type(image.shape[:-1][0])) <-int
        # new_shape = scale * image.shape[:-1] <- TypeError:
        # can't multiply sequence by non-int of type 'float'.
        # use map() to convert scale * tuple element products (floats)
        # to integers and recompose the tuple:
        new_shape = tuple(map(lambda x: int(scale * x), image.shape[:-1]))
        # print("new_shape:", new_shape, type(new_shape[0]))

        # Convert np.ndarray with shape (h, w, 3) to shape (1, h, w, 3)
        image = image[tf.newaxis, :]
        # print(image)
        # print(image.shape)

        # Resize image using bicubic interpolation, concurrently
        # converting np.ndarray to tf.tensor with shape (1, h_new, w_new, 3)
        # In Google Colab (tf 2.0):
        # image = tf.image.resize(image, (new_h, new_w), method='bicubic')
        # With tf 1.2:
        image = tf.image.resize_bicubic(image, new_shape)
        # print("Before clipping:", image)
        # print(image.shape)

        # Normalize image pixels to range [0..1]:
        image = image / 255
        # print("image:", type(image[0][0][0])) <-np.float, [0..1]

        # Since this is a float image, keep the pixel values between 0 and 1:
        # clip data to the valid range for plt.imshow with RGB data
        # ([0..1] for floats) <- required/requested by the script
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        # print("After clipping:", image)

        return image
