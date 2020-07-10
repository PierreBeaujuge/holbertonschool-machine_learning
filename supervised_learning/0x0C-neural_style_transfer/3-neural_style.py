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

        # Load the VGG19 model for the cost calculation
        self.load_model()

        # Instantiate the gram_style_features and content_feature attributes
        self.generate_features()

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

    def load_model(self):
        """function that instantiates a VGG19 model from Keras"""

        # Instantiate a base model, without the top classifier
        # default input size for this model is 224 x 224
        base_vgg = tf.keras.applications.VGG19(include_top=False,
                                               weights='imagenet',
                                               input_tensor=None,
                                               input_shape=None,
                                               pooling=None,
                                               classes=1000)
        # "Replace" the MaxPooling layers of the model by AvgPooling layers by
        # passing them to the loading mechanism via the custom_objects argument
        custom_object = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        base_vgg.save('base_vgg')
        vgg = tf.keras.models.load_model('base_vgg',
                                         custom_objects=custom_object)
        # This way of freezing the network does not work:
        # vgg.trainable = False
        # Freeze the model layer by layer instead
        for layer in vgg.layers:
            layer.trainable = False

        # Extract the output feature templates of the desired layers
        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output
        # Combine the output feature templates by layer concatenation
        outputs = style_outputs + [content_output]
        # Instantiate the custom/recomposed model using the outputs list
        # and save the model in a "model" instance
        self.model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    @staticmethod
    def gram_matrix(input_layer):
        """function that calculates a Gram matrix, taking a layer as input"""
        """
        the style of an image can be described by the means and correlations
        across the different feature maps. A Gram matrix that includes this
        across the different feature maps. A Gram matrix that includes this
        feature vector with itself at each location, and averaging that outer
        product over all locations.
        """

        err = "input_layer must be a tensor of rank 4"
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)):
            raise TypeError(err)
        # if input_layer.ndim != 4:
        # note: tf.Variable does not have a "ndim" attribute! (error raised)
        if len(input_layer.shape) != 4:
            raise TypeError(err)

        # Compute the outer product of the input tensor (feature map)
        # input_layer with itself
        result = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # Average over all the elements ("pixels" or "grid cell") of
        # the feature map
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        result = result / num_locations

        return result

    def generate_features(self):
        """function that extracts the style and content features
        used to calculate the neural style cost"""

        # Preprocess the content and style input images (rescale
        # pixels to 255 prior to preprocessing as per vgg19 model reqs)
        style_image = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255)
        content_image = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255)
        # Extract the actual output features from the model passing it
        # the style_image and the content_image
        outputs_style = self.model(style_image)
        # note: style_outputs is a list of tensors
        style_outputs = outputs_style[:-1]
        # print("style_outputs:", np.array(outputs_style[:-1]).shape,
        #       np.array(outputs_style[:-1]))
        outputs_content = self.model(content_image)
        # note: content_output is a tensor
        content_ouput = outputs_content[-1]
        # print("content_output:", np.array(outputs_content[-1]).shape)
        # Create the list of gram matrices calculated from the style layer
        # outputs of the style image
        self.gram_style_features = [self.gram_matrix(style_output)
                                    for style_output in style_outputs]
        # Content layer output of the content_image
        self.content_feature = content_ouput
