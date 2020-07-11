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

    def __init__(self, style_image, content_image, alpha=1e4, beta=1, var=10):
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
        # Weight for the variational cost
        self.var = var

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

    def layer_style_cost(self, style_output, gram_target):
        """function that calculates the style cost for a single
        style_output layer"""

        # Number of channels in style_output
        c = style_output.shape[-1]
        err_1 = "style_output must be a tensor of rank 4"
        if not isinstance(style_output, (tf.Tensor, tf.Variable)):
            raise TypeError(err_1)
        if len(style_output.shape) != 4:
            raise TypeError(err_1)
        err_2 = ("gram_target must be a tensor of shape [1, {}, {}]".
                 format(c, c))
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)):
            raise TypeError(err_2)
        if gram_target.shape != (1, c, c):
            raise TypeError(err_2)

        # Compute the gram matrix of the style_output layer
        gram_style = self.gram_matrix(style_output)
        # Calculate the mean squared error between gram_style and gram_target
        style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return style_cost

    def style_cost(self, style_outputs):
        """function that calculates the style cost for all
        style_output layers"""

        err = "style_outputs must be a list with a length of {}".format(
            len(self.style_layers))
        if not isinstance(style_outputs, list):
            raise TypeError(err)
        if len(self.style_layers) != len(style_outputs):
            raise TypeError(err)

        # Reminders:
        # style_layers is a list of name strings
        # style_outputs is a list of tensors

        style_costs = []
        # each layer should be weighted evenly with all weights summing to 1
        weight = 1 / len(self.style_layers)

        for style_output, gram_target in zip(
                style_outputs, self.gram_style_features):

            layer_style_cost = self.layer_style_cost(style_output, gram_target)
            weighted_layer_style_cost = weight * layer_style_cost
            style_costs.append(weighted_layer_style_cost)

        # Add all the tensor values from the list
        style_cost = tf.add_n(style_costs)

        return style_cost

    def content_cost(self, content_output):
        """function that calculates the content cost for content_output"""

        # Reminder:
        # content layer output of the content_image is
        # self.content_feature (tensor)

        # Convert content_output to a tensor of 4 dimensions
        # to match the shape of self.content_feature
        # if content_output.ndim == 3:
        #     content_output = content_output[tf.newaxis, ...]
        # print("content_output.shape:", content_output.shape)

        err = "content_output must be a tensor of shape {}".format(
            self.content_feature.shape)
        if not isinstance(content_output, (tf.Tensor, tf.Variable)):
            raise TypeError(err)
        if content_output.shape != self.content_feature.shape:
            raise TypeError(err)

        # Calculate the mean squared error between content_output and
        # self.content_feature
        content_cost = tf.reduce_mean(tf.square(
            content_output - self.content_feature))

        return content_cost

    def total_cost(self, generated_image):
        """function that calculates the total cost for the generated image"""

        err = "generated_image must be a tensor of shape {}".format(
            self.content_image.shape)
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(err)
        if generated_image.shape != self.content_image.shape:
            raise TypeError(err)

        # Preprocess the "generated" input image (initially: random noise)
        # (rescale pixels to 255 prior to preprocessing as per
        # vgg19 model reqs)
        generated_image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255)
        # Extract the actual output features from the model passing it
        # the generated_image
        outputs_generated = self.model(generated_image)
        # note: style_outputs should be a list of tensors
        style_outputs = outputs_generated[:-1]
        # print("style_outputs:", np.array(outputs_generated[:-1]).shape,
        #       np.array(outputs_generated[:-1]))
        # note: content_output should be a tensor
        content_ouput = outputs_generated[-1]
        # print("content_output:", np.array(outputs_generated[-1]).shape)

        # Evaluate the style_cost and content_cost from the output features
        # of the generated_image
        style_cost = self.style_cost(style_outputs)
        content_cost = self.content_cost(content_ouput)

        # Evaluate the variational cost from the generated_image
        # in a call to variational_cost()
        var_cost = self.variational_cost(generated_image)

        total_cost = (self.alpha * content_cost + self.beta * style_cost
                      + self.var * var_cost)

        return (total_cost, content_cost, style_cost, var_cost)

    def compute_grads(self, generated_image):
        """function that computes the gradients for the generated image"""

        err = "generated_image must be a tensor of shape {}".format(
            self.content_image.shape)
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(err)
        if generated_image.shape != self.content_image.shape:
            raise TypeError(err)

        # Note: in the main file, generated_image is defined as
        # a tf.Variable to contain the image to optimize. It is initialized
        # with self.content_image (the tf.Variable must be the same shape
        # as the content image)

        # Use tf.GradientTape to update the generated_image (float image)
        with tf.GradientTape() as tape:
            # Calculate the loss in a call to total_cost()
            loss = self.total_cost(generated_image)
            total_cost, content_cost, style_cost, var_cost = loss

        # Infer the gradients passing in the loss and the generated_image
        gradients = tape.gradient(total_cost, generated_image)

        return (gradients, total_cost, content_cost, style_cost, var_cost)

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """function that generates the neural style transfered image"""

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError(
                    "step must be positive and less than iterations")
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        # The generated_image is defined as a tf.Variable to contain
        # the image to optimize. It is initialized with self.content_image
        # (the tf.Variable must be the same shape as the content image)
        generated_image = tf.Variable(self.content_image)
        # Define the optimizer for training
        # In tf 2.0:
        # optimizer = tf.optimizers.Adam(learning_rate=lr,
        #                                beta_1=beta1, beta_2=beta2)
        # In tf 1.2:
        optimizer = tf.train.AdamOptimizer(learning_rate=lr,
                                           beta1=beta1, beta2=beta2)

        # Initialize the variables used to keep track of the best
        # cost and image, epoch by epoch
        prev_total_cost = float('inf')
        prev_image = generated_image

        # Set the range of epochs
        for i in range(iterations + 1):

            # Compute the gradients and return the various costs
            # in a call to compute_grads()
            computed = self.compute_grads(generated_image)
            (gradients, total_cost, content_cost,
             style_cost, var_cost) = computed

            # Print the costs at every "step"
            if i % step == 0 or i == iterations:
                strg = "Cost at iteration {}: {}, content {}, style {}, var {}"
                print(strg.format(i, total_cost, content_cost,
                                  style_cost, var_cost))

            # Backpropagation pass
            if i != iterations:
                # The generated_image (tensor, float image) is updated
                optimizer.apply_gradients([(gradients, generated_image)])
                # Then clipped to stay in the range [0..1]
                clipped_image = tf.clip_by_value(
                    generated_image, clip_value_min=0, clip_value_max=1)
                generated_image.assign(clipped_image)

            # Update prev_total_cost (best cost) and prev_image (best image)
            if total_cost <= prev_total_cost:
                prev_total_cost = total_cost
                # print("prev_total_cost:", "{}".format(prev_total_cost))
                prev_image = generated_image

        # Convert the tensors into numpy objects
        cost = prev_total_cost.numpy()
        # note: grab the tensor image (rank 4) at index,
        # to convert it to a tensor image of rank 3
        generated_image = prev_image[0].numpy()

        return (generated_image, cost)

    @staticmethod
    def variational_cost(generated_image):
        """function that calculates the variational cost
        for the generated image"""

        # The following standard tf implementation returns an numpy array
        # of this type: array([149419.88], dtype=float32):
        # loss = tf.image.total_variation(generated_image).numpy()
        # This returns a tf tensor of this type
        # tf.Tensor([8765200.], shape=(1,), dtype=float32):
        loss = tf.image.total_variation(generated_image)[0]
        # print("loss:", loss)

        return loss
