#!/usr/bin/env python3
"""
Process Outputs
"""
import tensorflow as tf
import numpy as np


class Yolo:
    """define the YOLO class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """define and initialize attributes and variables"""
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [class_name[:-1] for class_name in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """function that processes single-image predictions"""

        boxes = []
        box_confidences = []
        box_class_probs = []

        # Loop over the output feature maps (here 13x13, 26x26, 52x52)
        # so i ranges from 0 to 2
        for i, output in enumerate(outputs):
            # print("output {}:".format(i))
            grid_height = output.shape[0]
            # print(grid_height)
            grid_width = output.shape[1]
            # print(grid_width)
            anchor_boxes = output.shape[2]
            # print(anchor_boxes)
            boxs = output[..., :4]
            # print("boxes:", boxes.shape, boxes)
            # Extract the network output predictions ("raw" coordinates
            # and dimensions) to be processed into bounding box predictions
            t_x = boxs[..., 0]
            t_y = boxs[..., 1]
            t_w = boxs[..., 2]
            t_h = boxs[..., 3]
            # print("t_x:", t_x.shape, t_x)
            # print("t_y:", t_y.shape, t_y)
            # print("t_w:", t_w.shape, t_w)
            # print("t_h:", t_h.shape, t_h)

            # Create 3D arrays with the left-corner coordinates (c_x, c_y)
            # of each grid cell. Values added in the b_x, b_y formulae (below)
            # make a row vector of grid_width length
            c_x = np.arange(grid_width).reshape(1, grid_width)
            # print(c_x)
            # make a 2D array of grid_width columns and grid_height rows,
            # but do not transpose it
            c_x = np.repeat(c_x, grid_height, axis=0)
            # print(c_x)
            # add the third axis, duplicating the coordinate values by
            # anchor_boxes
            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=2)
            # print(c_x)
            # make a row vector of grid_width length
            c_y = np.arange(grid_width).reshape(1, grid_width)
            # print(c_y)
            # make a 2D array of grid_width columns and grid_height rows,
            # and transpose it
            c_y = np.repeat(c_y, grid_height, axis=0).T
            # print(c_y)
            # add the third axis, duplicating the coordinate values by
            # anchor_boxes
            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=2)
            # print(c_y)

            # The network output predictions are passed through a sigmoid
            # function, which squashes the output in a range from 0 to 1,
            # effectively keeping the center in the grid which is predicting.
            # Add the top-left coordinates of the grid (c_x and c_y),
            # because YOLO predicts offsets relative to the top-left corner
            # of the grid cell which is predicting the object.
            # The resultant predictions (b_x and b_y) are normalised by
            # the width and height of the grid, e.g. 13 x 13. i.e., if the
            # predictions b_x and b_y for the box containing the object
            # are (0.3, 0.8), the actual centre coordinates of the box
            # on the 13 x 13 feature map are (13 x 0.3, 13 x 0.8).
            b_x = (self.sigmoid(t_x) + c_x) / grid_width
            b_y = (self.sigmoid(t_y) + c_y) / grid_height

            # The dimensions of the bounding box (b_w, b_h) are predicted by
            # applying a log-space transform to the output, and then
            # multiplying with the anchor dimensions for the box.
            # The resultant predictions (b_w and b_h) are normalised by the
            # width and height of the image input to the model,
            # e.g. 416 x 416. i.e., if the predictions b_w and b_h for the
            # box containing the object are (0.4, 0.6), the actual width
            # and height of the box on the 416 x 416 image are
            # (416 x 0.4, 416 x 0.6).
            anchor_width = self.anchors[i, :, 0]
            anchor_height = self.anchors[i, :, 1]
            image_width = self.model.input.shape[1]
            image_height = self.model.input.shape[2]
            b_w = (anchor_width * np.exp(t_w)) / image_width
            b_h = (anchor_height * np.exp(t_h)) / image_height

            # Top-left corner coordinates of the bounding box
            x_1 = b_x - b_w / 2
            y_1 = b_y - b_h / 2
            # Bottom right-corner coordinates of the bounding box
            x_2 = x_1 + b_w
            y_2 = y_1 + b_h

            # Express the boundary box coordinates relative to
            # the original image
            x_1 *= image_size[1]
            y_1 *= image_size[0]
            x_2 *= image_size[1]
            y_2 *= image_size[0]

            # Update boxes according to the bounding box coordinates
            # inferred above
            boxs[..., 0] = x_1
            boxs[..., 1] = y_1
            boxs[..., 2] = x_2
            boxs[..., 3] = y_2
            # print(box)
            # Append the boxes coordinates to the boxes list
            boxes.append(boxs)

            # Extract the network output box_confidence prediction
            box_confidence = output[..., 4:5]
            # The prediction is passed through a sigmoid function,
            # which squashes the output in a range from 0 to 1,
            # to be interpreted as a probability.
            box_confidence = self.sigmoid(box_confidence)
            # print(box_confidence)
            # Append box_confidence to box_confidences
            box_confidences.append(box_confidence)

            # Extract the network ouput class_probability predictions
            classes = output[..., 5:]
            # The predictions are passed through a sigmoid function,
            # which squashes the output in a range from 0 to 1,
            # to be interpreted as a probability.
            # Note: before v3, YOLO used to softmax the class scores.
            # However, that design choice has been dropped in v3. The
            # reason is that Softmaxing class scores assume that the
            # classes are mutually exclusive. In simple words, if an object
            # belongs to one class, then it's guaranteed it cannot belong
            # to another class. Assumption that does not always hold true!
            classes = self.sigmoid(classes)
            # print(classes)
            # Append class_probability predictions to box_class_probs
            box_class_probs.append(classes)

        return (boxes, box_confidences, box_class_probs)

    def sigmoid(self, array):
        """define the sigmoid activation function"""
        return 1 / (1 + np.exp(-1 * array))
