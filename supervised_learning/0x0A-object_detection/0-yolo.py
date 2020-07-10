#!/usr/bin/env python3
"""
Initialize Yolo
"""
import tensorflow as tf


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
