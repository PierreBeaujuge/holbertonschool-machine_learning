#!/usr/bin/env python3
"""
File: utils.py
"""
import numpy as np
import cv2
import os
import csv


def load_images(images_path, as_array=True):
    """function that loads images from a directory or file, given the path"""

    filenames = []
    images = []

    # List the filenames in the directory pointed to by images_path
    # print("os.listdir(images_path):", os.listdir(images_path))
    # note: "os.listdir(images_path)" is a list!

    if len(os.listdir(images_path)):

        for filename in os.listdir(images_path):

            filenames.append(filename)
            # Compose the image path for cv2.imread
            image_path = os.path.join(images_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                # Convert BGR image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Image is appended as a np.ndarray
                images.append(image)
        # Sort images alphabetically
        indices = np.argsort(filenames)
        # print("indices:", indices)
        # note: "indices" is a np.ndarray!

        # Convert images to an np.ndarray before array indexing it
        images = np.array(images)[indices.astype(int)]
        if not as_array:
            images = images.tolist()
        # Convert filenames to an np.ndarray before array indexing it,
        # then convert back to a list
        filenames = np.array(filenames)[indices.astype(int)].tolist()

    # Convert the empty images list to an empty array if as_array
    # is True before returning images
    if not len(os.listdir(images_path)) and as_array:
        images = np.array(images)

    return (images, filenames)
