#!/usr/bin/env python3
"""
L2 Regularization Cost
"""
import tensorflow as tf
import numpy as np


def l2_reg_cost(cost):
    """function that calculates the cost of a nn with L2 regularization"""
    return cost + tf.losses.get_regularization_losses()
