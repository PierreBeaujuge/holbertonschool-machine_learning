#!/usr/bin/env python3
"""
Learning Rate Decay
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """function that updates the learning rate using inverse time decay"""
    alpha = alpha / (1 + decay_rate * int(global_step / decay_step))
    return alpha
