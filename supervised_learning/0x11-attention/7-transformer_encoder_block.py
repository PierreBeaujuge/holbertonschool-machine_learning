#!/usr/bin/env python3
"""
7-transformer_encoder_block.py
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """class that instantiates an Encoder block"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """constructor"""
        super(EncoderBlock, self).__init__()

        # Instantiate a multi-head attention block
        self.mha = MultiHeadAttention(dm, h)

        # Set up the 'Point wise feed forward network'
        # consisting of two fully-connected layers with
        # a ReLU activation in between
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # dense_hidden: shape (batch_size, seq_len, hidden)
        self.dense_output = tf.keras.layers.Dense(dm)
        # dense_output: shape (batch_size, seq_len, dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """function that builds the Encoder block"""

        # MHA block
        attention_output, _ = self.mha(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        # Note: sum with 'x' via the residual connection
        attention_output = self.layernorm1(attention_output + x)

        # Pass attention_output to the 'Point wise feed forward network' (ffn)
        ffn_output = self.dense_hidden(attention_output)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        # Note: sum with 'attention_output' via the residual connection
        encoder_output = self.layernorm2(ffn_output + attention_output)

        # Note: shape of all output tensors (batch_size, input_seq_len, dm)

        return encoder_output
