#!/usr/bin/env python3
"""
8-transformer_decoder_block.py
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """class that instantiates a Decoder block"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """constructor"""
        super(DecoderBlock, self).__init__()

        # Instantiate a multi-head attention block
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)

        # Set up the 'Point wise feed forward network'
        # consisting of two fully-connected layers with
        # a ReLU activation in between
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # dense_hidden: shape (batch_size, seq_len, hidden)
        self.dense_output = tf.keras.layers.Dense(dm)
        # dense_output: shape (batch_size, seq_len, dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """function that builds the Decoder block"""

        # First MHA block
        attention_output1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention_output1 = self.dropout1(attention_output1, training=training)
        # Note: sum with 'x' via the residual connection
        attention_output1 = self.layernorm1(attention_output1 + x)

        # Second MHA block
        attention_output2, _ = self.mha2(attention_output1, encoder_output,
                                         encoder_output, padding_mask)
        attention_output2 = self.dropout2(attention_output2, training=training)
        # Note: sum with 'attention_output1' via the residual connection
        attention_output2 = self.layernorm2(attention_output2 +
                                            attention_output1)

        # Pass attention_output2 to the 'Point wise feed forward network' (ffn)
        ffn_output = self.dense_hidden(attention_output2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        # Note: sum with 'attention_output2' via the residual connection
        decoder_output = self.layernorm3(ffn_output + attention_output2)

        # Note: shape of all output tensors (batch_size, input_seq_len, dm)

        return decoder_output
