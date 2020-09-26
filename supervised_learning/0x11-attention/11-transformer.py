#!/usr/bin/env python3
"""
11-transformer.py
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """class that instantiates a Transformer"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """constructor"""
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask,
             look_ahead_mask, decoder_mask):
        """function that builds a Transformer"""

        # Instantiate the encoder part
        encoder_output = self.encoder(inputs, training, encoder_mask)
        # encoder_output: shape (batch_size, inp_seq_len, dm)

        # Instantiate the decoder part
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        # decoder_output: shape (batch_size, tar_seq_len, dm)

        # Pass the decoder_output on to the 'classfier' layer
        output = self.linear(decoder_output)
        # output: shape (batch_size, tar_seq_len, target_vocab_size)

        return output
