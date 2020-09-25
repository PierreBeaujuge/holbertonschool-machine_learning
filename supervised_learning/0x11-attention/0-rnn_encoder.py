#!/usr/bin/env python3
"""
0-rnn_encoder.py
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """class that instantiates a RNN Encoder"""

    def __init__(self, vocab, embedding, units, batch):
        """constructor"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """function that initializes the hidden state to a tensor of zeros"""

        initializer = tf.keras.initializers.Zeros()
        tensor = initializer(shape=(self.batch, self.units))

        return tensor

    def call(self, x, initial):
        """function that builds the encoder"""

        # Compute the embeddings
        embeddings = self.embedding(x)
        # print("embeddings.shape:", embeddings.shape)
        # embeddings --> shape (batch, input_seq_len, embedding)

        # Pass the embeddings on to the GRU layer
        full_seq_outputs, last_hidden_state = self.gru(embeddings,
                                                       initial_state=initial)
        # full_seq_outputs --> shape (batch, input_seq_len, units)
        # last_hidden_state --> shape (batch, units)

        return full_seq_outputs, last_hidden_state
