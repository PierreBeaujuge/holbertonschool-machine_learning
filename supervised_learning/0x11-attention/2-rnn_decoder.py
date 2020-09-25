#!/usr/bin/env python3
"""
2-rnn_decoder.py
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """class that instantiates a RNN Decoder"""

    def __init__(self, vocab, embedding, units, batch):
        """constructor"""
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """function that builds the decoder"""

        # Instantiate a self_attention layer (takes arg: units)
        # s_prev: previous decoder hidden state; shape (batch, units)
        self_attention = SelfAttention(self.units)

        # Compute context vector and attention weights
        # s_prev: previous decoder hidden state; shape (batch, units)
        # hidden_states: encoder outputs; shape (batch, input_seq_len, units)
        context_vector, attention_weights = self_attention(s_prev,
                                                           hidden_states)

        # Compute the embeddings
        embeddings = self.embedding(x)
        # print("embeddings.shape:", embeddings.shape)
        # embeddings --> shape (batch, input_seq_len, embedding)

        # Reshape context
        context_vector = tf.expand_dims(context_vector, 1)

        # Concatenate embeddings and context vector --> decoder inputs
        inputs = tf.concat([context_vector, embeddings], axis=-1)

        # Pass the decoder inputs on to the GRU layer
        # initial = hidden_states[:, -1]
        decoder_outputs, last_hidden_state = self.gru(inputs)
        #                                               initial_state=initial)
        # print("decoder_outputs.shape:", decoder_outputs.shape)
        # print("last_hidden_state.shape:", last_hidden_state)
        # decoder_outputs --> shape (batch, input_seq_len, units)
        # last_hidden_state --> (batch, units)

        # Reshape decoder_outputs to (batch, units)
        y = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))
        # print("y.shape:", y.shape)
        # y --> (batch, units)
        # Final model output (output word) should be a one-hot
        # vector of 'vocab_size' dimensionality
        y = self.F(y)
        # print("self.F(y):", y)
        # print("self.F(y).shape:", y.shape)
        # y --> (batch, vocab)

        return y, last_hidden_state
