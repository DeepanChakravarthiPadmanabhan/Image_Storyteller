import tensorflow as tf

class BahdanauAttention(tf.keras.Model):
    '''
    Bahdanau Attention. Main paper: https://arxiv.org/pdf/1409.0473.pdf
    Neural Machine Translation by Jointly Learning to Align and Translate.

    Useful material for Bahdanau Attention: https://blog.floydhub.com/attention-mechanism/

    Based on the paper: https://arxiv.org/abs/1406.1078v1
    Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation.
    '''

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder, input) shape == (batch_size, 64, embedding_dim)

        # hidden_shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)

        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score_shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features  # context_vector shape: (bs, 64, 256)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # context_vector shape: (bs, 256)

        return context_vector, attention_weights
