import tensorflow as tf
from attention import BahdanauAttention

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,  # Dimension of output space.
                                       return_sequences=True,
                                       # whether to return the last output in the output sequence or full sequence.
                                       return_state=True,  # whether to return the last state in addition to the output.
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.int32, name='x'),
    #        tf.TensorSpec(shape=[None, 64, 256], dtype=tf.float32, name="features"),
    #        tf.TensorSpec(shape=[None, 512], dtype=tf.float32, name="hidden")])
    def call(self, x, features, hidden):
        # (x, features, hidden) shape = (bs, 1) (bs, 64, embedding_size) (bs, units)
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)
        # context_vector shape: (bs, 256)
        # attention_weights shape: (bs, 64, 1)

        # x shape before embedding == (batch_size, 1)
        # 1 is the previous word index value.
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)  # x shape: (bs, 1, 256)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)  # x shape: (bs, 1, 512)

        # passing the concatenated vector to the GRU
        # output shape == (bs, 1, 512)
        # state shape == (bs, 512)
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)  # x shape: (bs, 1, 512)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))  # x shape: (bs, 512)

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)  # x shape: (bs, 5001)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
