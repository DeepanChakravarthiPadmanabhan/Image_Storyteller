import tensorflow as tf

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle.
    # This encoder passes those features through a fully connected layer.

    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)
    # @tf.function
    def call(self, x):
        x = self.fc(x) # x input shape: (bs, 64, 2048)
        x = tf.nn.relu(x) # x output shape: (bs, 64, 256)
        return x 