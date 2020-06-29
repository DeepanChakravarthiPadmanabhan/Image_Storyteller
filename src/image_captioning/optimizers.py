import tensorflow as tf

def get_Adam(lr):
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    return optimizer

def get_SGD(lr):
    optimizer = tf.keras.optimizers.SGD(lr=lr)
    return optimizer

def get_RMSprop(lr):
    optimizer = tf.keras.optimizers.RMSprop(lr=lr)
    return optimizer

def select_optimizer(optimizer, learning_rate):
    if optimizer == 'Adam':
        return get_Adam(learning_rate)

    elif optimizer == 'SGD':
        return get_SGD(learning_rate)

    else:
        return get_RMSprop(learning_rate)
