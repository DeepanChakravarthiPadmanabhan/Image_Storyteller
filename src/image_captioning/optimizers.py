import tensorflow as tf

def get_Adam(lr=0.0001):
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    return optimizer

def get_SGD(lr=0.0001):
    optimizer = tf.keras.optimizers.SGD(lr=lr)
    return optimizer

def get_RMSprop(lr=0.0001):
    optimizer = tf.keras.optimizers.RMSprop(lr=lr)
    return optimizer

def select_optimizer(optimizer):
    if optimizer == 'Adam':
        return get_Adam()

    elif optimizer == 'SGD':
        return get_SGD()

    else:
        return get_RMSprop()
