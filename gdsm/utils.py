import tensorflow as tf


def my_reduce_prod(input_tensor, axis=None, keep_dims=False, name=None):

    temp = tf.log(input_tensor)
    temp = tf.reduce_sum(temp, axis=axis, keep_dims=keep_dims)
    return tf.exp(temp, name=name)
