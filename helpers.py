import tensorflow as tf


def convolve(x, w, strides):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.005, shape=shape)
    return tf.Variable(initial)


def max_pool(x, size):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='VALID')


def relu(feats):
    return tf.nn.relu(feats)


def dropout(x, probs):
    return tf.nn.dropout(x, probs)
