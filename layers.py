import helpers
import tensorflow as tf

class conv:

    def __init__(self, filtersize, nooffilters, channels, stride):
        self.weights = helpers.weight_variable([filtersize, filtersize, channels, nooffilters])
        self.biases = helpers.bias_variable([nooffilters])
        self.strides = 4 * [stride]

    def result(self, data):
        return helpers.relu(helpers.convolve(data, self.weights, self.strides) + self.biases)


class maxPool:

    def __init__(self, size):
        self.size = size

    def result(self, data):
        return helpers.max_pool(data, self.size)


class dense:

    def __init__(self, inputsize, noofnuerons, dropout_rate, reshape_needed=False):
        self.inputsize = inputsize
        self.weights = helpers.weight_variable([inputsize, noofnuerons])
        self.biases = helpers.bias_variable([noofnuerons])
        self.reshape = reshape_needed
        self.dropout = dropout_rate
        #self.dropoutTensor = tf.constant([1.0-dropout_rate] * noofnuerons)

    def result(self, data):
        if self.reshape:
            data = tf.reshape(data, [-1, self.inputsize])
        computed = helpers.relu(tf.matmul(data, self.weights) + self.biases)
        #if self.dropout > 0:
        computed = helpers.dropout(computed, self.dropout)
        return computed

class softmax:

    def __init__(self, inputsize, noofclasses):
        self.noOfClasses = noofclasses
        self.weights = helpers.weight_variable([inputsize, noofclasses])
        self.biases = helpers.bias_variable([noofclasses])

    def result(self, data):
        return tf.matmul(data, self.weights) + self.biases