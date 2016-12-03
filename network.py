import layers
import tensorflow as tf
from datahelper import *

class network:

    reportFrequency = 100

    def __init__(self):
        self.dropoutRate = tf.placeholder(tf.float32)
        self.session = None
        self.dataHelper = None
        self.layers = []
        self.layers.append(layers.conv(7, 96, 50, 1))
        self.layers.append(layers.maxPool(2))
        self.layers.append(layers.conv(5, 192, 1, 2))
        self.layers.append(layers.maxPool(2))
        self.layers.append(layers.conv(3, 512, 1, 1))
        self.layers.append(layers.maxPool(2))
        self.layers.append(layers.dense(512, 4096, reshape_needed=True, dropout_rate=self.dropoutRate))
        self.layers.append(layers.dense(4096, 2048, dropout_rate=self.dropoutRate))
        self.layers.append(layers.dense(4096, 2048, dropout_rate=self.dropoutRate))
        self.layers.append(layers.softmax(4096, 2))
        self.input = tf.placeholder(tf.float32, [None, 60, 60, 50])
        self.results = []
        self.results.append(self.layers[0].result(self.input))
        for i in xrange(1, len(self.layers)):
            self.results.append(self.layers[i].result(self.results[i-1]))
        self.finalResult = self.results[len(self.results) - 1]
        self.labels = tf.placeholder(tf.float32, [None, 2])
        self.crossEntropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.finalResult, self.labels))
        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.crossEntropy)
        self.correct_prediction = tf.equal(tf.argmax(self.finalResult, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self, path, epochs, batchsize):
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.dataHelper = datahelper(path)
        for i in xrange(epochs):
            batch = self.dataHelper.getnextbatch(batchsize)
            if i % network.reportFrequency == 0 and i > 0:
                train_accuracy = self.accuracy.eval(feed_dict={
                    self.input: batch.data, self.labels: batch.labels, self.dropoutRate: 0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            self.train_step.run(feed_dict={self.input: batch[0], self.labels: batch[1]})
        testdata = self.dataHelper.gettestdata()
        print("final accuracy %g" % self.accuracy.eval(feed_dict={
            self.input: testdata.data, self.labels: testdata.data, self.dropoutRate: 0}))




