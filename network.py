import layers
import tensorflow as tf
from datahelper import *
import logging
import time

class network:

    reportFrequency = 50

    def __init__(self):
		self.global_step = tf.Variable(0, trainable=False)
		self.starter_learning_rate = 0.004
		self.end_learning_rate = 0.00001
		self.decay_steps = 1000
		self.learning_rate = tf.train.polynomial_decay(self.starter_learning_rate, self.global_step, self.decay_steps, self.end_learning_rate, power=0.5)
		self.dropoutRate = tf.placeholder(tf.float32)
		self.session = None
		self.dataHelper = None
		self.layers = []
		self.layers.append(layers.conv(7, 96, 50, 1))
		self.layers.append(layers.maxPool(2))
		self.layers.append(layers.conv(5, 192, 96, 2))
		self.layers.append(layers.maxPool(2))
		self.layers.append(layers.conv(3, 512, 192, 1))
		self.layers.append(layers.maxPool(2))
		self.layers.append(layers.conv(2, 4096, 512, 1))
		self.layers.append(layers.dense(3*4096, 4096, dropout_rate=self.dropoutRate, reshape_needed=True))
		self.layers.append(layers.dense(4096, 2048, dropout_rate=self.dropoutRate))
		self.layers.append(layers.dense(2048, 2))
		self.input = tf.placeholder(tf.float32, [None, 60, 40, 50])
		self.normalizedInput = tf.truediv(tf.sub(self.input, tf.constant(128.)), tf.constant(128.), name = "NormalizedInput")
		self.results = []
		self.results.append(self.layers[0].result(self.normalizedInput))
		for i in xrange(1, len(self.layers)):
			try:
				self.results.append(self.layers[i].result(self.results[i-1]))
			except:
				print(i)
				raise
		self.finalResult = self.results[len(self.results) - 1]
		self.labels = tf.placeholder(tf.float32, [None, 2])
		self.crossEntropy = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits(self.finalResult, self.labels))
		self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.crossEntropy, global_step=self.global_step)
		self.correct_prediction = tf.equal(tf.argmax(self.finalResult, 1), tf.argmax(self.labels, 1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
		self.saver = None

    def test(self, path):
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        self.dataHelper = datahelper(path)
        data = self.dataHelper.getsingledata()
        print(data.data.shape)
        print(len(data.labels))
        res = self.results[len(self.results)-1].eval(feed_dict={
            self.input: data.data, self.labels: data.labels, self.dropoutRate: 0.5})
        print("finshed, output %g", res)

    def train(self, path, epochs, batchsize):
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        self.dataHelper = datahelper(path)
        print("started")
        logging.basicConfig(filename=time.ctime() + '.log', level=logging.DEBUG)
        logging.info("Started at" + time.ctime())
        for i in xrange(epochs):
            newbatch = self.dataHelper.getnextbatch(batchsize)
            if i % network.reportFrequency == 0 and i > 0:
                results = self.session.run([self.accuracy, self.crossEntropy],feed_dict={self.input: newbatch.data, self.labels: newbatch.labels, self.dropoutRate: 1})
                print(results)
            self.train_step.run(feed_dict={self.input: newbatch.data, self.labels: newbatch.labels, self.dropoutRate: 0.6})
        logging.info("Finished training at" + time.ctime())
        testdata = self.dataHelper.gettestdata()
        finAcc = self.accuracy.eval(feed_dict={
            self.input: testdata.data, self.labels: testdata.labels, self.dropoutRate: 1})
        print(finAcc)
        tf.train.write_graph(self.session.graph_def, '/home/samba/lubonp/inzynierka', 'graf'+time.ctime())
        saver = tf.train.Saver(tf.all_variables())
        saver.save(self.session,"checkpoint.data")
        logging.info("final accuracy %g" % finAcc)
        logging.info("Finished run at" + time.ctime())




