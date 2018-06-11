#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import shutil

#constant
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('total_epoch', 100, "the number of train epochs")
flags.DEFINE_integer('cpu_device', 0, 'the number of cpu device ')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate for optimization')
flags.DEFINE_string('board_path', "./board/lab02-3_board", "board directory")
flags.DEFINE_string('model_path', "./model/", "model directory")

if not os.path.exists(FLAGS.model_path):
    os.mkdir(FLAGS.model_path)

if os.path.exists(FLAGS.board_path):
    shutil.rmtree(FLAGS.board_path)

#load data
x_train = np.array([[3.5], [4.2], [3.8], [5.0], [5.1], [0.2], [1.3], [2.4], [1.5], [0.5]])
y_train = np.array([[3.3], [4.4], [4.0], [4.8], [5.6], [0.1], [1.2], [2.4], [1.7], [0.6]])


def linear(input_op, output_dim, name):
    with tf.variable_scope(name):
        input_shape = input_op.get_shape().as_list()
        W = tf.get_variable(name='W', shape=[input_shape[-1],output_dim], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], initializer=tf.zeros_initializer())
        output = tf.nn.bias_add(tf.multiply(input_op, W), b, name='output')
    return W, b, output


class Model(object):
    def __init__(self, g, seed):
        self.graph = g
        self.seed = seed
        self._build_model()


    def _build_model(self):
        with self.graph.as_default(), self.graph.device('/cpu:{}'.format(FLAGS.cpu_device)):
            tf.set_random_seed(self.seed)
            with tf.variable_scope("Inputs"):
                #create placeholder for input
                self.X = tf.placeholder(shape=[None,1], dtype=tf.float32, name='X')
                self.Y = tf.placeholder(shape=[None,1], dtype=tf.float32, name='Y')
                self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

            W, b, self.output = linear(input_op=self.X, output_dim=1, name="Linear")
            self.loss = self.get_loss("Loss")
            self.optim = self.get_optim(self.learning_rate, "Optimization")
            self.writer = self.get_writer(FLAGS.board_path)
            self.saver = self.get_saver()

            with tf.variable_scope("Summary"):
                W_hist = tf.summary.histogram('W_hist', W)
                b_hist = tf.summary.histogram('b_hist', b)
                loss_scalar = tf.summary.scalar('loss_scalar', self.loss)
                self.merged = tf.summary.merge_all()


    def fit(self, x_train, y_train, learning_rate):
        with tf.Session(graph=self.graph) as sess:
            self.writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())

            for epoch in range(FLAGS.total_epoch):
                m, l, _ = sess.run([self.merged, self.loss, self.optim],
                                   feed_dict={self.X:x_train,
                                              self.Y:y_train,
                                              self.learning_rate:learning_rate})

                self.writer.add_summary(m, global_step=epoch)
                if (epoch+1) %10 == 0:
                    print("Epoch [{:3d}/{:3d}], loss = {:.6f}".format(epoch+1, FLAGS.total_epoch, l))
                    # save model per epoch
                    self.saver.save(sess, FLAGS.model_path + "my_model_{}/model".format(epoch+1))


    def evaluate(self, x_test, y_test):
        with tf.Session(graph=self.graph) as sess:
            return self.sess.run(self.loss, feed_dict={self.X:x_test, self.Y:y_test})


    def prediction(self, x_test):
        with tf.Session(graph=self.graph) as sess:
            return self.sess.run(self.output, feed_dict={self.X:x_test})


    def get_loss(self, name):
        with tf.variable_scope(name):
            loss = tf.reduce_mean(tf.square(self.Y - self.output), name='loss')
        return loss


    def get_optim(self, learning_rate, name):
        with tf.variable_scope(name):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            optim = optimizer.minimize(self.loss)
        return optim


    def get_writer(self, path):
        return tf.summary.FileWriter(path)


    def get_saver(self):
        return tf.train.Saver()


g = tf.Graph()
seed = 0
m = Model(g, seed)
m.fit(x_train, y_train, FLAGS.learning_rate)