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
flags.DEFINE_string('board_path', "./board/lab01-7_board", "board directory")

if os.path.exists(FLAGS.board_path):
    shutil.rmtree(FLAGS.board_path)

#load data
x_train = np.array([[3.5], [4.2], [3.8], [5.0], [5.1], [0.2], [1.3], [2.4], [1.5], [0.5]])
y_train = np.array([[3.3], [4.4], [4.0], [4.8], [5.6], [0.1], [1.2], [2.4], [1.7], [0.6]])

g = tf.Graph()


with g.as_default(), tf.device('/cpu:{}'.format(FLAGS.cpu_device)):
    tf.set_random_seed(0)
    with tf.variable_scope("Inputs"):
        #create placeholder for input
        X = tf.placeholder(shape=[None,1], dtype=tf.float32, name='X')
        Y = tf.placeholder(shape=[None,1], dtype=tf.float32, name='Y')

    with tf.variable_scope("Weight_and_Bias"):
        #weigth and bias
        W = tf.Variable(tf.truncated_normal(shape=[1]), name='W')
        b = tf.Variable(tf.zeros([1]), name='b')

    with tf.variable_scope("Output"):
        output = tf.nn.bias_add(tf.multiply(X, W), b, name='output')

    with tf.variable_scope("Loss"):
        loss = tf.reduce_mean(tf.square(Y-output), name='loss')

    with tf.variable_scope("Optimization"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        optim = optimizer.minimize(loss)

    with tf.variable_scope("Summary"):
        W_hist = tf.summary.histogram('W_hist', W)
        b_hist = tf.summary.histogram('b_hist', b)
        loss_scalar = tf.summary.scalar('loss_scalar', loss)
        merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter(FLAGS.board_path)


    init_op = tf.global_variables_initializer()
    with tf.Session(graph=g) as sess:
        writer.add_graph(sess.graph)
        sess.run(init_op)

        for epoch in range(FLAGS.total_epoch):
            m, l, _ = sess.run([merged, loss, optim], feed_dict={X:x_train, Y:y_train})
            writer.add_summary(m, global_step=epoch)
            if (epoch+1) %10 == 0:
                print("Epoch [{:3d}/{:3d}], loss = {:.6f}".format(epoch + 1, FLAGS.total_epoch, l))