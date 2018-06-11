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
flags.DEFINE_string('board_path', "./board/lab02-2_board", "board directory")
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


def get_loss(output, Y_op, name):
    with tf.variable_scope(name):
        loss = tf.reduce_mean(tf.square(Y_op - output), name='loss')
    return loss


def get_optim(loss, learning_rate, name):
    with tf.variable_scope(name):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        optim = optimizer.minimize(loss)
    return optim


def get_writer(path):
    return tf.summary.FileWriter(path)


def get_saver():
    return tf.train.Saver()


def build_model(g):
    with g.as_default(), g.device('/cpu:{}'.format(FLAGS.cpu_device)):
        tf.set_random_seed(0)
        with tf.variable_scope("Inputs"):
            #create placeholder for input
            X = tf.placeholder(shape=[None,1], dtype=tf.float32, name='X')
            Y = tf.placeholder(shape=[None,1], dtype=tf.float32, name='Y')

        W, b, output = linear(input_op=X, output_dim=1, name="Linear")
        loss = get_loss(output, Y, "Loss")
        optim = get_optim(loss, 0.001, "Optimization")
        writer = get_writer(FLAGS.board_path)
        saver = get_saver()

        with tf.variable_scope("Summary"):
            W_hist = tf.summary.histogram('W_hist', W)
            b_hist = tf.summary.histogram('b_hist', b)
            loss_scalar = tf.summary.scalar('loss_scalar', loss)
            merged = tf.summary.merge_all()

    return (X, Y, W, b, loss, optim, writer, saver, merged)

def fit(g, X, Y, W, b, loss, optim, writer, saver, merged, x_train, y_train):
    with tf.Session(graph=g) as sess:
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch in range(FLAGS.total_epoch):
            m, l, _ = sess.run([merged, loss, optim], feed_dict={X:x_train, Y:y_train})
            writer.add_summary(m, global_step=epoch)
            if (epoch+1) %10 == 0:
                print("Epoch [{:3d}/{:3d}], loss = {:.6f}".format(epoch+1, FLAGS.total_epoch, l))
                # save model per epoch
                saver.save(sess, FLAGS.model_path + "my_model_{}/model".format(epoch+1))

g = tf.Graph()
X, Y, W, b, loss, optim, writer, saver, merged = build_model(g)
run(g, X, Y, W, b, loss, optim, writer, saver, merged, x_train, y_train)