#tensorboard_summary with deep regression
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import load_data
import shutil
import os

NPOINTS = 1000
TOTAL_EPOCH = 100
BOARD_PATH = "./board/lab03-5_board"
if os.path.exists(BOARD_PATH):
    shutil.rmtree(BOARD_PATH)

def linear(x, output_dim, with_W, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [x.get_shape()[-1], output_dim], dtype = tf.float32, 
                            initializer= tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        if with_W:
            return h, W
        else:
            return h

def sigmoid_linear(x, output_dim, with_W, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name = 'W', shape = [x.get_shape()[-1], output_dim], dtype = tf.float32, 
                            initializer= tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        if with_W:
            return h, W
        else:
            return h

dataX, dataY = load_data.generate_data_for_linear_regression(NPOINTS)

g = tf.Graph()
with g.as_default(), tf.variable_scope("Model1"):
    tf.set_random_seed(0)
    with tf.variable_scope("Inputs"):
        g1_X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
        g1_Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

    g1_h1 = sigmoid_linear(g1_X, 5, False, 'FC_Layer1')
    g1_h2 = sigmoid_linear(g1_h1, 10, False, 'FC_Layer2')
    g1_h3 = linear(g1_h2, 1, False, 'FC_Layer3')
    g1_hypothesis = tf.identity(g1_h3, name = 'hypothesis')

    with tf.variable_scope('Optimization'):
        g1_loss = tf.reduce_mean(tf.square(g1_Y-g1_hypothesis), name = 'loss')
        g1_optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(g1_loss)

    init_op_1 = tf.global_variables_initializer()

with g.as_default(), tf.variable_scope("Model2"):
    tf.set_random_seed(1)
    with tf.variable_scope("Inputs"):
        g2_X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
        g2_Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

    g2_h1 = sigmoid_linear(g2_X, 5, False, 'FC_Layer1')
    g2_h2 = sigmoid_linear(g2_h1, 10, False, 'FC_Layer2')
    g2_h3 = linear(g2_h2, 1, False, 'FC_Layer3')
    g2_hypothesis = tf.identity(g2_h3, name = 'hypothesis')

    with tf.variable_scope('Optimization'):
        g2_loss = tf.reduce_mean(tf.square(g2_Y-g2_hypothesis), name = 'loss')
        g2_optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(g2_loss)

with g.as_default(), tf.variable_scope("Model3"):
    tf.set_random_seed(2)
    with tf.variable_scope("Inputs"):
        g3_X = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'X')
        g3_Y = tf.placeholder(shape = [None, 1], dtype = tf.float32, name = 'Y')

    g3_h1 = sigmoid_linear(g3_X, 5, False, 'FC_Layer1')
    g3_h2 = sigmoid_linear(g3_h1, 10, False, 'FC_Layer2')
    g3_h3 = linear(g3_h2, 1, False, 'FC_Layer3')
    g3_hypothesis = tf.identity(g3_h3, name = 'hypothesis')

    with tf.variable_scope('Optimization'):
        g3_loss = tf.reduce_mean(tf.square(g3_Y-g3_hypothesis), name = 'loss')
        g3_optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(g3_loss)

with g.as_default():
    with tf.variable_scope("Summary"):
        writer = tf.summary.FileWriter(BOARD_PATH)
    with tf.variable_scope("Init_op"):
        init_op = tf.global_variables_initializer()

with tf.Session(graph = g) as sess:
    writer.add_graph(sess.graph)
    sess.run(init_op)
    for epoch in range(TOTAL_EPOCH):
        l1, _ = sess.run([g1_loss,g1_optim], feed_dict={g1_X: dataX, g1_Y: dataY})
        l2, _ = sess.run([g2_loss, g2_optim], feed_dict={g2_X: dataX, g2_Y: dataY})
        l3, _ = sess.run([g3_loss, g3_optim], feed_dict={g3_X: dataX, g3_Y: dataY})
        if (epoch+1) %10 == 0:
            print("Epoch [{:3d}/{:3d}], loss1 = {:.6f}, loss2 = {:.6f}, loss3 = {:.6f}".format(epoch + 1, TOTAL_EPOCH, l1, l2, l3))

'''
Epoch [ 10/100], loss1 = 0.011773, loss2 = 1.440321, loss3 = 4.194822
Epoch [ 20/100], loss1 = 0.010828, loss2 = 1.161871, loss3 = 3.502508
Epoch [ 30/100], loss1 = 0.010030, loss2 = 0.938139, loss3 = 2.928084
Epoch [ 40/100], loss1 = 0.009357, loss2 = 0.758266, loss3 = 2.450624
Epoch [ 50/100], loss1 = 0.008789, loss2 = 0.613586, loss3 = 2.053164
Epoch [ 60/100], loss1 = 0.008309, loss2 = 0.497166, loss3 = 1.721883
Epoch [ 70/100], loss1 = 0.007905, loss2 = 0.403455, loss3 = 1.445469
Epoch [ 80/100], loss1 = 0.007563, loss2 = 0.328005, loss3 = 1.214631
Epoch [ 90/100], loss1 = 0.007275, loss2 = 0.267243, loss3 = 1.021710
Epoch [100/100], loss1 = 0.007031, loss2 = 0.218301, loss3 = 0.860377
'''
