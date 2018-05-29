# -*- coding: utf-8 -*-
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import shutil
import time

import load_data

x_train, x_validation, x_test, y_train, y_validation, y_test = load_data.load_mnist('./data/mnist/', seed=0,
                                                                                    as_image=True, scaling=True)

BOARD_PATH = "./board/lab08-4_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30
INIT_LEARNING_RATE = 0.001

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

image_width = np.size(x_train, 1)
image_height = np.size(x_train, 2)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)


def l1_loss(tensor_op, name='l1_loss'):
    output = tf.reduce_sum(tf.abs(tensor_op), name=name)
    return output


def l2_loss(tensor_op, name='l2_loss'):
    output = tf.reduce_sum(tf.square(tensor_op), name=name) / 2
    return output


def linear(tensor_op, output_dim, weight_decay=None, regularizer=None, with_W=False, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W) * weight_decay
            elif regularizer == 'l2':
                wd = l2_loss(W) * weight_decay
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


def relu_layer(tensor_op, output_dim, weight_decay=None, regularizer=None,
               keep_prob=1.0, is_training=False, with_W=False, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='pre_op')
        bn = tf.contrib.layers.batch_norm(pre_activation,
                                          is_training=is_training,
                                          updates_collections=None)
        h = tf.nn.relu(bn, name='relu_op')
        dr = tf.nn.dropout(h, keep_prob=keep_prob, name='dropout_op')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W) * weight_decay
            elif regularizer == 'l2':
                wd = l2_loss(W) * weight_decay
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return dr, W
        else:
            return dr


def conv2d(tensor_op, stride_w, stride_h, shape, name='Conv'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=shape, dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name='b', shape=shape[-1], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(tensor_op, W, strides=[1, stride_w, stride_h, 1], padding='SAME', name='conv')
    return conv


def max_pooling(tensor_op, ksize_w, ksize_h, stride_w, stride_h, name='MaxPool'):
    with tf.variable_scope(name):
        p = tf.nn.max_pool(tensor_op, ksize=[1, ksize_w, ksize_h, 1], strides=[1, stride_w, stride_h, 1],
                           padding='SAME', name='p')
    return p


def bn_layer(x, is_training, name):
    with tf.variable_scope(name):
        bn = tf.contrib.layers.batch_norm(x, updates_collections=None, scale=True, is_training=is_training)
        post_activation = tf.nn.relu(bn, name='relu')
    return post_activation


with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    is_training = tf.placeholder(tf.bool, name='is_training')

h1 = conv2d(X, 1, 1, [5, 5, 1, 32], name='Conv1')
b1 = bn_layer(h1, is_training, name='bn1')
p1 = max_pooling(b1, 2, 2, 2, 2, name='MaxPool1')
h2 = conv2d(p1, 1, 1, [5, 5, 32, 64], name='Conv2')
b2 = bn_layer(h2, is_training, name='bn2')
p2 = max_pooling(b2, 2, 2, 2, 2, name='MaxPool2')
h3 = conv2d(p2, 1, 1, [5, 5, 64, 128], name='Conv3')
b3 = bn_layer(h3, is_training, name='bn3')
p3 = max_pooling(b3, 2, 2, 2, 2, name='MaxPool3')

flat_op = tf.reshape(p3, [-1, 4 * 4 * 128], name = 'flat_op')
f1 = relu_layer(flat_op, 1024, name='FC_Relu')
logits = linear(f1, NCLASS, name='FC_Linear')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot),
                                name='loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss + tf.reduce_sum(weight_decay_loss)
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.variable_scope("Prediction"):
    predict = tf.argmax(hypothesis, axis=1)

with tf.variable_scope("Accuracy"):
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis=1)), tf.float32))

with tf.variable_scope("Summary"):
    avg_train_loss = tf.placeholder(tf.float32)
    loss_train_avg = tf.summary.scalar('avg_train_loss', avg_train_loss)
    avg_train_acc = tf.placeholder(tf.float32)
    acc_train_avg = tf.summary.scalar('avg_train_acc', avg_train_acc)
    avg_validation_loss = tf.placeholder(tf.float32)
    loss_validation_avg = tf.summary.scalar('avg_validation_loss', avg_validation_loss)
    avg_validation_acc = tf.placeholder(tf.float32)
    acc_validation_avg = tf.summary.scalar('avg_validation_acc', avg_validation_acc)
    merged = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
total_step = int(ntrain / BATCH_SIZE)
print("Total step : ", total_step)
with tf.Session() as sess:
    if os.path.exists(BOARD_PATH):
        shutil.rmtree(BOARD_PATH)
    writer = tf.summary.FileWriter(BOARD_PATH)
    writer.add_graph(sess.graph)

    sess.run(init_op)

    train_start_time = time.perf_counter()
    u = INIT_LEARNING_RATE
    for epoch in range(TOTAL_EPOCH):
        loss_per_epoch = 0
        acc_per_epoch = 0

        np.random.seed(epoch)
        mask = np.random.permutation(len(x_train))

        epoch_start_time = time.perf_counter()
        for step in range(total_step):
            s = BATCH_SIZE * step
            t = BATCH_SIZE * (step + 1)
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :], is_training:True, learning_rate:u})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step * BATCH_SIZE
        acc_per_epoch /= total_step * BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation, is_training:False})
        epoch_valid_acc = va / len(x_validation)
        epoch_valid_loss = vl / len(x_validation)

        s = sess.run(merged, feed_dict={avg_train_loss: loss_per_epoch, avg_train_acc: acc_per_epoch,
                                        avg_validation_loss: epoch_valid_loss, avg_validation_acc: epoch_valid_acc})
        writer.add_summary(s, global_step=epoch)

        u = u*0.95
        if (epoch + 1) % 1 == 0:
            print("Epoch [{:2d}/{:2d}], train loss = {:.6f}, train accuracy = {:.2%}, "
                  "valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)"
                  .format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc,
                          epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

    ta = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, is_training:False})
    print("Test Accraucy : {:.2%}".format(ta / ntest))

'''
GTX 1080Ti
Epoch [ 1/30], train loss = 0.205854, train accuracy = 94.48%, valid loss = 0.077242, valid accuracy = 97.70%, duration = 8.236485(s)
Epoch [ 2/30], train loss = 0.062815, train accuracy = 98.09%, valid loss = 0.055174, valid accuracy = 98.22%, duration = 7.497140(s)
Epoch [ 3/30], train loss = 0.044701, train accuracy = 98.57%, valid loss = 0.054312, valid accuracy = 98.25%, duration = 7.342453(s)
Epoch [ 4/30], train loss = 0.030428, train accuracy = 99.04%, valid loss = 0.074415, valid accuracy = 97.88%, duration = 7.330510(s)
Epoch [ 5/30], train loss = 0.023784, train accuracy = 99.23%, valid loss = 0.069428, valid accuracy = 98.18%, duration = 7.186586(s)
Epoch [ 6/30], train loss = 0.018400, train accuracy = 99.43%, valid loss = 0.066427, valid accuracy = 97.97%, duration = 7.255452(s)
Epoch [ 7/30], train loss = 0.014048, train accuracy = 99.55%, valid loss = 0.082245, valid accuracy = 97.97%, duration = 7.249749(s)
Epoch [ 8/30], train loss = 0.011823, train accuracy = 99.62%, valid loss = 0.060935, valid accuracy = 98.65%, duration = 7.217030(s)
Epoch [ 9/30], train loss = 0.010729, train accuracy = 99.69%, valid loss = 0.047962, valid accuracy = 98.73%, duration = 7.192811(s)
Epoch [10/30], train loss = 0.009774, train accuracy = 99.67%, valid loss = 0.047439, valid accuracy = 98.97%, duration = 7.170126(s)
Epoch [11/30], train loss = 0.008257, train accuracy = 99.74%, valid loss = 0.061849, valid accuracy = 98.33%, duration = 7.239045(s)
Epoch [12/30], train loss = 0.007372, train accuracy = 99.76%, valid loss = 0.068596, valid accuracy = 98.68%, duration = 7.183827(s)
Epoch [13/30], train loss = 0.006427, train accuracy = 99.80%, valid loss = 0.061122, valid accuracy = 98.55%, duration = 7.168444(s)
Epoch [14/30], train loss = 0.005981, train accuracy = 99.79%, valid loss = 0.046466, valid accuracy = 99.08%, duration = 7.184030(s)
Epoch [15/30], train loss = 0.006851, train accuracy = 99.78%, valid loss = 0.056608, valid accuracy = 98.78%, duration = 7.185708(s)
Epoch [16/30], train loss = 0.005956, train accuracy = 99.83%, valid loss = 0.065761, valid accuracy = 98.78%, duration = 7.232588(s)
Epoch [17/30], train loss = 0.005458, train accuracy = 99.83%, valid loss = 0.059319, valid accuracy = 98.85%, duration = 7.207815(s)
Epoch [18/30], train loss = 0.006398, train accuracy = 99.82%, valid loss = 0.072243, valid accuracy = 98.82%, duration = 7.261676(s)
Epoch [19/30], train loss = 0.006442, train accuracy = 99.81%, valid loss = 0.057623, valid accuracy = 99.10%, duration = 7.171810(s)
Epoch [20/30], train loss = 0.006468, train accuracy = 99.84%, valid loss = 0.113074, valid accuracy = 98.25%, duration = 7.179477(s)
Epoch [21/30], train loss = 0.006453, train accuracy = 99.83%, valid loss = 0.074354, valid accuracy = 98.67%, duration = 7.168584(s)
Epoch [22/30], train loss = 0.004226, train accuracy = 99.89%, valid loss = 0.047570, valid accuracy = 99.22%, duration = 7.100572(s)
Epoch [23/30], train loss = 0.005580, train accuracy = 99.87%, valid loss = 0.078931, valid accuracy = 98.75%, duration = 7.219139(s)
Epoch [24/30], train loss = 0.004368, train accuracy = 99.86%, valid loss = 0.087982, valid accuracy = 98.68%, duration = 7.204615(s)
Epoch [25/30], train loss = 0.004302, train accuracy = 99.87%, valid loss = 0.061161, valid accuracy = 99.05%, duration = 7.176215(s)
Epoch [26/30], train loss = 0.005457, train accuracy = 99.87%, valid loss = 0.076442, valid accuracy = 98.93%, duration = 7.113804(s)
Epoch [27/30], train loss = 0.004393, train accuracy = 99.90%, valid loss = 0.063605, valid accuracy = 98.95%, duration = 7.172110(s)
Epoch [28/30], train loss = 0.002971, train accuracy = 99.91%, valid loss = 0.063514, valid accuracy = 99.03%, duration = 7.159093(s)
Epoch [29/30], train loss = 0.005452, train accuracy = 99.88%, valid loss = 0.065635, valid accuracy = 99.05%, duration = 7.183179(s)
Epoch [30/30], train loss = 0.003111, train accuracy = 99.92%, valid loss = 0.052762, valid accuracy = 99.22%, duration = 7.187706(s)
Duration for train : 219.883155(s)
<<< Train Finished >>>
Test Accraucy : 99.11%
'''
