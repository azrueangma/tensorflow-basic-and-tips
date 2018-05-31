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

BOARD_PATH = "./board/lab08-1_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

image_width = np.size(x_train, 1)
image_height = np.size(x_train, 2)

print("\nThe number of train samples : ", ntrain)
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
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='h')
    return h


def max_pooling(tensor_op, ksize_w, ksize_h, stride_w, stride_h, name='MaxPool'):
    with tf.variable_scope(name):
        p = tf.nn.max_pool(tensor_op, ksize=[1, ksize_w, ksize_h, 1], strides=[1, stride_w, stride_h, 1],
                           padding='SAME', name='p')
    return p


with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')

h1 = conv2d(X, 1, 1, [5, 5, 1, 32], name='Conv1')
p1 = max_pooling(h1, 2, 2, 2, 2, name='MaxPool1')
h2 = conv2d(p1, 1, 1, [5, 5, 32, 64], name='Conv2')
p2 = max_pooling(h2, 2, 2, 2, 2, name='MaxPool2')
h3 = conv2d(p2, 1, 1, [5, 5, 64, 128], name='Conv3')
p3 = max_pooling(h3, 2, 2, 2, 2, name='MaxPool3')

flat_op = tf.reshape(p3, [-1, 4 * 4 * 128], name = 'flat_op')
f1 = relu_layer(flat_op, 1024, name='FC_Relu')
logits = linear(f1, NCLASS, name='FC_Linear')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot),
                                name='loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss + tf.reduce_sum(weight_decay_loss)
    optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

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
    for epoch in range(TOTAL_EPOCH):
        loss_per_epoch = 0
        acc_per_epoch = 0

        np.random.seed(epoch)
        mask = np.random.permutation(len(x_train))

        epoch_start_time = time.perf_counter()
        for step in range(total_step):
            s = BATCH_SIZE * step
            t = BATCH_SIZE * (step + 1)
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :]})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step * BATCH_SIZE
        acc_per_epoch /= total_step * BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation})
        epoch_valid_acc = va / len(x_validation)
        epoch_valid_loss = vl / len(x_validation)

        s = sess.run(merged, feed_dict={avg_train_loss: loss_per_epoch, avg_train_acc: acc_per_epoch,
                                        avg_validation_loss: epoch_valid_loss, avg_validation_acc: epoch_valid_acc})
        writer.add_summary(s, global_step=epoch)

        if (epoch + 1) % 1 == 0:
            print("Epoch [{:2d}/{:2d}], train loss = {:.6f}, train accuracy = {:.2%}, "
                  "valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)"
                  .format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc,
                          epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

    ta = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
    print("Test Accraucy : {:.2%}".format(ta / ntest))

'''
using GTX 1080 ti
Epoch [ 1/30], train loss = 79.122849, train accuracy = 92.13%, valid loss = 3.010287, valid accuracy = 95.17%, duration = 7.923739(s)
Epoch [ 2/30], train loss = 1.363020, train accuracy = 96.53%, valid loss = 0.891806, valid accuracy = 96.32%, duration = 6.571883(s)
Epoch [ 3/30], train loss = 0.637619, train accuracy = 97.05%, valid loss = 0.613560, valid accuracy = 97.00%, duration = 6.037434(s)
Epoch [ 4/30], train loss = 0.389393, train accuracy = 97.70%, valid loss = 0.703584, valid accuracy = 95.85%, duration = 6.154356(s)
Epoch [ 5/30], train loss = 0.327045, train accuracy = 97.62%, valid loss = 0.668021, valid accuracy = 96.57%, duration = 6.041801(s)
Epoch [ 6/30], train loss = 0.314757, train accuracy = 97.74%, valid loss = 0.514855, valid accuracy = 97.00%, duration = 6.009547(s)
Epoch [ 7/30], train loss = 0.209898, train accuracy = 97.97%, valid loss = 0.589057, valid accuracy = 96.10%, duration = 6.022144(s)
Epoch [ 8/30], train loss = 0.206383, train accuracy = 98.09%, valid loss = 0.411355, valid accuracy = 96.33%, duration = 6.000596(s)
Epoch [ 9/30], train loss = 0.187513, train accuracy = 98.26%, valid loss = 0.303086, valid accuracy = 97.30%, duration = 6.021909(s)
Epoch [10/30], train loss = 0.128003, train accuracy = 98.60%, valid loss = 0.358258, valid accuracy = 97.68%, duration = 6.024030(s)
Epoch [11/30], train loss = 0.169659, train accuracy = 98.50%, valid loss = 0.351423, valid accuracy = 97.48%, duration = 6.012955(s)
Epoch [12/30], train loss = 0.129909, train accuracy = 98.74%, valid loss = 0.255839, valid accuracy = 98.05%, duration = 6.031613(s)
Epoch [13/30], train loss = 0.128919, train accuracy = 98.70%, valid loss = 0.310243, valid accuracy = 97.63%, duration = 5.981495(s)
Epoch [14/30], train loss = 0.107500, train accuracy = 98.87%, valid loss = 0.268815, valid accuracy = 98.03%, duration = 6.033234(s)
Epoch [15/30], train loss = 0.107283, train accuracy = 98.98%, valid loss = 0.348185, valid accuracy = 97.57%, duration = 6.024305(s)
Epoch [16/30], train loss = 0.111463, train accuracy = 98.97%, valid loss = 0.256906, valid accuracy = 98.07%, duration = 5.997611(s)
Epoch [17/30], train loss = 0.093527, train accuracy = 99.10%, valid loss = 0.343424, valid accuracy = 97.62%, duration = 6.001765(s)
Epoch [18/30], train loss = 0.106897, train accuracy = 99.05%, valid loss = 0.230483, valid accuracy = 98.22%, duration = 6.057292(s)
Epoch [19/30], train loss = 0.093087, train accuracy = 99.11%, valid loss = 0.300954, valid accuracy = 98.28%, duration = 6.021824(s)
Epoch [20/30], train loss = 0.079430, train accuracy = 99.26%, valid loss = 0.398739, valid accuracy = 98.02%, duration = 5.993112(s)
Epoch [21/30], train loss = 0.089002, train accuracy = 99.13%, valid loss = 0.266285, valid accuracy = 98.77%, duration = 6.017154(s)
Epoch [22/30], train loss = 0.085100, train accuracy = 99.30%, valid loss = 0.352144, valid accuracy = 98.27%, duration = 6.062466(s)
Epoch [23/30], train loss = 0.094507, train accuracy = 99.25%, valid loss = 0.208958, valid accuracy = 98.23%, duration = 5.993033(s)
Epoch [24/30], train loss = 0.065771, train accuracy = 99.29%, valid loss = 0.228500, valid accuracy = 98.32%, duration = 6.016311(s)
Epoch [25/30], train loss = 0.085776, train accuracy = 99.25%, valid loss = 0.426815, valid accuracy = 98.17%, duration = 5.983092(s)
Epoch [26/30], train loss = 0.085173, train accuracy = 99.22%, valid loss = 0.383753, valid accuracy = 98.03%, duration = 5.993463(s)
Epoch [27/30], train loss = 0.078761, train accuracy = 99.28%, valid loss = 0.470332, valid accuracy = 98.25%, duration = 6.027120(s)
Epoch [28/30], train loss = 0.073437, train accuracy = 99.26%, valid loss = 0.321497, valid accuracy = 98.02%, duration = 6.054472(s)
Epoch [29/30], train loss = 0.061855, train accuracy = 99.35%, valid loss = 0.446503, valid accuracy = 98.45%, duration = 5.991873(s)
Epoch [30/30], train loss = 0.070156, train accuracy = 99.39%, valid loss = 0.271249, valid accuracy = 98.67%, duration = 5.941667(s)
Duration for train : 185.128159(s)
<<< Train Finished >>>
Test Accraucy : 98.59%
'''
