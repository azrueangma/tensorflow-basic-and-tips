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

BOARD_PATH = "./board/lab08-3_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30
ALPHA = 0
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


def linear(tensor_op, output_dim, weight_decay=False, regularizer=None, with_W=False, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)
            elif regularizer == 'l2':
                wd = l2_loss(W)
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


def relu_layer(tensor_op, output_dim, weight_decay=False, regularizer=None,
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
                wd = l1_loss(W)
            elif regularizer == 'l2':
                wd = l2_loss(W)
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


def dropout_layer(tensor_op, keep_prob, name):
    with tf.variable_scope(name):
        d = tf.nn.dropout(tensor_op, keep_prob=keep_prob, name = 'd')
    return d


with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

h1 = conv2d(X, 1, 1, [5, 5, 1, 32], name='Conv1')
p1 = max_pooling(h1, 2, 2, 2, 2, name='MaxPool1')
h2 = conv2d(p1, 1, 1, [5, 5, 32, 64], name='Conv2')
p2 = max_pooling(h2, 2, 2, 2, 2, name='MaxPool2')
h3 = conv2d(p2, 1, 1, [5, 5, 64, 128], name='Conv3')
p3 = max_pooling(h3, 2, 2, 2, 2, name='MaxPool3')

flat_op = tf.reshape(p3, [-1, 4 * 4 * 128], name = 'flat_op')
f1 = relu_layer(flat_op, 1024, name='FC_Relu')
d1 = dropout_layer(f1, keep_prob=keep_prob, name='Dropout')
logits = linear(d1, NCLASS, name='FC_Linear')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot),
                                name='loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss + ALPHA*tf.reduce_sum(weight_decay_loss)
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
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :], learning_rate:u, keep_prob:0.7})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step * BATCH_SIZE
        acc_per_epoch /= total_step * BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation, learning_rate:u, keep_prob:1.0})
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

    ta = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, learning_rate:u, keep_prob:1.0})
    print("Test Accraucy : {:.2%}".format(ta / ntest))

'''
GTX 1080Ti
Epoch [ 1/30], train loss = 43.742633, train accuracy = 74.08%, valid loss = 0.696470, valid accuracy = 83.43%, duration = 7.484261(s)
Epoch [ 2/30], train loss = 0.888118, train accuracy = 74.69%, valid loss = 0.456101, valid accuracy = 85.08%, duration = 6.458541(s)
Epoch [ 3/30], train loss = 0.816911, train accuracy = 75.49%, valid loss = 0.349813, valid accuracy = 88.03%, duration = 6.274705(s)
Epoch [ 4/30], train loss = 0.712124, train accuracy = 79.11%, valid loss = 0.397725, valid accuracy = 88.07%, duration = 6.469452(s)
Epoch [ 5/30], train loss = 0.602608, train accuracy = 83.54%, valid loss = 0.177175, valid accuracy = 96.20%, duration = 6.456702(s)
Epoch [ 6/30], train loss = 0.429042, train accuracy = 89.21%, valid loss = 0.145712, valid accuracy = 96.57%, duration = 6.424071(s)
Epoch [ 7/30], train loss = 0.308233, train accuracy = 92.52%, valid loss = 0.174757, valid accuracy = 96.72%, duration = 6.376632(s)
Epoch [ 8/30], train loss = 0.265639, train accuracy = 93.73%, valid loss = 0.119694, valid accuracy = 97.02%, duration = 6.167238(s)
Epoch [ 9/30], train loss = 0.207185, train accuracy = 95.04%, valid loss = 0.151744, valid accuracy = 97.53%, duration = 6.090970(s)
Epoch [10/30], train loss = 0.168159, train accuracy = 95.74%, valid loss = 0.148356, valid accuracy = 97.85%, duration = 6.013134(s)
Epoch [11/30], train loss = 0.151621, train accuracy = 96.52%, valid loss = 0.177105, valid accuracy = 97.70%, duration = 6.062005(s)
Epoch [12/30], train loss = 0.121791, train accuracy = 97.19%, valid loss = 0.127190, valid accuracy = 98.10%, duration = 5.999162(s)
Epoch [13/30], train loss = 0.100676, train accuracy = 97.67%, valid loss = 0.139032, valid accuracy = 98.17%, duration = 6.143201(s)
Epoch [14/30], train loss = 0.091838, train accuracy = 97.99%, valid loss = 0.174494, valid accuracy = 98.28%, duration = 6.022509(s)
Epoch [15/30], train loss = 0.077207, train accuracy = 98.24%, valid loss = 0.110429, valid accuracy = 98.32%, duration = 6.025359(s)
Epoch [16/30], train loss = 0.076444, train accuracy = 98.26%, valid loss = 0.112299, valid accuracy = 98.12%, duration = 6.060747(s)
Epoch [17/30], train loss = 0.069797, train accuracy = 98.48%, valid loss = 0.157862, valid accuracy = 98.42%, duration = 6.075090(s)
Epoch [18/30], train loss = 0.059691, train accuracy = 98.70%, valid loss = 0.207504, valid accuracy = 98.42%, duration = 6.025432(s)
Epoch [19/30], train loss = 0.059397, train accuracy = 98.79%, valid loss = 0.151991, valid accuracy = 98.40%, duration = 6.021180(s)
Epoch [20/30], train loss = 0.047609, train accuracy = 98.83%, valid loss = 0.175007, valid accuracy = 98.38%, duration = 6.072882(s)
Epoch [21/30], train loss = 0.051300, train accuracy = 98.90%, valid loss = 0.160408, valid accuracy = 98.43%, duration = 6.065989(s)
Epoch [22/30], train loss = 0.041795, train accuracy = 99.03%, valid loss = 0.222659, valid accuracy = 98.57%, duration = 6.034239(s)
Epoch [23/30], train loss = 0.040842, train accuracy = 99.02%, valid loss = 0.185830, valid accuracy = 98.67%, duration = 6.038304(s)
Epoch [24/30], train loss = 0.035592, train accuracy = 99.18%, valid loss = 0.219639, valid accuracy = 98.60%, duration = 6.080778(s)
Epoch [25/30], train loss = 0.032042, train accuracy = 99.20%, valid loss = 0.215544, valid accuracy = 98.70%, duration = 6.020294(s)
Epoch [26/30], train loss = 0.033435, train accuracy = 99.23%, valid loss = 0.228346, valid accuracy = 98.62%, duration = 6.049701(s)
Epoch [27/30], train loss = 0.030857, train accuracy = 99.25%, valid loss = 0.173975, valid accuracy = 98.60%, duration = 6.070219(s)
Epoch [28/30], train loss = 0.032057, train accuracy = 99.34%, valid loss = 0.237137, valid accuracy = 98.62%, duration = 6.055686(s)
Epoch [29/30], train loss = 0.026043, train accuracy = 99.39%, valid loss = 0.186861, valid accuracy = 98.75%, duration = 6.044253(s)
Epoch [30/30], train loss = 0.022616, train accuracy = 99.45%, valid loss = 0.212284, valid accuracy = 98.60%, duration = 6.017646(s)
Duration for train : 187.353141(s)
<<< Train Finished >>>
Test Accraucy : 98.67%
'''
