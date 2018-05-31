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

BOARD_PATH = "./board/lab08-2_board"
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
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

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
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :], learning_rate:u})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step * BATCH_SIZE
        acc_per_epoch /= total_step * BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation, learning_rate:u})
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

    ta = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, learning_rate:u})
    print("Test Accraucy : {:.2%}".format(ta / ntest))

'''
Epoch [ 1/30], train loss = 50.480966, train accuracy = 92.29%, valid loss = 1.071147, valid accuracy = 95.32%, duration = 7.188274(s)
Epoch [ 2/30], train loss = 0.577060, train accuracy = 95.57%, valid loss = 0.334411, valid accuracy = 96.07%, duration = 6.098888(s)
Epoch [ 3/30], train loss = 0.268173, train accuracy = 95.70%, valid loss = 0.323395, valid accuracy = 95.10%, duration = 6.153902(s)
Epoch [ 4/30], train loss = 0.211657, train accuracy = 96.35%, valid loss = 0.317598, valid accuracy = 96.23%, duration = 6.230876(s)
Epoch [ 5/30], train loss = 0.139600, train accuracy = 97.26%, valid loss = 0.137754, valid accuracy = 97.52%, duration = 6.079298(s)
Epoch [ 6/30], train loss = 0.144554, train accuracy = 97.42%, valid loss = 0.179263, valid accuracy = 97.18%, duration = 6.229432(s)
Epoch [ 7/30], train loss = 0.091342, train accuracy = 98.16%, valid loss = 0.217744, valid accuracy = 97.37%, duration = 6.350821(s)
Epoch [ 8/30], train loss = 0.085016, train accuracy = 98.43%, valid loss = 0.124634, valid accuracy = 98.35%, duration = 6.190582(s)
Epoch [ 9/30], train loss = 0.070699, train accuracy = 98.81%, valid loss = 0.159391, valid accuracy = 97.98%, duration = 6.130049(s)
Epoch [10/30], train loss = 0.051355, train accuracy = 99.03%, valid loss = 0.145041, valid accuracy = 98.25%, duration = 6.318030(s)
Epoch [11/30], train loss = 0.044517, train accuracy = 99.17%, valid loss = 0.197262, valid accuracy = 97.98%, duration = 6.464935(s)
Epoch [12/30], train loss = 0.054397, train accuracy = 99.11%, valid loss = 0.202694, valid accuracy = 98.32%, duration = 6.628080(s)
Epoch [13/30], train loss = 0.029912, train accuracy = 99.44%, valid loss = 0.189768, valid accuracy = 98.03%, duration = 6.348047(s)
Epoch [14/30], train loss = 0.030463, train accuracy = 99.43%, valid loss = 0.138600, valid accuracy = 98.57%, duration = 6.257944(s)
Epoch [15/30], train loss = 0.024363, train accuracy = 99.54%, valid loss = 0.152262, valid accuracy = 98.48%, duration = 6.248278(s)
Epoch [16/30], train loss = 0.022390, train accuracy = 99.60%, valid loss = 0.142153, valid accuracy = 98.65%, duration = 6.196154(s)
Epoch [17/30], train loss = 0.014270, train accuracy = 99.70%, valid loss = 0.151353, valid accuracy = 98.53%, duration = 6.274406(s)
Epoch [18/30], train loss = 0.016993, train accuracy = 99.66%, valid loss = 0.159162, valid accuracy = 98.73%, duration = 6.249281(s)
Epoch [19/30], train loss = 0.012001, train accuracy = 99.72%, valid loss = 0.195991, valid accuracy = 98.82%, duration = 6.224332(s)
Epoch [20/30], train loss = 0.013719, train accuracy = 99.73%, valid loss = 0.192748, valid accuracy = 98.58%, duration = 6.742021(s)
Epoch [21/30], train loss = 0.009409, train accuracy = 99.80%, valid loss = 0.186807, valid accuracy = 98.85%, duration = 6.543235(s)
Epoch [22/30], train loss = 0.012568, train accuracy = 99.76%, valid loss = 0.180936, valid accuracy = 98.97%, duration = 6.287116(s)
Epoch [23/30], train loss = 0.008316, train accuracy = 99.85%, valid loss = 0.162876, valid accuracy = 98.98%, duration = 6.388044(s)
Epoch [24/30], train loss = 0.007413, train accuracy = 99.84%, valid loss = 0.212497, valid accuracy = 98.73%, duration = 6.151240(s)
Epoch [25/30], train loss = 0.007528, train accuracy = 99.86%, valid loss = 0.153053, valid accuracy = 98.98%, duration = 6.188583(s)
Epoch [26/30], train loss = 0.005476, train accuracy = 99.88%, valid loss = 0.192748, valid accuracy = 98.82%, duration = 6.183470(s)
Epoch [27/30], train loss = 0.003120, train accuracy = 99.92%, valid loss = 0.178866, valid accuracy = 98.70%, duration = 6.138709(s)
Epoch [28/30], train loss = 0.005593, train accuracy = 99.90%, valid loss = 0.189623, valid accuracy = 98.77%, duration = 6.123214(s)
Epoch [29/30], train loss = 0.006486, train accuracy = 99.90%, valid loss = 0.173785, valid accuracy = 98.97%, duration = 6.126772(s)
Epoch [30/30], train loss = 0.003388, train accuracy = 99.92%, valid loss = 0.174410, valid accuracy = 98.98%, duration = 6.102933(s)
Duration for train : 190.959576(s)
<<< Train Finished >>>
Test Accraucy : 98.66%
'''
