# -*- coding: utf-8 -*-
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import shutil
import time

import load_data

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_cifar('./data/cifar/', seed=0, as_image=True, scaling=True)

BOARD_PATH = "./board/lab08-7_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

image_width = np.size(x_train, 1)
image_height = np.size(x_train, 2)
n_channels = np.size(x_train, 3)

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
    input_shape = tensor_op.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[input_shape[1], output_dim], dtype=tf.float32,
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
    input_shape = tensor_op.get_shape().as_list()
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[input_shape[1], output_dim], dtype=tf.float32,
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


def to_flat(tensor_op, name):
    with tf.variable_scope(name):
        input_shape = tensor_op.get_shape().as_list()
        dim = np.prod(input_shape[1:])
        flat = tf.reshape(tensor_op, [-1, dim])
    return flat


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
    X = tf.placeholder(shape=[None, image_width, image_height, n_channels], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')

h1 = conv2d(X, 1, 1, [5, 5, n_channels, 32], name='Conv1')
p1 = max_pooling(h1, 2, 2, 2, 2, name='MaxPool1')
h2 = conv2d(p1, 1, 1, [5, 5, 32, 64], name='Conv2')
p2 = max_pooling(h2, 2, 2, 2, 2, name='MaxPool2')
h3 = conv2d(p2, 1, 1, [5, 5, 64, 128], name='Conv3')
p3 = max_pooling(h3, 2, 2, 2, 2, name='MaxPool3')

flat_op = to_flat(p3, name='flat_op')
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
Epoch [ 1/30], train loss = 200.899635, train accuracy = 23.06%, valid loss = 2.339901, valid accuracy = 11.64%, duration = 6.513727(s)
Epoch [ 2/30], train loss = 2.264170, train accuracy = 14.09%, valid loss = 2.252645, valid accuracy = 13.74%, duration = 5.510284(s)
Epoch [ 3/30], train loss = 2.218608, train accuracy = 15.45%, valid loss = 2.201304, valid accuracy = 15.68%, duration = 5.540681(s)
Epoch [ 4/30], train loss = 2.208277, train accuracy = 15.33%, valid loss = 2.196798, valid accuracy = 14.92%, duration = 5.465994(s)
Epoch [ 5/30], train loss = 2.188588, train accuracy = 16.09%, valid loss = 2.115020, valid accuracy = 18.18%, duration = 5.429817(s)
Epoch [ 6/30], train loss = 2.081322, train accuracy = 19.28%, valid loss = 2.077514, valid accuracy = 17.50%, duration = 5.856875(s)
Epoch [ 7/30], train loss = 2.007986, train accuracy = 19.98%, valid loss = 2.033609, valid accuracy = 21.62%, duration = 5.984635(s)
Epoch [ 8/30], train loss = 1.958741, train accuracy = 20.00%, valid loss = 1.913893, valid accuracy = 22.74%, duration = 5.736807(s)
Epoch [ 9/30], train loss = 1.950073, train accuracy = 19.65%, valid loss = 1.968188, valid accuracy = 19.22%, duration = 5.415017(s)
Epoch [10/30], train loss = 1.958600, train accuracy = 19.18%, valid loss = 1.993525, valid accuracy = 19.78%, duration = 5.449771(s)
Epoch [11/30], train loss = 1.888652, train accuracy = 21.53%, valid loss = 1.894480, valid accuracy = 22.92%, duration = 5.442701(s)
Epoch [12/30], train loss = 1.856988, train accuracy = 23.90%, valid loss = 1.838244, valid accuracy = 26.08%, duration = 5.426718(s)
Epoch [13/30], train loss = 1.771899, train accuracy = 28.50%, valid loss = 1.696073, valid accuracy = 32.58%, duration = 5.384594(s)
Epoch [14/30], train loss = 1.676016, train accuracy = 31.82%, valid loss = 1.714358, valid accuracy = 30.44%, duration = 5.273775(s)
Epoch [15/30], train loss = 1.619320, train accuracy = 33.55%, valid loss = 1.675868, valid accuracy = 33.80%, duration = 5.250268(s)
Epoch [16/30], train loss = 1.566601, train accuracy = 35.37%, valid loss = 1.653894, valid accuracy = 34.02%, duration = 5.282101(s)
Epoch [17/30], train loss = 1.522345, train accuracy = 37.10%, valid loss = 1.966064, valid accuracy = 35.40%, duration = 5.341120(s)
Epoch [18/30], train loss = 1.478378, train accuracy = 38.35%, valid loss = 1.629811, valid accuracy = 36.50%, duration = 5.398389(s)
Epoch [19/30], train loss = 1.432097, train accuracy = 39.88%, valid loss = 1.657893, valid accuracy = 35.00%, duration = 5.290717(s)
Epoch [20/30], train loss = 1.393715, train accuracy = 41.51%, valid loss = 1.668186, valid accuracy = 36.00%, duration = 5.194412(s)
Epoch [21/30], train loss = 1.359304, train accuracy = 43.05%, valid loss = 1.736925, valid accuracy = 33.96%, duration = 5.195266(s)
Epoch [22/30], train loss = 1.310180, train accuracy = 45.67%, valid loss = 1.684366, valid accuracy = 41.26%, duration = 5.262785(s)
Epoch [23/30], train loss = 1.241499, train accuracy = 50.19%, valid loss = 1.677227, valid accuracy = 42.32%, duration = 5.377462(s)
Epoch [24/30], train loss = 1.175267, train accuracy = 53.15%, valid loss = 1.712395, valid accuracy = 45.00%, duration = 5.340678(s)
Epoch [25/30], train loss = 1.101918, train accuracy = 56.89%, valid loss = 1.674665, valid accuracy = 47.56%, duration = 5.284660(s)
Epoch [26/30], train loss = 1.032870, train accuracy = 60.46%, valid loss = 1.664520, valid accuracy = 49.42%, duration = 5.368186(s)
Epoch [27/30], train loss = 0.975860, train accuracy = 62.91%, valid loss = 1.643884, valid accuracy = 50.20%, duration = 5.318499(s)
Epoch [28/30], train loss = 0.916641, train accuracy = 65.61%, valid loss = 1.690655, valid accuracy = 50.60%, duration = 5.277067(s)
Epoch [29/30], train loss = 0.885096, train accuracy = 66.76%, valid loss = 1.773383, valid accuracy = 50.08%, duration = 5.333714(s)
Epoch [30/30], train loss = 0.840563, train accuracy = 68.64%, valid loss = 1.778045, valid accuracy = 50.46%, duration = 5.216823(s)
Duration for train : 168.274887(s)
<<< Train Finished >>>
Test Accraucy : 49.18%
'''
