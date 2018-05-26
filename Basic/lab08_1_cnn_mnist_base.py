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
Epoch [ 1/30], train loss = 69.036995, train accuracy = 91.10%, valid loss = 1.283954, valid accuracy = 96.62%, duration = 6.171185(s)
Epoch [ 2/30], train loss = 1.169566, train accuracy = 96.04%, valid loss = 0.630596, valid accuracy = 95.92%, duration = 5.312827(s)
Epoch [ 3/30], train loss = 0.439967, train accuracy = 96.81%, valid loss = 0.400938, valid accuracy = 95.88%, duration = 5.302464(s)
Epoch [ 4/30], train loss = 0.292906, train accuracy = 97.06%, valid loss = 0.351068, valid accuracy = 96.82%, duration = 5.295797(s)
Epoch [ 5/30], train loss = 0.279480, train accuracy = 97.18%, valid loss = 0.528072, valid accuracy = 96.18%, duration = 5.311831(s)
Epoch [ 6/30], train loss = 0.288272, train accuracy = 97.32%, valid loss = 0.269747, valid accuracy = 96.70%, duration = 5.297302(s)
Epoch [ 7/30], train loss = 0.194420, train accuracy = 97.80%, valid loss = 0.163581, valid accuracy = 97.18%, duration = 5.326022(s)
Epoch [ 8/30], train loss = 0.163184, train accuracy = 97.98%, valid loss = 0.310952, valid accuracy = 96.07%, duration = 5.301883(s)
Epoch [ 9/30], train loss = 0.171471, train accuracy = 98.06%, valid loss = 0.318407, valid accuracy = 96.97%, duration = 5.340584(s)
Epoch [10/30], train loss = 0.161173, train accuracy = 98.27%, valid loss = 0.403049, valid accuracy = 97.12%, duration = 5.302447(s)
Epoch [11/30], train loss = 0.180605, train accuracy = 98.44%, valid loss = 0.259148, valid accuracy = 97.75%, duration = 5.319323(s)
Epoch [12/30], train loss = 0.094278, train accuracy = 98.84%, valid loss = 0.245751, valid accuracy = 97.52%, duration = 5.305499(s)
Epoch [13/30], train loss = 0.111316, train accuracy = 98.76%, valid loss = 0.312322, valid accuracy = 97.50%, duration = 5.337669(s)
Epoch [14/30], train loss = 0.101861, train accuracy = 98.95%, valid loss = 0.344512, valid accuracy = 98.25%, duration = 5.280129(s)
Epoch [15/30], train loss = 0.094754, train accuracy = 99.01%, valid loss = 0.282543, valid accuracy = 97.65%, duration = 5.288226(s)
Epoch [16/30], train loss = 0.106008, train accuracy = 99.01%, valid loss = 0.463566, valid accuracy = 97.33%, duration = 5.294326(s)
Epoch [17/30], train loss = 0.102507, train accuracy = 99.05%, valid loss = 0.299066, valid accuracy = 98.15%, duration = 5.297423(s)
Epoch [18/30], train loss = 0.083335, train accuracy = 99.22%, valid loss = 0.317557, valid accuracy = 97.92%, duration = 5.302902(s)
Epoch [19/30], train loss = 0.104153, train accuracy = 99.10%, valid loss = 0.417045, valid accuracy = 98.00%, duration = 5.337284(s)
Epoch [20/30], train loss = 0.069875, train accuracy = 99.35%, valid loss = 0.352112, valid accuracy = 97.85%, duration = 5.308168(s)
Epoch [21/30], train loss = 0.064472, train accuracy = 99.35%, valid loss = 0.316677, valid accuracy = 98.43%, duration = 5.350518(s)
Epoch [22/30], train loss = 0.092686, train accuracy = 99.34%, valid loss = 0.463703, valid accuracy = 97.93%, duration = 5.313004(s)
Epoch [23/30], train loss = 0.073073, train accuracy = 99.44%, valid loss = 0.386083, valid accuracy = 98.13%, duration = 5.293895(s)
Epoch [24/30], train loss = 0.072758, train accuracy = 99.43%, valid loss = 0.487467, valid accuracy = 98.13%, duration = 5.319027(s)
Epoch [25/30], train loss = 0.103619, train accuracy = 99.27%, valid loss = 0.412348, valid accuracy = 98.28%, duration = 5.306162(s)
Epoch [26/30], train loss = 0.051800, train accuracy = 99.58%, valid loss = 0.410174, valid accuracy = 98.40%, duration = 5.317333(s)
Epoch [27/30], train loss = 0.077051, train accuracy = 99.39%, valid loss = 0.374284, valid accuracy = 98.35%, duration = 5.306609(s)
Epoch [28/30], train loss = 0.065178, train accuracy = 99.41%, valid loss = 0.319922, valid accuracy = 98.28%, duration = 5.290809(s)
Epoch [29/30], train loss = 0.074810, train accuracy = 99.41%, valid loss = 0.473712, valid accuracy = 98.22%, duration = 5.292579(s)
Epoch [30/30], train loss = 0.088996, train accuracy = 99.45%, valid loss = 0.427322, valid accuracy = 98.33%, duration = 5.295960(s)
Duration for train : 163.055973(s)
<<< Train Finished >>>
Test Accraucy : 98.03%
'''
