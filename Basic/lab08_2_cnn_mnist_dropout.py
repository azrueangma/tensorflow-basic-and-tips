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


def dropout_layer(tensor_op, keep_prob, name):
    with tf.variable_scope(name):
        d = tf.nn.dropout(tensor_op, keep_prob=keep_prob, name = 'd')
    return d

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

h1 = conv2d(X, 1, 1, [5, 5, 1, 32], name='Conv1')
p1 = max_pooling(h1, 2, 2, 2, 2, name='MaxPool1')
h2 = conv2d(p1, 1, 1, [5, 5, 32, 64], name='Conv2')
p2 = max_pooling(h2, 2, 2, 2, 2, name='MaxPool2')
h3 = conv2d(p2, 1, 1, [5, 5, 64, 128], name='Conv3')
p3 = max_pooling(h3, 2, 2, 2, 2, name='MaxPool3')

flat_op = tf.reshape(p3, [-1, 4 * 4 * 128], name = 'flat_op')
f1 = relu_layer(flat_op, 1024, name='FC_Relu')
d1 = dropout_layer(f1, keep_prob=keep_prob, name='Dropout')
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
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :], keep_prob:0.7})
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
                                        avg_validation_loss: epoch_valid_loss, avg_validation_acc: epoch_valid_acc, keep_prob:1.0})
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

    ta = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob:1.0})
    print("Test Accraucy : {:.2%}".format(ta / ntest))

'''
GTX 1080Ti
Epoch [ 1/30], train loss = 71.983186, train accuracy = 91.14%, valid loss = 1.360490, valid accuracy = 95.15%, duration = 7.975482(s)
Epoch [ 2/30], train loss = 0.881136, train accuracy = 94.87%, valid loss = 0.472438, valid accuracy = 93.88%, duration = 6.261841(s)
Epoch [ 3/30], train loss = 0.304386, train accuracy = 95.16%, valid loss = 0.323132, valid accuracy = 94.73%, duration = 6.382113(s)
Epoch [ 4/30], train loss = 0.247704, train accuracy = 95.63%, valid loss = 0.264604, valid accuracy = 95.58%, duration = 6.261739(s)
Epoch [ 5/30], train loss = 0.230500, train accuracy = 95.95%, valid loss = 0.260140, valid accuracy = 95.83%, duration = 6.274034(s)
Epoch [ 6/30], train loss = 0.176803, train accuracy = 96.72%, valid loss = 0.240807, valid accuracy = 96.05%, duration = 6.254083(s)
Epoch [ 7/30], train loss = 0.157753, train accuracy = 97.15%, valid loss = 0.214881, valid accuracy = 97.07%, duration = 6.279343(s)
Epoch [ 8/30], train loss = 0.130400, train accuracy = 97.54%, valid loss = 0.241506, valid accuracy = 96.98%, duration = 6.350048(s)
Epoch [ 9/30], train loss = 0.130600, train accuracy = 97.78%, valid loss = 0.175573, valid accuracy = 97.87%, duration = 6.234193(s)
Epoch [10/30], train loss = 0.099406, train accuracy = 98.09%, valid loss = 0.213321, valid accuracy = 97.15%, duration = 6.359722(s)
Epoch [11/30], train loss = 0.091582, train accuracy = 98.34%, valid loss = 0.182144, valid accuracy = 97.70%, duration = 6.228282(s)
Epoch [12/30], train loss = 0.089791, train accuracy = 98.39%, valid loss = 0.214165, valid accuracy = 98.00%, duration = 6.283438(s)
Epoch [13/30], train loss = 0.088815, train accuracy = 98.42%, valid loss = 0.187166, valid accuracy = 97.70%, duration = 6.247146(s)
Epoch [14/30], train loss = 0.081354, train accuracy = 98.54%, valid loss = 0.190199, valid accuracy = 98.07%, duration = 6.312285(s)
Epoch [15/30], train loss = 0.067881, train accuracy = 98.77%, valid loss = 0.213103, valid accuracy = 97.67%, duration = 6.283272(s)
Epoch [16/30], train loss = 0.071438, train accuracy = 98.76%, valid loss = 0.147501, valid accuracy = 98.12%, duration = 6.314790(s)
Epoch [17/30], train loss = 0.071684, train accuracy = 98.85%, valid loss = 0.206304, valid accuracy = 97.97%, duration = 6.236103(s)
Epoch [18/30], train loss = 0.063186, train accuracy = 98.82%, valid loss = 0.267870, valid accuracy = 97.60%, duration = 6.310880(s)
Epoch [19/30], train loss = 0.064834, train accuracy = 98.91%, valid loss = 0.155728, valid accuracy = 97.33%, duration = 6.254456(s)
Epoch [20/30], train loss = 0.069445, train accuracy = 98.84%, valid loss = 0.230861, valid accuracy = 97.83%, duration = 6.199328(s)
Epoch [21/30], train loss = 0.067353, train accuracy = 98.91%, valid loss = 0.254405, valid accuracy = 97.80%, duration = 6.216170(s)
Epoch [22/30], train loss = 0.055431, train accuracy = 99.02%, valid loss = 0.276950, valid accuracy = 98.08%, duration = 6.213202(s)
Epoch [23/30], train loss = 0.061371, train accuracy = 99.05%, valid loss = 0.234498, valid accuracy = 97.20%, duration = 6.234744(s)
Epoch [24/30], train loss = 0.067273, train accuracy = 98.91%, valid loss = 0.244470, valid accuracy = 98.22%, duration = 6.187704(s)
Epoch [25/30], train loss = 0.053583, train accuracy = 99.05%, valid loss = 0.228938, valid accuracy = 98.02%, duration = 6.269302(s)
Epoch [26/30], train loss = 0.076276, train accuracy = 98.86%, valid loss = 0.209932, valid accuracy = 97.95%, duration = 6.232442(s)
Epoch [27/30], train loss = 0.063741, train accuracy = 99.11%, valid loss = 0.206586, valid accuracy = 98.48%, duration = 6.165354(s)
Epoch [28/30], train loss = 0.055489, train accuracy = 99.10%, valid loss = 0.238409, valid accuracy = 98.25%, duration = 6.283799(s)
Epoch [29/30], train loss = 0.059995, train accuracy = 99.07%, valid loss = 0.364559, valid accuracy = 97.62%, duration = 6.307492(s)
Epoch [30/30], train loss = 0.062804, train accuracy = 98.98%, valid loss = 0.178412, valid accuracy = 97.75%, duration = 6.248280(s)
Duration for train : 192.332412(s)
<<< Train Finished >>>
Test Accraucy : 97.58%
'''
