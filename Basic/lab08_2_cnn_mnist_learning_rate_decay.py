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
Epoch [ 1/30], train loss = 61.855218, train accuracy = 91.49%, valid loss = 1.074106, valid accuracy = 95.03%, duration = 7.297709(s)
Epoch [ 2/30], train loss = 0.779498, train accuracy = 94.98%, valid loss = 0.612148, valid accuracy = 94.87%, duration = 5.960168(s)
Epoch [ 3/30], train loss = 0.301694, train accuracy = 95.41%, valid loss = 0.404263, valid accuracy = 95.13%, duration = 5.903726(s)
Epoch [ 4/30], train loss = 0.216653, train accuracy = 95.89%, valid loss = 0.257635, valid accuracy = 95.52%, duration = 5.828407(s)
Epoch [ 5/30], train loss = 0.187013, train accuracy = 96.42%, valid loss = 0.223977, valid accuracy = 95.73%, duration = 5.916283(s)
Epoch [ 6/30], train loss = 0.150490, train accuracy = 97.15%, valid loss = 0.166472, valid accuracy = 96.63%, duration = 5.836584(s)
Epoch [ 7/30], train loss = 0.121634, train accuracy = 97.66%, valid loss = 0.210902, valid accuracy = 96.45%, duration = 5.909424(s)
Epoch [ 8/30], train loss = 0.108062, train accuracy = 97.93%, valid loss = 0.215350, valid accuracy = 97.00%, duration = 5.884050(s)
Epoch [ 9/30], train loss = 0.075905, train accuracy = 98.49%, valid loss = 0.152548, valid accuracy = 97.88%, duration = 5.919775(s)
Epoch [10/30], train loss = 0.064584, train accuracy = 98.74%, valid loss = 0.224286, valid accuracy = 97.38%, duration = 5.892210(s)
Epoch [11/30], train loss = 0.061698, train accuracy = 98.83%, valid loss = 0.160143, valid accuracy = 98.02%, duration = 5.893835(s)
Epoch [12/30], train loss = 0.044295, train accuracy = 99.15%, valid loss = 0.210050, valid accuracy = 97.92%, duration = 5.885631(s)
Epoch [13/30], train loss = 0.041291, train accuracy = 99.24%, valid loss = 0.195910, valid accuracy = 98.08%, duration = 5.852824(s)
Epoch [14/30], train loss = 0.033464, train accuracy = 99.35%, valid loss = 0.212777, valid accuracy = 97.92%, duration = 5.816824(s)
Epoch [15/30], train loss = 0.033235, train accuracy = 99.49%, valid loss = 0.266609, valid accuracy = 97.98%, duration = 5.902916(s)
Epoch [16/30], train loss = 0.029676, train accuracy = 99.51%, valid loss = 0.226310, valid accuracy = 98.45%, duration = 5.918966(s)
Epoch [17/30], train loss = 0.019848, train accuracy = 99.63%, valid loss = 0.241289, valid accuracy = 98.18%, duration = 5.880997(s)
Epoch [18/30], train loss = 0.020628, train accuracy = 99.65%, valid loss = 0.254130, valid accuracy = 98.20%, duration = 5.849389(s)
Epoch [19/30], train loss = 0.017130, train accuracy = 99.71%, valid loss = 0.328758, valid accuracy = 98.17%, duration = 5.871610(s)
Epoch [20/30], train loss = 0.012141, train accuracy = 99.80%, valid loss = 0.284120, valid accuracy = 98.40%, duration = 5.807514(s)
Epoch [21/30], train loss = 0.014659, train accuracy = 99.80%, valid loss = 0.257571, valid accuracy = 98.42%, duration = 5.889241(s)
Epoch [22/30], train loss = 0.015684, train accuracy = 99.78%, valid loss = 0.282967, valid accuracy = 98.50%, duration = 5.872723(s)
Epoch [23/30], train loss = 0.010486, train accuracy = 99.84%, valid loss = 0.286500, valid accuracy = 98.58%, duration = 5.902422(s)
Epoch [24/30], train loss = 0.007323, train accuracy = 99.86%, valid loss = 0.334210, valid accuracy = 98.47%, duration = 5.902370(s)
Epoch [25/30], train loss = 0.007591, train accuracy = 99.87%, valid loss = 0.472680, valid accuracy = 98.20%, duration = 5.899461(s)
Epoch [26/30], train loss = 0.009259, train accuracy = 99.86%, valid loss = 0.310762, valid accuracy = 98.57%, duration = 5.875110(s)
Epoch [27/30], train loss = 0.007069, train accuracy = 99.90%, valid loss = 0.298091, valid accuracy = 98.58%, duration = 5.869275(s)
Epoch [28/30], train loss = 0.007210, train accuracy = 99.90%, valid loss = 0.283738, valid accuracy = 98.60%, duration = 5.889427(s)
Epoch [29/30], train loss = 0.004836, train accuracy = 99.92%, valid loss = 0.269304, valid accuracy = 98.62%, duration = 5.877535(s)
Epoch [30/30], train loss = 0.003796, train accuracy = 99.93%, valid loss = 0.249079, valid accuracy = 98.67%, duration = 5.809046(s)
Duration for train : 180.362757(s)
<<< Train Finished >>>
Test Accraucy : 98.56%
'''
