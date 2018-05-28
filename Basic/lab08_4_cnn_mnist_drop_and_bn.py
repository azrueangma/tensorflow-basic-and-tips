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


def dropout_layer(tensor_op, keep_prob, name):
    with tf.variable_scope(name):
        d = tf.nn.dropout(tensor_op, keep_prob=keep_prob, name = 'd')
    return d


def bn_layer(x, is_training, name):
    with tf.variable_scope(name):
        bn = tf.contrib.layers.batch_norm(x, updates_collections=None, scale=True, is_training=is_training)
        post_activation = tf.nn.relu(bn, name='relu')
    return post_activation


with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
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
d1 = dropout_layer(f1, keep_prob=keep_prob, name='Dropout')
logits = linear(d1, NCLASS, name='FC_Linear')

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
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :],
                                                                   is_training:True,  keep_prob:0.7})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step * BATCH_SIZE
        acc_per_epoch /= total_step * BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation, is_training:False,  keep_prob:1.0})
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

    ta = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, is_training:False, keep_prob:1.0})
    print("Test Accraucy : {:.2%}".format(ta / ntest))

'''
GTX 1080Ti
Epoch [ 1/30], train loss = 0.224251, train accuracy = 93.70%, valid loss = 0.075233, valid accuracy = 97.85%, duration = 8.395782(s)
Epoch [ 2/30], train loss = 0.080966, train accuracy = 97.52%, valid loss = 0.058354, valid accuracy = 98.17%, duration = 7.314087(s)
Epoch [ 3/30], train loss = 0.056383, train accuracy = 98.26%, valid loss = 0.048533, valid accuracy = 98.73%, duration = 7.324345(s)
Epoch [ 4/30], train loss = 0.039994, train accuracy = 98.78%, valid loss = 0.040624, valid accuracy = 98.67%, duration = 7.374239(s)
Epoch [ 5/30], train loss = 0.031402, train accuracy = 99.03%, valid loss = 0.041567, valid accuracy = 98.93%, duration = 7.389460(s)
Epoch [ 6/30], train loss = 0.024718, train accuracy = 99.19%, valid loss = 0.028521, valid accuracy = 99.00%, duration = 7.480419(s)
Epoch [ 7/30], train loss = 0.016906, train accuracy = 99.44%, valid loss = 0.052489, valid accuracy = 98.50%, duration = 7.576799(s)
Epoch [ 8/30], train loss = 0.016159, train accuracy = 99.49%, valid loss = 0.059493, valid accuracy = 98.50%, duration = 7.721816(s)
Epoch [ 9/30], train loss = 0.015756, train accuracy = 99.52%, valid loss = 0.044172, valid accuracy = 98.92%, duration = 7.858904(s)
Epoch [10/30], train loss = 0.011488, train accuracy = 99.62%, valid loss = 0.034589, valid accuracy = 99.27%, duration = 7.439918(s)
Epoch [11/30], train loss = 0.011561, train accuracy = 99.61%, valid loss = 0.042726, valid accuracy = 98.83%, duration = 7.678686(s)
Epoch [12/30], train loss = 0.009654, train accuracy = 99.71%, valid loss = 0.042981, valid accuracy = 99.03%, duration = 7.369441(s)
Epoch [13/30], train loss = 0.009063, train accuracy = 99.69%, valid loss = 0.064979, valid accuracy = 98.35%, duration = 7.560632(s)
Epoch [14/30], train loss = 0.009345, train accuracy = 99.71%, valid loss = 0.039730, valid accuracy = 99.07%, duration = 7.560968(s)
Epoch [15/30], train loss = 0.007291, train accuracy = 99.79%, valid loss = 0.052404, valid accuracy = 98.85%, duration = 7.308596(s)
Epoch [16/30], train loss = 0.007494, train accuracy = 99.76%, valid loss = 0.065655, valid accuracy = 98.70%, duration = 7.431540(s)
Epoch [17/30], train loss = 0.008870, train accuracy = 99.74%, valid loss = 0.046358, valid accuracy = 99.08%, duration = 7.706116(s)
Epoch [18/30], train loss = 0.006412, train accuracy = 99.80%, valid loss = 0.063334, valid accuracy = 99.02%, duration = 7.473938(s)
Epoch [19/30], train loss = 0.007110, train accuracy = 99.79%, valid loss = 0.073273, valid accuracy = 98.80%, duration = 7.353324(s)
Epoch [20/30], train loss = 0.006090, train accuracy = 99.83%, valid loss = 0.069522, valid accuracy = 98.87%, duration = 7.208037(s)
Epoch [21/30], train loss = 0.006046, train accuracy = 99.82%, valid loss = 0.073471, valid accuracy = 99.03%, duration = 7.390309(s)
Epoch [22/30], train loss = 0.007187, train accuracy = 99.81%, valid loss = 0.077222, valid accuracy = 98.88%, duration = 7.287893(s)
Epoch [23/30], train loss = 0.005427, train accuracy = 99.84%, valid loss = 0.079699, valid accuracy = 98.93%, duration = 7.224590(s)
Epoch [24/30], train loss = 0.005352, train accuracy = 99.86%, valid loss = 0.076574, valid accuracy = 99.07%, duration = 7.268875(s)
Epoch [25/30], train loss = 0.008511, train accuracy = 99.80%, valid loss = 0.058702, valid accuracy = 99.22%, duration = 7.266805(s)
Epoch [26/30], train loss = 0.005703, train accuracy = 99.84%, valid loss = 0.111303, valid accuracy = 98.92%, duration = 7.180955(s)
Epoch [27/30], train loss = 0.006486, train accuracy = 99.82%, valid loss = 0.075943, valid accuracy = 99.05%, duration = 7.195368(s)
Epoch [28/30], train loss = 0.006659, train accuracy = 99.83%, valid loss = 0.069782, valid accuracy = 99.00%, duration = 7.244043(s)
Epoch [29/30], train loss = 0.005291, train accuracy = 99.85%, valid loss = 0.073314, valid accuracy = 99.10%, duration = 7.220531(s)
Epoch [30/30], train loss = 0.004778, train accuracy = 99.89%, valid loss = 0.084239, valid accuracy = 99.10%, duration = 7.305650(s)
Duration for train : 225.680591(s)
<<< Train Finished >>>
Test Accraucy : 99.05%
'''
