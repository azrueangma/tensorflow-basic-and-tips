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

def bn_layer(x, is_training, name):
    with tf.variable_scope(name):
        bn = tf.contrib.layers.batch_norm(x, updates_collections=None, scale=True, is_training=is_training)
        post_activation = tf.nn.leaky_relu(bn, name='leaky_relu')
    return post_activation

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, image_width, image_height, 1], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    is_training = tf.placeholder(tf.bool, name='is_training')

h1 = conv2d(X, 1, 1, [5, 5, 1, 32], name='Conv1')
b1 = bn_layer(h1, is_training, name='bn1')
h2 = conv2d(b1, 1, 1, [5, 5, 32, 64], name='Conv2')
b2 = bn_layer(h2, is_training, name='bn2')
h3 = conv2d(b2, 1, 1, [5, 5, 64, 128], name='Conv3')
b3 = bn_layer(h3, is_training, name='bn3')

flat_op = tf.reshape(b3, [-1, 28 * 28 * 128], name = 'flat_op')
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
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :], is_training:True})
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
Epoch [ 1/30], train loss = 0.820616, train accuracy = 93.36%, valid loss = 0.160984, valid accuracy = 96.23%, duration = 42.102062(s)
Epoch [ 2/30], train loss = 0.137022, train accuracy = 96.72%, valid loss = 0.124485, valid accuracy = 97.10%, duration = 40.948998(s)
Epoch [ 3/30], train loss = 0.088490, train accuracy = 97.81%, valid loss = 0.117157, valid accuracy = 97.62%, duration = 40.877700(s)
Epoch [ 4/30], train loss = 0.070800, train accuracy = 98.44%, valid loss = 0.101075, valid accuracy = 98.00%, duration = 40.903972(s)
Epoch [ 5/30], train loss = 0.058258, train accuracy = 98.68%, valid loss = 0.112592, valid accuracy = 97.93%, duration = 40.934499(s)
Epoch [ 6/30], train loss = 0.049631, train accuracy = 99.00%, valid loss = 0.142543, valid accuracy = 97.80%, duration = 40.900226(s)
Epoch [ 7/30], train loss = 0.038983, train accuracy = 99.19%, valid loss = 0.141517, valid accuracy = 98.10%, duration = 40.866753(s)
Epoch [ 8/30], train loss = 0.033704, train accuracy = 99.31%, valid loss = 0.105326, valid accuracy = 98.43%, duration = 40.951791(s)
Epoch [ 9/30], train loss = 0.026514, train accuracy = 99.42%, valid loss = 0.158577, valid accuracy = 97.83%, duration = 40.939239(s)
Epoch [10/30], train loss = 0.027756, train accuracy = 99.41%, valid loss = 0.164004, valid accuracy = 97.98%, duration = 40.910259(s)
Epoch [11/30], train loss = 0.015382, train accuracy = 99.62%, valid loss = 0.092952, valid accuracy = 98.65%, duration = 40.956067(s)
Epoch [12/30], train loss = 0.017793, train accuracy = 99.60%, valid loss = 0.099941, valid accuracy = 98.65%, duration = 40.906945(s)
Epoch [13/30], train loss = 0.012797, train accuracy = 99.71%, valid loss = 0.118730, valid accuracy = 98.33%, duration = 40.936645(s)
Epoch [14/30], train loss = 0.017615, train accuracy = 99.68%, valid loss = 0.087722, valid accuracy = 98.68%, duration = 40.924638(s)
Epoch [15/30], train loss = 0.011137, train accuracy = 99.74%, valid loss = 0.129889, valid accuracy = 98.13%, duration = 40.902193(s)
Epoch [16/30], train loss = 0.011545, train accuracy = 99.76%, valid loss = 0.097541, valid accuracy = 98.80%, duration = 40.904945(s)
Epoch [17/30], train loss = 0.012950, train accuracy = 99.69%, valid loss = 0.103547, valid accuracy = 98.58%, duration = 40.900688(s)
Epoch [18/30], train loss = 0.008995, train accuracy = 99.80%, valid loss = 0.090064, valid accuracy = 98.67%, duration = 40.894740(s)
Epoch [19/30], train loss = 0.012497, train accuracy = 99.74%, valid loss = 0.110868, valid accuracy = 98.52%, duration = 40.899242(s)
Epoch [20/30], train loss = 0.009024, train accuracy = 99.81%, valid loss = 0.121217, valid accuracy = 98.67%, duration = 40.916661(s)
Epoch [21/30], train loss = 0.011802, train accuracy = 99.76%, valid loss = 0.096543, valid accuracy = 98.83%, duration = 40.950034(s)
Epoch [22/30], train loss = 0.010721, train accuracy = 99.77%, valid loss = 0.093054, valid accuracy = 98.85%, duration = 40.932916(s)
Epoch [23/30], train loss = 0.008092, train accuracy = 99.83%, valid loss = 0.125067, valid accuracy = 98.58%, duration = 40.904046(s)
Epoch [24/30], train loss = 0.009657, train accuracy = 99.81%, valid loss = 0.118037, valid accuracy = 98.70%, duration = 40.963870(s)
Epoch [25/30], train loss = 0.007336, train accuracy = 99.85%, valid loss = 0.094512, valid accuracy = 98.95%, duration = 40.926419(s)
Epoch [26/30], train loss = 0.010637, train accuracy = 99.80%, valid loss = 0.128749, valid accuracy = 98.67%, duration = 40.908247(s)
Epoch [27/30], train loss = 0.007724, train accuracy = 99.86%, valid loss = 0.124485, valid accuracy = 98.85%, duration = 40.867022(s)
Epoch [28/30], train loss = 0.008106, train accuracy = 99.81%, valid loss = 0.144652, valid accuracy = 98.70%, duration = 40.902088(s)
Epoch [29/30], train loss = 0.009270, train accuracy = 99.80%, valid loss = 0.139713, valid accuracy = 98.80%, duration = 40.892692(s)
Epoch [30/30], train loss = 0.009072, train accuracy = 99.85%, valid loss = 0.116736, valid accuracy = 98.88%, duration = 40.923705(s)
Duration for train : 1245.798934(s)
<<< Train Finished >>>
'''
