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
Epoch [ 1/30], train loss = 0.198616, train accuracy = 94.67%, valid loss = 0.079332, valid accuracy = 97.62%, duration = 8.502317(s)
Epoch [ 2/30], train loss = 0.062671, train accuracy = 97.96%, valid loss = 0.062961, valid accuracy = 98.25%, duration = 7.272597(s)
Epoch [ 3/30], train loss = 0.042805, train accuracy = 98.69%, valid loss = 0.046908, valid accuracy = 98.65%, duration = 7.408269(s)
Epoch [ 4/30], train loss = 0.030084, train accuracy = 99.04%, valid loss = 0.050262, valid accuracy = 98.52%, duration = 7.064562(s)
Epoch [ 5/30], train loss = 0.022156, train accuracy = 99.25%, valid loss = 0.058490, valid accuracy = 98.53%, duration = 7.134675(s)
Epoch [ 6/30], train loss = 0.018413, train accuracy = 99.40%, valid loss = 0.061452, valid accuracy = 98.33%, duration = 7.108449(s)
Epoch [ 7/30], train loss = 0.012619, train accuracy = 99.58%, valid loss = 0.056415, valid accuracy = 98.77%, duration = 7.162494(s)
Epoch [ 8/30], train loss = 0.011978, train accuracy = 99.61%, valid loss = 0.056922, valid accuracy = 98.53%, duration = 7.251233(s)
Epoch [ 9/30], train loss = 0.010619, train accuracy = 99.66%, valid loss = 0.055182, valid accuracy = 98.72%, duration = 7.369036(s)
Epoch [10/30], train loss = 0.010871, train accuracy = 99.66%, valid loss = 0.053804, valid accuracy = 98.70%, duration = 7.272920(s)
Epoch [11/30], train loss = 0.007018, train accuracy = 99.80%, valid loss = 0.046012, valid accuracy = 98.88%, duration = 7.387367(s)
Epoch [12/30], train loss = 0.007576, train accuracy = 99.76%, valid loss = 0.055152, valid accuracy = 98.68%, duration = 7.171099(s)
Epoch [13/30], train loss = 0.008825, train accuracy = 99.71%, valid loss = 0.115257, valid accuracy = 97.85%, duration = 7.145249(s)
Epoch [14/30], train loss = 0.006711, train accuracy = 99.78%, valid loss = 0.068705, valid accuracy = 98.70%, duration = 7.156266(s)
Epoch [15/30], train loss = 0.004414, train accuracy = 99.87%, valid loss = 0.055108, valid accuracy = 99.02%, duration = 7.407526(s)
Epoch [16/30], train loss = 0.006463, train accuracy = 99.77%, valid loss = 0.049174, valid accuracy = 99.13%, duration = 7.253779(s)
Epoch [17/30], train loss = 0.006852, train accuracy = 99.80%, valid loss = 0.063685, valid accuracy = 98.83%, duration = 7.316010(s)
Epoch [18/30], train loss = 0.004376, train accuracy = 99.89%, valid loss = 0.063449, valid accuracy = 98.70%, duration = 7.329724(s)
Epoch [19/30], train loss = 0.007364, train accuracy = 99.80%, valid loss = 0.061465, valid accuracy = 98.88%, duration = 7.448450(s)
Epoch [20/30], train loss = 0.003400, train accuracy = 99.89%, valid loss = 0.073690, valid accuracy = 98.75%, duration = 7.139037(s)
Epoch [21/30], train loss = 0.005366, train accuracy = 99.86%, valid loss = 0.070993, valid accuracy = 98.97%, duration = 7.184584(s)
Epoch [22/30], train loss = 0.004790, train accuracy = 99.87%, valid loss = 0.071976, valid accuracy = 98.87%, duration = 7.163523(s)
Epoch [23/30], train loss = 0.006341, train accuracy = 99.83%, valid loss = 0.054988, valid accuracy = 98.97%, duration = 7.103466(s)
Epoch [24/30], train loss = 0.004220, train accuracy = 99.88%, valid loss = 0.072304, valid accuracy = 98.82%, duration = 7.151508(s)
Epoch [25/30], train loss = 0.004665, train accuracy = 99.89%, valid loss = 0.062733, valid accuracy = 99.02%, duration = 7.132829(s)
Epoch [26/30], train loss = 0.005493, train accuracy = 99.87%, valid loss = 0.074161, valid accuracy = 98.90%, duration = 7.277383(s)
Epoch [27/30], train loss = 0.003489, train accuracy = 99.91%, valid loss = 0.059226, valid accuracy = 99.13%, duration = 7.192769(s)
Epoch [28/30], train loss = 0.004707, train accuracy = 99.87%, valid loss = 0.091629, valid accuracy = 98.80%, duration = 7.212351(s)
Epoch [29/30], train loss = 0.006953, train accuracy = 99.84%, valid loss = 0.062968, valid accuracy = 98.98%, duration = 7.284219(s)
Epoch [30/30], train loss = 0.002547, train accuracy = 99.94%, valid loss = 0.059641, valid accuracy = 99.22%, duration = 7.196823(s)
Duration for train : 220.670189(s)
<<< Train Finished >>>
Test Accraucy : 99.14%
'''
