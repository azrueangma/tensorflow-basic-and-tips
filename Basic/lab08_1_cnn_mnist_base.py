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

f1 = relu_layer(tf.reshape(p3, [-1, 4 * 4 * 128]), 1024, name='FC_Relu')
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
Epoch [ 1/30], train loss = 67.045388, train accuracy = 90.62%, valid loss = 1.202753, valid accuracy = 95.50%, duration = 249.514007(s)
Epoch [ 2/30], train loss = 0.822756, train accuracy = 95.52%, valid loss = 0.449453, valid accuracy = 95.27%, duration = 249.551120(s)
Epoch [ 3/30], train loss = 0.313919, train accuracy = 96.19%, valid loss = 0.252502, valid accuracy = 95.83%, duration = 248.520499(s)
Epoch [ 4/30], train loss = 0.195801, train accuracy = 96.66%, valid loss = 0.232173, valid accuracy = 95.98%, duration = 248.376500(s)
Epoch [ 5/30], train loss = 0.181621, train accuracy = 96.89%, valid loss = 0.319832, valid accuracy = 94.92%, duration = 247.646465(s)
Epoch [ 6/30], train loss = 0.139641, train accuracy = 97.28%, valid loss = 0.136535, valid accuracy = 97.08%, duration = 247.441017(s)
Epoch [ 7/30], train loss = 0.112595, train accuracy = 97.80%, valid loss = 0.192330, valid accuracy = 96.78%, duration = 247.555870(s)
Epoch [ 8/30], train loss = 0.121314, train accuracy = 97.83%, valid loss = 0.164857, valid accuracy = 97.08%, duration = 247.306219(s)
Epoch [ 9/30], train loss = 0.099863, train accuracy = 98.17%, valid loss = 0.170422, valid accuracy = 97.55%, duration = 247.089334(s)
Epoch [10/30], train loss = 0.088848, train accuracy = 98.46%, valid loss = 0.152632, valid accuracy = 97.68%, duration = 246.965926(s)
Epoch [11/30], train loss = 0.075586, train accuracy = 98.67%, valid loss = 0.226657, valid accuracy = 97.28%, duration = 247.027399(s)
Epoch [12/30], train loss = 0.072054, train accuracy = 98.73%, valid loss = 0.127366, valid accuracy = 98.05%, duration = 291.256055(s)
Epoch [13/30], train loss = 0.062148, train accuracy = 98.85%, valid loss = 0.175140, valid accuracy = 97.87%, duration = 385.382388(s)
Epoch [14/30], train loss = 0.068105, train accuracy = 98.92%, valid loss = 0.214626, valid accuracy = 97.65%, duration = 328.833815(s)
Epoch [15/30], train loss = 0.065326, train accuracy = 98.89%, valid loss = 0.340350, valid accuracy = 97.28%, duration = 250.123869(s)
Epoch [16/30], train loss = 0.061755, train accuracy = 99.00%, valid loss = 0.164236, valid accuracy = 97.68%, duration = 250.477169(s)
Epoch [17/30], train loss = 0.056983, train accuracy = 99.01%, valid loss = 0.287542, valid accuracy = 97.90%, duration = 247.945096(s)
Epoch [18/30], train loss = 0.061127, train accuracy = 99.02%, valid loss = 0.189298, valid accuracy = 98.02%, duration = 249.472947(s)
Epoch [19/30], train loss = 0.050025, train accuracy = 99.15%, valid loss = 0.234700, valid accuracy = 98.07%, duration = 247.998653(s)
Epoch [20/30], train loss = 0.059264, train accuracy = 99.04%, valid loss = 0.220676, valid accuracy = 97.75%, duration = 248.523440(s)
Epoch [21/30], train loss = 0.057702, train accuracy = 99.08%, valid loss = 0.196775, valid accuracy = 97.35%, duration = 250.072631(s)
Epoch [22/30], train loss = 0.049007, train accuracy = 99.16%, valid loss = 0.203669, valid accuracy = 98.20%, duration = 247.802732(s)
Epoch [23/30], train loss = 0.051194, train accuracy = 99.16%, valid loss = 0.256138, valid accuracy = 97.50%, duration = 248.157494(s)
Epoch [24/30], train loss = 0.062960, train accuracy = 99.03%, valid loss = 0.222745, valid accuracy = 97.97%, duration = 248.042849(s)
Epoch [25/30], train loss = 0.054461, train accuracy = 99.21%, valid loss = 0.322918, valid accuracy = 97.98%, duration = 248.632697(s)
Epoch [26/30], train loss = 0.048962, train accuracy = 99.21%, valid loss = 0.271001, valid accuracy = 98.43%, duration = 247.623388(s)
Epoch [27/30], train loss = 0.039637, train accuracy = 99.31%, valid loss = 0.613616, valid accuracy = 97.20%, duration = 247.512629(s)
Epoch [28/30], train loss = 0.047772, train accuracy = 99.23%, valid loss = 0.201451, valid accuracy = 97.82%, duration = 248.155890(s)
Epoch [29/30], train loss = 0.053821, train accuracy = 99.28%, valid loss = 0.266388, valid accuracy = 97.90%, duration = 248.246002(s)
Epoch [30/30], train loss = 0.048067, train accuracy = 99.37%, valid loss = 0.355632, valid accuracy = 98.02%, duration = 247.995473(s)
Duration for train : 7940.431499(s)
<<< Train Finished >>>
Test Accraucy : 98.01%
'''
