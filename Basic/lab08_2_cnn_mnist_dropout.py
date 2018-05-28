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
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :], keep_prob:0.7})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step * BATCH_SIZE
        acc_per_epoch /= total_step * BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation, keep_prob:1.0})
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
Epoch [ 1/30], train loss = 40.018334, train accuracy = 64.50%, valid loss = 0.903355, valid accuracy = 71.22%, duration = 7.126951(s)
Epoch [ 2/30], train loss = 1.247690, train accuracy = 61.09%, valid loss = 0.933224, valid accuracy = 71.63%, duration = 6.267367(s)
Epoch [ 3/30], train loss = 1.382510, train accuracy = 55.52%, valid loss = 0.971781, valid accuracy = 64.45%, duration = 6.260686(s)
Epoch [ 4/30], train loss = 1.355900, train accuracy = 53.26%, valid loss = 0.880156, valid accuracy = 66.20%, duration = 6.101747(s)
Epoch [ 5/30], train loss = 1.122753, train accuracy = 63.31%, valid loss = 0.512139, valid accuracy = 85.20%, duration = 6.146805(s)
Epoch [ 6/30], train loss = 0.911623, train accuracy = 70.83%, valid loss = 0.433213, valid accuracy = 86.53%, duration = 6.171985(s)
Epoch [ 7/30], train loss = 0.717976, train accuracy = 79.36%, valid loss = 0.437966, valid accuracy = 91.37%, duration = 6.154975(s)
Epoch [ 8/30], train loss = 0.518241, train accuracy = 86.28%, valid loss = 0.240270, valid accuracy = 95.97%, duration = 6.179891(s)
Epoch [ 9/30], train loss = 0.447216, train accuracy = 88.60%, valid loss = 0.204682, valid accuracy = 96.38%, duration = 6.164421(s)
Epoch [10/30], train loss = 0.369246, train accuracy = 90.76%, valid loss = 0.204696, valid accuracy = 96.08%, duration = 6.160854(s)
Epoch [11/30], train loss = 0.271914, train accuracy = 93.47%, valid loss = 0.183535, valid accuracy = 97.03%, duration = 6.171123(s)
Epoch [12/30], train loss = 0.246633, train accuracy = 94.53%, valid loss = 0.171529, valid accuracy = 97.22%, duration = 6.166233(s)
Epoch [13/30], train loss = 0.205068, train accuracy = 95.22%, valid loss = 0.196080, valid accuracy = 96.55%, duration = 6.140027(s)
Epoch [14/30], train loss = 0.186448, train accuracy = 95.68%, valid loss = 0.183151, valid accuracy = 97.53%, duration = 6.144626(s)
Epoch [15/30], train loss = 0.159913, train accuracy = 95.87%, valid loss = 0.240679, valid accuracy = 97.52%, duration = 6.149046(s)
Epoch [16/30], train loss = 0.171676, train accuracy = 96.00%, valid loss = 0.183951, valid accuracy = 97.18%, duration = 6.150193(s)
Epoch [17/30], train loss = 0.160595, train accuracy = 96.12%, valid loss = 0.176261, valid accuracy = 97.35%, duration = 6.077607(s)
Epoch [18/30], train loss = 0.143506, train accuracy = 96.43%, valid loss = 0.143182, valid accuracy = 97.57%, duration = 6.162312(s)
Epoch [19/30], train loss = 0.147587, train accuracy = 96.46%, valid loss = 0.170134, valid accuracy = 97.33%, duration = 6.171379(s)
Epoch [20/30], train loss = 0.142680, train accuracy = 96.69%, valid loss = 0.151567, valid accuracy = 97.63%, duration = 6.074516(s)
Epoch [21/30], train loss = 0.139254, train accuracy = 97.16%, valid loss = 0.182141, valid accuracy = 97.37%, duration = 6.165867(s)
Epoch [22/30], train loss = 0.140985, train accuracy = 97.09%, valid loss = 0.215731, valid accuracy = 97.77%, duration = 6.174803(s)
Epoch [23/30], train loss = 0.132337, train accuracy = 97.16%, valid loss = 0.187081, valid accuracy = 98.22%, duration = 6.145131(s)
Epoch [24/30], train loss = 0.135612, train accuracy = 97.17%, valid loss = 0.191909, valid accuracy = 97.82%, duration = 6.171934(s)
Epoch [25/30], train loss = 0.119056, train accuracy = 97.50%, valid loss = 0.197600, valid accuracy = 98.05%, duration = 6.096891(s)
Epoch [26/30], train loss = 0.135310, train accuracy = 97.21%, valid loss = 0.197189, valid accuracy = 98.03%, duration = 6.082767(s)
Epoch [27/30], train loss = 0.115443, train accuracy = 97.53%, valid loss = 0.202906, valid accuracy = 97.88%, duration = 6.114164(s)
Epoch [28/30], train loss = 0.139967, train accuracy = 97.22%, valid loss = 0.175774, valid accuracy = 97.78%, duration = 6.114776(s)
Epoch [29/30], train loss = 0.118935, train accuracy = 97.48%, valid loss = 0.182493, valid accuracy = 98.08%, duration = 6.171460(s)
Epoch [30/30], train loss = 0.122882, train accuracy = 97.53%, valid loss = 0.266563, valid accuracy = 97.87%, duration = 6.166159(s)
Duration for train : 188.097642(s)
<<< Train Finished >>>
Test Accraucy : 97.77%
'''
