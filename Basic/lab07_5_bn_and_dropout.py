#-*- coding: utf-8 -*-
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import shutil
import time

import load_data

x_train, x_validation, x_test, y_train, y_validation, y_test=load_data.load_mnist('./data/mnist/', seed=0, as_image=False, scaling=True)

BOARD_PATH = "./board/lab07-5_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30
KEEP_PROB = 0.7

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)


def l1_loss (tensor_op, name='l1_loss'):
    output = tf.reduce_sum(tf.abs(tensor_op), name=name)
    return output


def l2_loss (tensor_op, name='l2_loss'):
    output = tf.reduce_sum(tf.square(tensor_op), name=name)/2
    return output


def linear(tensor_op,
           output_dim,
           weight_decay=None,
           regularizer=None,
           with_W=False,
           name='linear'):

    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)*weight_decay
            elif regularizer == 'l2':
                wd = l2_loss(W)*weight_decay
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


def relu_layer(
        tensor_op,
        output_dim,
        weight_decay=None,
        regularizer=None,
        keep_prob=1.0,
        is_training=False,
        with_W=False,
        name='relu_layer'):

    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name = 'pre_op')
        bn = tf.contrib.layers.batch_norm(pre_activation,
                                          is_training=is_training,
                                          updates_collections=None)
        h = tf.nn.relu(bn, name='relu_op')
        dr = tf.nn.dropout(h, keep_prob=keep_prob, name='dropout_op')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)*weight_decay
            elif regularizer == 'l2':
                wd = l2_loss(W)*weight_decay
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return dr, W
        else:
            return dr


tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape = [None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    is_training = tf.placeholder(tf.bool, name = 'is_training')

h1 = relu_layer(X, 256, keep_prob=keep_prob, is_training=is_training, name='Relu_Layer1')
h2 = relu_layer(h1, 128, keep_prob=keep_prob, is_training=is_training, name='Relu_Layer2')
logits = linear(h2, NCLASS, name='Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss+tf.reduce_sum(weight_decay_loss)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optim = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

with tf.variable_scope("Prediction"):
    predict = tf.argmax(hypothesis, axis=1)

with tf.variable_scope("Accuracy"):
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis=1)), tf.float32))

with tf.variable_scope("Summary"):
    avg_train_loss = tf.placeholder(tf.float32)
    loss_train_avg  = tf.summary.scalar('avg_train_loss', avg_train_loss)
    avg_train_acc = tf.placeholder(tf.float32)
    acc_train_avg = tf.summary.scalar('avg_train_acc', avg_train_acc)
    avg_validation_loss = tf.placeholder(tf.float32)
    loss_validation_avg = tf.summary.scalar('avg_validation_loss', avg_validation_loss)
    avg_validation_acc = tf.placeholder(tf.float32)
    acc_validation_avg = tf.summary.scalar('avg_validation_acc', avg_validation_acc)
    merged = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
total_step = int(ntrain/BATCH_SIZE)
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
            s = BATCH_SIZE*step
            t = BATCH_SIZE*(step+1)
            a, l, _ = sess.run([accuracy, loss, optim],
                               feed_dict={X:x_train[mask[s:t],:], Y:y_train[mask[s:t],:], keep_prob:KEEP_PROB, is_training:True})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step*BATCH_SIZE
        acc_per_epoch /= total_step*BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X:x_validation, Y:y_validation, keep_prob:1.0, is_training:False})
        epoch_valid_acc = va / len(x_validation)
        epoch_valid_loss = vl / len(x_validation)

        s = sess.run(merged, feed_dict = {avg_train_loss:loss_per_epoch, avg_train_acc:acc_per_epoch,
                                          avg_validation_loss:epoch_valid_loss, avg_validation_acc:epoch_valid_acc})
        writer.add_summary(s, global_step = epoch)

        if (epoch+1) % 1 == 0:

            print("Epoch [{:2d}/{:2d}], train loss = {:.6f}, train accuracy = {:.2%}, "
                  "valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)"
                  .format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

    ta = sess.run(accuracy, feed_dict = {X:x_test, Y:y_test, keep_prob:1.0, is_training:False})
    print("Test Accraucy : {:.2%}".format(ta/ntest))

'''
Epoch [ 1/30], train loss = 0.348572, train accuracy = 89.50%, valid loss = 0.130289, valid accuracy = 95.90%, duration = 10.024309(s)
Epoch [ 2/30], train loss = 0.191999, train accuracy = 94.11%, valid loss = 0.103907, valid accuracy = 96.88%, duration = 9.751308(s)
Epoch [ 3/30], train loss = 0.155435, train accuracy = 95.22%, valid loss = 0.082427, valid accuracy = 97.45%, duration = 9.731394(s)
Epoch [ 4/30], train loss = 0.133997, train accuracy = 95.90%, valid loss = 0.080213, valid accuracy = 97.72%, duration = 9.792622(s)
Epoch [ 5/30], train loss = 0.117011, train accuracy = 96.38%, valid loss = 0.068831, valid accuracy = 97.65%, duration = 9.789911(s)
Epoch [ 6/30], train loss = 0.107588, train accuracy = 96.66%, valid loss = 0.058784, valid accuracy = 98.05%, duration = 9.771992(s)
Epoch [ 7/30], train loss = 0.099870, train accuracy = 96.83%, valid loss = 0.063038, valid accuracy = 98.00%, duration = 9.794094(s)
Epoch [ 8/30], train loss = 0.092155, train accuracy = 97.05%, valid loss = 0.054667, valid accuracy = 98.28%, duration = 10.520486(s)
Epoch [ 9/30], train loss = 0.084174, train accuracy = 97.31%, valid loss = 0.052229, valid accuracy = 98.40%, duration = 9.858578(s)
Epoch [10/30], train loss = 0.082588, train accuracy = 97.36%, valid loss = 0.052077, valid accuracy = 98.27%, duration = 9.766160(s)
Epoch [11/30], train loss = 0.077668, train accuracy = 97.49%, valid loss = 0.054054, valid accuracy = 98.37%, duration = 9.776694(s)
Epoch [12/30], train loss = 0.075046, train accuracy = 97.63%, valid loss = 0.058035, valid accuracy = 98.17%, duration = 9.831264(s)
Epoch [13/30], train loss = 0.070593, train accuracy = 97.68%, valid loss = 0.055410, valid accuracy = 98.18%, duration = 9.779766(s)
Epoch [14/30], train loss = 0.066256, train accuracy = 97.75%, valid loss = 0.051336, valid accuracy = 98.47%, duration = 9.866138(s)
Epoch [15/30], train loss = 0.064652, train accuracy = 97.93%, valid loss = 0.051548, valid accuracy = 98.55%, duration = 9.831883(s)
Epoch [16/30], train loss = 0.063774, train accuracy = 98.02%, valid loss = 0.049933, valid accuracy = 98.40%, duration = 9.756706(s)
Epoch [17/30], train loss = 0.060125, train accuracy = 98.04%, valid loss = 0.055815, valid accuracy = 98.40%, duration = 9.759060(s)
Epoch [18/30], train loss = 0.059056, train accuracy = 98.06%, valid loss = 0.049852, valid accuracy = 98.57%, duration = 9.766696(s)
Epoch [19/30], train loss = 0.058953, train accuracy = 98.05%, valid loss = 0.050649, valid accuracy = 98.48%, duration = 9.812998(s)
Epoch [20/30], train loss = 0.055375, train accuracy = 98.19%, valid loss = 0.051737, valid accuracy = 98.47%, duration = 9.729388(s)
Epoch [21/30], train loss = 0.054094, train accuracy = 98.24%, valid loss = 0.054430, valid accuracy = 98.33%, duration = 9.725099(s)
Epoch [22/30], train loss = 0.052675, train accuracy = 98.22%, valid loss = 0.048933, valid accuracy = 98.52%, duration = 9.852203(s)
Epoch [23/30], train loss = 0.051002, train accuracy = 98.30%, valid loss = 0.051918, valid accuracy = 98.48%, duration = 9.814731(s)
Epoch [24/30], train loss = 0.049756, train accuracy = 98.41%, valid loss = 0.050729, valid accuracy = 98.50%, duration = 9.755710(s)
Epoch [25/30], train loss = 0.051244, train accuracy = 98.31%, valid loss = 0.047559, valid accuracy = 98.58%, duration = 9.869120(s)
Epoch [26/30], train loss = 0.047058, train accuracy = 98.44%, valid loss = 0.054023, valid accuracy = 98.27%, duration = 9.774142(s)
Epoch [27/30], train loss = 0.048537, train accuracy = 98.32%, valid loss = 0.053031, valid accuracy = 98.68%, duration = 9.796319(s)
Epoch [28/30], train loss = 0.044357, train accuracy = 98.52%, valid loss = 0.047426, valid accuracy = 98.62%, duration = 9.942471(s)
Epoch [29/30], train loss = 0.046671, train accuracy = 98.47%, valid loss = 0.051829, valid accuracy = 98.52%, duration = 10.140459(s)
Epoch [30/30], train loss = 0.042361, train accuracy = 98.59%, valid loss = 0.055782, valid accuracy = 98.58%, duration = 9.937024(s)
Duration for train : 298.084331(s)
<<< Train Finished >>>
Test Accraucy : 98.42%
'''
