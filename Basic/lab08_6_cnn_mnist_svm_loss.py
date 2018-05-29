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

BOARD_PATH = "./board/lab08-6_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

C = 10.0

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
    Y_svm_target = tf.subtract(tf.multiply(Y_one_hot, 2.), 1., 'Y_svm_target')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
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
logits, W = linear(d1, NCLASS, with_W=True, name='FC_Linear')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    l2_norm = tf.reduce_sum(tf.square(W),axis=0, name = 'L2_norm')
    tmp =  tf.subtract(1., tf.multiply(logits, Y_svm_target))
    hinge_loss = tf.reduce_sum(tf.square(tf.maximum(tf.zeros_like(tmp),tmp)),axis=0, name = 'l2_hinge_loss')
    normal_loss = tf.reduce_mean(0.5*l2_norm+C*hinge_loss, name = 'loss')
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
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :],
                                                                   is_training:True,  keep_prob:0.7, learning_rate:u})
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

    ta = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, is_training:False, keep_prob:1.0})
    print("Test Accraucy : {:.2%}".format(ta / ntest))

'''
GTX 1080Ti
Epoch [ 1/30], train loss = 0.725152, train accuracy = 93.19%, valid loss = 0.162050, valid accuracy = 97.78%, duration = 8.551778(s)
Epoch [ 2/30], train loss = 0.223454, train accuracy = 97.38%, valid loss = 0.144314, valid accuracy = 98.30%, duration = 7.449821(s)
Epoch [ 3/30], train loss = 0.165718, train accuracy = 98.13%, valid loss = 0.111337, valid accuracy = 98.63%, duration = 7.475504(s)
Epoch [ 4/30], train loss = 0.122242, train accuracy = 98.68%, valid loss = 0.095516, valid accuracy = 98.87%, duration = 7.460023(s)
Epoch [ 5/30], train loss = 0.088208, train accuracy = 99.00%, valid loss = 0.110388, valid accuracy = 98.63%, duration = 7.441085(s)
Epoch [ 6/30], train loss = 0.069842, train accuracy = 99.25%, valid loss = 0.086397, valid accuracy = 98.78%, duration = 7.448368(s)
Epoch [ 7/30], train loss = 0.046515, train accuracy = 99.51%, valid loss = 0.104909, valid accuracy = 98.80%, duration = 7.453481(s)
Epoch [ 8/30], train loss = 0.037056, train accuracy = 99.63%, valid loss = 0.078740, valid accuracy = 99.10%, duration = 7.519116(s)
Epoch [ 9/30], train loss = 0.032606, train accuracy = 99.71%, valid loss = 0.090898, valid accuracy = 99.07%, duration = 7.493262(s)
Epoch [10/30], train loss = 0.024875, train accuracy = 99.80%, valid loss = 0.088370, valid accuracy = 99.02%, duration = 7.494019(s)
Epoch [11/30], train loss = 0.023508, train accuracy = 99.80%, valid loss = 0.098382, valid accuracy = 98.93%, duration = 7.461090(s)
Epoch [12/30], train loss = 0.017263, train accuracy = 99.87%, valid loss = 0.097401, valid accuracy = 99.08%, duration = 7.465534(s)
Epoch [13/30], train loss = 0.014597, train accuracy = 99.90%, valid loss = 0.101877, valid accuracy = 98.93%, duration = 7.469124(s)
Epoch [14/30], train loss = 0.012755, train accuracy = 99.93%, valid loss = 0.125269, valid accuracy = 99.00%, duration = 7.441227(s)
Epoch [15/30], train loss = 0.012069, train accuracy = 99.94%, valid loss = 0.090761, valid accuracy = 99.03%, duration = 7.452426(s)
Epoch [16/30], train loss = 0.008851, train accuracy = 99.96%, valid loss = 0.083626, valid accuracy = 99.08%, duration = 7.426422(s)
Epoch [17/30], train loss = 0.009496, train accuracy = 99.95%, valid loss = 0.090663, valid accuracy = 99.12%, duration = 7.413916(s)
Epoch [18/30], train loss = 0.007402, train accuracy = 99.97%, valid loss = 0.097692, valid accuracy = 99.10%, duration = 7.371123(s)
Epoch [19/30], train loss = 0.007737, train accuracy = 99.95%, valid loss = 0.078234, valid accuracy = 99.13%, duration = 7.430650(s)
Epoch [20/30], train loss = 0.005358, train accuracy = 99.98%, valid loss = 0.094876, valid accuracy = 99.00%, duration = 7.467032(s)
Epoch [21/30], train loss = 0.005634, train accuracy = 99.99%, valid loss = 0.085390, valid accuracy = 99.27%, duration = 7.444438(s)
Epoch [22/30], train loss = 0.005836, train accuracy = 99.98%, valid loss = 0.086016, valid accuracy = 99.12%, duration = 7.429972(s)
Epoch [23/30], train loss = 0.004377, train accuracy = 99.99%, valid loss = 0.093953, valid accuracy = 99.18%, duration = 7.432015(s)
Epoch [24/30], train loss = 0.004913, train accuracy = 99.98%, valid loss = 0.073723, valid accuracy = 99.17%, duration = 7.448395(s)
Epoch [25/30], train loss = 0.003509, train accuracy = 99.99%, valid loss = 0.077926, valid accuracy = 99.25%, duration = 7.466389(s)
Epoch [26/30], train loss = 0.003657, train accuracy = 99.99%, valid loss = 0.080333, valid accuracy = 99.17%, duration = 7.521119(s)
Epoch [27/30], train loss = 0.003482, train accuracy = 99.99%, valid loss = 0.068838, valid accuracy = 99.28%, duration = 7.519182(s)
Epoch [28/30], train loss = 0.003253, train accuracy = 100.00%, valid loss = 0.078058, valid accuracy = 99.15%, duration = 7.498806(s)
Epoch [29/30], train loss = 0.002785, train accuracy = 100.00%, valid loss = 0.088273, valid accuracy = 98.97%, duration = 7.554176(s)
Epoch [30/30], train loss = 0.002582, train accuracy = 99.99%, valid loss = 0.079762, valid accuracy = 99.17%, duration = 7.462539(s)
Duration for train : 227.588991(s)
<<< Train Finished >>>
Test Accraucy : 99.15%
'''
