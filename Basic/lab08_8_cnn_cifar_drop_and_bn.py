# -*- coding: utf-8 -*-
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import shutil
import time

import load_data

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_cifar('./data/cifar/', seed=0, as_image=True, scaling=True)

BOARD_PATH = "./board/lab08-8_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30
ALPHA = 0
INIT_LEARNING_RATE = 0.001

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

image_width = np.size(x_train, 1)
image_height = np.size(x_train, 2)
n_channels = np.size(x_train, 3)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)


def l1_loss(tensor_op, name='l1_loss'):
    output = tf.reduce_sum(tf.abs(tensor_op), name=name)
    return output


def l2_loss(tensor_op, name='l2_loss'):
    output = tf.reduce_sum(tf.square(tensor_op), name=name) / 2
    return output


def linear(tensor_op, output_dim, weight_decay=False, regularizer=None, with_W=False, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)
            elif regularizer == 'l2':
                wd = l2_loss(W)
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


def relu_layer(tensor_op, output_dim, weight_decay=False, regularizer=None,
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
                wd = l1_loss(W)
            elif regularizer == 'l2':
                wd = l2_loss(W)
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return dr, W
        else:
            return dr


def to_flat(tensor_op, name):
    with tf.variable_scope(name):
        input_shape = tensor_op.get_shape().as_list()
        dim = np.prod(input_shape[1:])
        flat = tf.reshape(tensor_op, [-1, dim])
    return flat


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
    X = tf.placeholder(shape=[None, image_width, image_height, n_channels], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    is_training = tf.placeholder(tf.bool, name='is_training')

h1 = conv2d(X, 1, 1, [5, 5, n_channels, 32], name='Conv1')
b1 = bn_layer(h1, is_training, name='bn1')
p1 = max_pooling(b1, 2, 2, 2, 2, name='MaxPool1')
h2 = conv2d(p1, 1, 1, [5, 5, 32, 64], name='Conv2')
b2 = bn_layer(h2, is_training, name='bn2')
p2 = max_pooling(b2, 2, 2, 2, 2, name='MaxPool2')
h3 = conv2d(p2, 1, 1, [5, 5, 64, 128], name='Conv3')
b3 = bn_layer(h3, is_training, name='bn3')
p3 = max_pooling(b3, 2, 2, 2, 2, name='MaxPool3')

flat_op = to_flat(p3, name='flat_op')
f1 = relu_layer(flat_op, 1024, name='FC_Relu')
d1 = dropout_layer(f1, keep_prob=keep_prob, name='Dropout')
logits = linear(d1, NCLASS, name='FC_Linear')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot),
                                name='loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss + ALPHA*tf.reduce_sum(weight_decay_loss)
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
Epoch [ 1/30], train loss = 1.580174, train accuracy = 43.99%, valid loss = 1.530488, valid accuracy = 46.14%, duration = 8.674794(s)
Epoch [ 2/30], train loss = 1.229530, train accuracy = 55.72%, valid loss = 1.171480, valid accuracy = 58.18%, duration = 7.329366(s)
Epoch [ 3/30], train loss = 1.071020, train accuracy = 61.95%, valid loss = 1.142017, valid accuracy = 60.60%, duration = 7.152961(s)
Epoch [ 4/30], train loss = 0.960023, train accuracy = 65.91%, valid loss = 1.051308, valid accuracy = 63.54%, duration = 7.094958(s)
Epoch [ 5/30], train loss = 0.870375, train accuracy = 69.11%, valid loss = 1.054334, valid accuracy = 64.28%, duration = 7.084734(s)
Epoch [ 6/30], train loss = 0.794335, train accuracy = 71.71%, valid loss = 0.886955, valid accuracy = 68.96%, duration = 7.092709(s)
Epoch [ 7/30], train loss = 0.726832, train accuracy = 73.87%, valid loss = 0.913623, valid accuracy = 69.20%, duration = 7.088420(s)
Epoch [ 8/30], train loss = 0.655500, train accuracy = 76.45%, valid loss = 0.943504, valid accuracy = 67.92%, duration = 7.092557(s)
Epoch [ 9/30], train loss = 0.604250, train accuracy = 78.52%, valid loss = 0.898422, valid accuracy = 69.56%, duration = 7.089425(s)
Epoch [10/30], train loss = 0.548482, train accuracy = 80.33%, valid loss = 0.916037, valid accuracy = 70.14%, duration = 7.090857(s)
Epoch [11/30], train loss = 0.497211, train accuracy = 82.19%, valid loss = 0.965055, valid accuracy = 69.14%, duration = 7.080398(s)
Epoch [12/30], train loss = 0.457150, train accuracy = 83.49%, valid loss = 0.939719, valid accuracy = 70.54%, duration = 7.075083(s)
Epoch [13/30], train loss = 0.413688, train accuracy = 84.90%, valid loss = 1.007263, valid accuracy = 70.58%, duration = 7.084975(s)
Epoch [14/30], train loss = 0.380958, train accuracy = 86.28%, valid loss = 1.020491, valid accuracy = 69.94%, duration = 7.077705(s)
Epoch [15/30], train loss = 0.344165, train accuracy = 87.56%, valid loss = 0.996769, valid accuracy = 70.86%, duration = 7.098861(s)
Epoch [16/30], train loss = 0.312732, train accuracy = 88.53%, valid loss = 1.024584, valid accuracy = 71.64%, duration = 7.106520(s)
Epoch [17/30], train loss = 0.286263, train accuracy = 89.54%, valid loss = 1.067168, valid accuracy = 71.54%, duration = 7.101044(s)
Epoch [18/30], train loss = 0.264663, train accuracy = 90.35%, valid loss = 1.106063, valid accuracy = 70.86%, duration = 7.101669(s)
Epoch [19/30], train loss = 0.243829, train accuracy = 91.15%, valid loss = 1.165094, valid accuracy = 70.60%, duration = 7.117390(s)
Epoch [20/30], train loss = 0.227301, train accuracy = 91.63%, valid loss = 1.241879, valid accuracy = 71.28%, duration = 7.080816(s)
Epoch [21/30], train loss = 0.212261, train accuracy = 92.34%, valid loss = 1.244280, valid accuracy = 71.48%, duration = 7.093775(s)
Epoch [22/30], train loss = 0.198753, train accuracy = 92.74%, valid loss = 1.250269, valid accuracy = 71.20%, duration = 7.090009(s)
Epoch [23/30], train loss = 0.182673, train accuracy = 93.37%, valid loss = 1.371458, valid accuracy = 70.50%, duration = 7.102429(s)
Epoch [24/30], train loss = 0.171322, train accuracy = 93.72%, valid loss = 1.388679, valid accuracy = 70.62%, duration = 7.098127(s)
Epoch [25/30], train loss = 0.160269, train accuracy = 94.23%, valid loss = 1.363257, valid accuracy = 70.80%, duration = 7.109059(s)
Epoch [26/30], train loss = 0.151804, train accuracy = 94.52%, valid loss = 1.514375, valid accuracy = 69.98%, duration = 7.112707(s)
Epoch [27/30], train loss = 0.141785, train accuracy = 94.94%, valid loss = 1.385524, valid accuracy = 71.14%, duration = 7.103462(s)
Epoch [28/30], train loss = 0.133530, train accuracy = 95.08%, valid loss = 1.490424, valid accuracy = 70.94%, duration = 7.118020(s)
Epoch [29/30], train loss = 0.127907, train accuracy = 95.35%, valid loss = 1.514643, valid accuracy = 71.48%, duration = 7.106353(s)
Epoch [30/30], train loss = 0.119459, train accuracy = 95.81%, valid loss = 1.447257, valid accuracy = 71.32%, duration = 7.390690(s)
Duration for train : 220.281687(s)
<<< Train Finished >>>
Test Accraucy : 70.50%
'''
