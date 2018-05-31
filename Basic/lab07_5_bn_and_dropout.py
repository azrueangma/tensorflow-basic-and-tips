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
ALPHA = 0

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


def linear(tensor_op, output_dim, weight_decay = False, regularizer = None, with_W = False, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name = 'h')

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


def relu_layer(tensor_op, output_dim, weight_decay=True, regularizer=None,
               keep_prob=1.0, is_training=False, with_W=False, name='relu_layer'):

    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='pre_op')
        bn = tf.contrib.layers.batch_norm(pre_activation, is_training=is_training,updates_collections=None)
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


tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype=tf.float32, name='X')
    Y = tf.placeholder(shape = [None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    is_training = tf.placeholder(tf.bool, name = 'is_training')

h1 = relu_layer(tensor_op=X, output_dim=256, keep_prob=keep_prob, is_training=is_training, name='Relu_Layer1')
h2 = relu_layer(tensor_op=h1, output_dim=128, keep_prob=keep_prob,is_training=is_training, name='Relu_Layer2')
logits = linear(tensor_op=h2, output_dim=NCLASS, name='Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    normal_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss+ALPHA*tf.reduce_mean(weight_decay_loss)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

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
                               feed_dict={X:x_train[mask[s:t],:], Y:y_train[mask[s:t],:], is_training:True, keep_prob:0.7})
            acc_per_epoch += a
            loss_per_epoch += l
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        acc_per_epoch /= total_step*BATCH_SIZE
        loss_per_epoch /= total_step

        va, vl = sess.run([accuracy, normal_loss], feed_dict={X:x_validation, Y:y_validation, is_training:False, keep_prob:1.0})
        epoch_valid_acc = va / len(x_validation)
        epoch_valid_loss = vl

        s = sess.run(merged, feed_dict={avg_train_loss:loss_per_epoch, avg_train_acc:acc_per_epoch,
                                        avg_validation_loss:epoch_valid_loss, avg_validation_acc:epoch_valid_acc})
        writer.add_summary(s, global_step=epoch)

        if (epoch+1) % 1 == 0:

            print("Epoch [{:2d}/{:2d}], train loss = {:.6f}, train accuracy = {:.2%}, "
                  "valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)"
                  .format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

    ta = sess.run(accuracy, feed_dict = {X:x_test, Y:y_test, is_training:False, keep_prob:1.0})
    print("Test Accraucy : {:.2%}".format(ta/ntest))

'''
Epoch [ 1/30], train loss = 0.346886, train accuracy = 89.58%, valid loss = 0.134348, valid accuracy = 96.05%, duration = 9.245941(s)
Epoch [ 2/30], train loss = 0.189186, train accuracy = 94.20%, valid loss = 0.117465, valid accuracy = 96.28%, duration = 9.051481(s)
Epoch [ 3/30], train loss = 0.157482, train accuracy = 95.13%, valid loss = 0.081247, valid accuracy = 97.28%, duration = 9.129317(s)
Epoch [ 4/30], train loss = 0.130318, train accuracy = 95.95%, valid loss = 0.076543, valid accuracy = 97.62%, duration = 9.013022(s)
Epoch [ 5/30], train loss = 0.117092, train accuracy = 96.27%, valid loss = 0.068529, valid accuracy = 97.68%, duration = 8.997830(s)
Epoch [ 6/30], train loss = 0.108581, train accuracy = 96.62%, valid loss = 0.061245, valid accuracy = 97.97%, duration = 9.080328(s)
Epoch [ 7/30], train loss = 0.099612, train accuracy = 96.88%, valid loss = 0.065421, valid accuracy = 98.00%, duration = 9.016864(s)
Epoch [ 8/30], train loss = 0.093041, train accuracy = 97.11%, valid loss = 0.057976, valid accuracy = 98.10%, duration = 8.966382(s)
Epoch [ 9/30], train loss = 0.084259, train accuracy = 97.30%, valid loss = 0.059129, valid accuracy = 98.12%, duration = 8.933116(s)
Epoch [10/30], train loss = 0.084779, train accuracy = 97.26%, valid loss = 0.059970, valid accuracy = 98.10%, duration = 8.900164(s)
Epoch [11/30], train loss = 0.078818, train accuracy = 97.48%, valid loss = 0.058218, valid accuracy = 98.27%, duration = 8.911190(s)
Epoch [12/30], train loss = 0.074486, train accuracy = 97.62%, valid loss = 0.055585, valid accuracy = 98.20%, duration = 8.842525(s)
Epoch [13/30], train loss = 0.070306, train accuracy = 97.74%, valid loss = 0.059074, valid accuracy = 98.18%, duration = 8.904966(s)
Epoch [14/30], train loss = 0.067831, train accuracy = 97.78%, valid loss = 0.053069, valid accuracy = 98.30%, duration = 8.949672(s)
Epoch [15/30], train loss = 0.063580, train accuracy = 97.87%, valid loss = 0.053640, valid accuracy = 98.22%, duration = 8.985716(s)
Epoch [16/30], train loss = 0.060871, train accuracy = 98.02%, valid loss = 0.057441, valid accuracy = 98.20%, duration = 8.843795(s)
Epoch [17/30], train loss = 0.061104, train accuracy = 97.97%, valid loss = 0.055318, valid accuracy = 98.30%, duration = 8.883912(s)
Epoch [18/30], train loss = 0.058809, train accuracy = 97.98%, valid loss = 0.053511, valid accuracy = 98.53%, duration = 8.861478(s)
Epoch [19/30], train loss = 0.057874, train accuracy = 98.12%, valid loss = 0.055308, valid accuracy = 98.50%, duration = 8.920200(s)
Epoch [20/30], train loss = 0.057143, train accuracy = 98.07%, valid loss = 0.051436, valid accuracy = 98.45%, duration = 8.886987(s)
Epoch [21/30], train loss = 0.053058, train accuracy = 98.22%, valid loss = 0.055347, valid accuracy = 98.37%, duration = 8.897479(s)
Epoch [22/30], train loss = 0.054299, train accuracy = 98.22%, valid loss = 0.055196, valid accuracy = 98.28%, duration = 8.902227(s)
Epoch [23/30], train loss = 0.053134, train accuracy = 98.27%, valid loss = 0.053848, valid accuracy = 98.52%, duration = 8.915055(s)
Epoch [24/30], train loss = 0.048578, train accuracy = 98.35%, valid loss = 0.050030, valid accuracy = 98.50%, duration = 8.856457(s)
Epoch [25/30], train loss = 0.050245, train accuracy = 98.33%, valid loss = 0.052393, valid accuracy = 98.45%, duration = 8.880122(s)
Epoch [26/30], train loss = 0.047952, train accuracy = 98.40%, valid loss = 0.053984, valid accuracy = 98.38%, duration = 8.863753(s)
Epoch [27/30], train loss = 0.047036, train accuracy = 98.43%, valid loss = 0.052539, valid accuracy = 98.42%, duration = 8.927843(s)
Epoch [28/30], train loss = 0.046428, train accuracy = 98.45%, valid loss = 0.047997, valid accuracy = 98.43%, duration = 8.872632(s)
Epoch [29/30], train loss = 0.044024, train accuracy = 98.55%, valid loss = 0.050144, valid accuracy = 98.62%, duration = 8.919408(s)
Epoch [30/30], train loss = 0.044512, train accuracy = 98.50%, valid loss = 0.048336, valid accuracy = 98.60%, duration = 8.855113(s)
Duration for train : 270.625688(s)
<<< Train Finished >>>
Test Accraucy : 98.47%
'''
