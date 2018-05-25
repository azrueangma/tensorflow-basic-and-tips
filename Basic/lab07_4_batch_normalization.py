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
h3 = relu_layer(h2, 64, keep_prob=keep_prob, is_training=is_training, name='Relu_Layer3')
logits = linear(h3, NCLASS, name='Linear_Layer')

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
Epoch [ 1/30], train loss = 0.216998, train accuracy = 93.63%, valid loss = 0.127238, valid accuracy = 95.98%, duration = 8.549717(s)
Epoch [ 2/30], train loss = 0.102663, train accuracy = 96.79%, valid loss = 0.091414, valid accuracy = 96.95%, duration = 8.588652(s)
Epoch [ 3/30], train loss = 0.077104, train accuracy = 97.55%, valid loss = 0.076894, valid accuracy = 97.43%, duration = 8.425400(s)
Epoch [ 4/30], train loss = 0.057248, train accuracy = 98.23%, valid loss = 0.088217, valid accuracy = 97.18%, duration = 8.381147(s)
Epoch [ 5/30], train loss = 0.049149, train accuracy = 98.41%, valid loss = 0.068583, valid accuracy = 97.97%, duration = 8.427155(s)
Epoch [ 6/30], train loss = 0.041829, train accuracy = 98.63%, valid loss = 0.064420, valid accuracy = 98.03%, duration = 8.339999(s)
Epoch [ 7/30], train loss = 0.034687, train accuracy = 98.87%, valid loss = 0.065347, valid accuracy = 98.02%, duration = 8.442866(s)
Epoch [ 8/30], train loss = 0.031577, train accuracy = 98.93%, valid loss = 0.072057, valid accuracy = 97.63%, duration = 8.346914(s)
Epoch [ 9/30], train loss = 0.028480, train accuracy = 99.09%, valid loss = 0.060715, valid accuracy = 98.15%, duration = 8.341565(s)
Epoch [10/30], train loss = 0.023569, train accuracy = 99.27%, valid loss = 0.068371, valid accuracy = 97.90%, duration = 8.403822(s)
Epoch [11/30], train loss = 0.022885, train accuracy = 99.25%, valid loss = 0.069122, valid accuracy = 98.12%, duration = 8.368143(s)
Epoch [12/30], train loss = 0.020370, train accuracy = 99.32%, valid loss = 0.060624, valid accuracy = 98.25%, duration = 8.351002(s)
Epoch [13/30], train loss = 0.019371, train accuracy = 99.38%, valid loss = 0.071092, valid accuracy = 97.97%, duration = 8.369540(s)
Epoch [14/30], train loss = 0.017284, train accuracy = 99.42%, valid loss = 0.067235, valid accuracy = 98.18%, duration = 8.387190(s)
Epoch [15/30], train loss = 0.015492, train accuracy = 99.47%, valid loss = 0.059842, valid accuracy = 98.37%, duration = 8.398807(s)
Epoch [16/30], train loss = 0.014686, train accuracy = 99.51%, valid loss = 0.064421, valid accuracy = 98.35%, duration = 8.194739(s)
Epoch [17/30], train loss = 0.014650, train accuracy = 99.47%, valid loss = 0.062607, valid accuracy = 98.18%, duration = 8.800863(s)
Epoch [18/30], train loss = 0.012660, train accuracy = 99.56%, valid loss = 0.074689, valid accuracy = 98.05%, duration = 8.383959(s)
Epoch [19/30], train loss = 0.013479, train accuracy = 99.55%, valid loss = 0.056186, valid accuracy = 98.20%, duration = 8.293366(s)
Epoch [20/30], train loss = 0.012555, train accuracy = 99.58%, valid loss = 0.074992, valid accuracy = 98.08%, duration = 8.122805(s)
Epoch [21/30], train loss = 0.011329, train accuracy = 99.64%, valid loss = 0.058429, valid accuracy = 98.47%, duration = 8.028875(s)
Epoch [22/30], train loss = 0.013053, train accuracy = 99.55%, valid loss = 0.071587, valid accuracy = 98.17%, duration = 8.066233(s)
Epoch [23/30], train loss = 0.009771, train accuracy = 99.65%, valid loss = 0.060313, valid accuracy = 98.37%, duration = 8.458525(s)
Epoch [24/30], train loss = 0.009857, train accuracy = 99.67%, valid loss = 0.069763, valid accuracy = 98.20%, duration = 8.144391(s)
Epoch [25/30], train loss = 0.010672, train accuracy = 99.64%, valid loss = 0.060760, valid accuracy = 98.42%, duration = 8.309526(s)
Epoch [26/30], train loss = 0.009471, train accuracy = 99.69%, valid loss = 0.075763, valid accuracy = 98.20%, duration = 8.410730(s)
Epoch [27/30], train loss = 0.009783, train accuracy = 99.68%, valid loss = 0.060813, valid accuracy = 98.40%, duration = 8.514108(s)
Epoch [28/30], train loss = 0.008711, train accuracy = 99.73%, valid loss = 0.071098, valid accuracy = 98.32%, duration = 8.909684(s)
Epoch [29/30], train loss = 0.007909, train accuracy = 99.73%, valid loss = 0.075390, valid accuracy = 98.13%, duration = 9.105400(s)
Epoch [30/30], train loss = 0.008867, train accuracy = 99.73%, valid loss = 0.075731, valid accuracy = 98.15%, duration = 9.372928(s)
Duration for train : 255.382644(s)
<<< Train Finished >>>
Test Accraucy : 98.00%
'''
