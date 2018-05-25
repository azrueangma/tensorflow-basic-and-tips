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

TOTAL_EPOCH = 20
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
logits = linear(h1, NCLASS, name='Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss+tf.reduce_mean(weight_decay_loss)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optim = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

with tf.variable_scope("Pred_and_Acc"):
    predict = tf.argmax(hypothesis, axis=1)
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
Epoch [ 1/20], train loss = 0.278197, train accuracy = 91.70%, valid loss = 0.146344, valid accuracy = 95.68%, duration = 5.984467(s)
Epoch [ 2/20], train loss = 0.162514, train accuracy = 95.05%, valid loss = 0.097284, valid accuracy = 97.07%, duration = 5.980847(s)
Epoch [ 3/20], train loss = 0.127874, train accuracy = 96.09%, valid loss = 0.087340, valid accuracy = 97.25%, duration = 5.915001(s)
Epoch [ 4/20], train loss = 0.110216, train accuracy = 96.62%, valid loss = 0.092016, valid accuracy = 97.13%, duration = 6.128223(s)
Epoch [ 5/20], train loss = 0.097769, train accuracy = 96.99%, valid loss = 0.074007, valid accuracy = 97.63%, duration = 6.118597(s)
Epoch [ 6/20], train loss = 0.087535, train accuracy = 97.26%, valid loss = 0.072045, valid accuracy = 97.80%, duration = 6.217004(s)
Epoch [ 7/20], train loss = 0.081147, train accuracy = 97.33%, valid loss = 0.073511, valid accuracy = 97.82%, duration = 6.195332(s)
Epoch [ 8/20], train loss = 0.073307, train accuracy = 97.72%, valid loss = 0.065179, valid accuracy = 98.02%, duration = 6.145204(s)
Epoch [ 9/20], train loss = 0.066916, train accuracy = 97.91%, valid loss = 0.067502, valid accuracy = 97.88%, duration = 6.086376(s)
Epoch [10/20], train loss = 0.062281, train accuracy = 98.02%, valid loss = 0.061561, valid accuracy = 98.17%, duration = 6.049290(s)
Epoch [11/20], train loss = 0.058211, train accuracy = 98.13%, valid loss = 0.063428, valid accuracy = 98.07%, duration = 5.959576(s)
Epoch [12/20], train loss = 0.056687, train accuracy = 98.15%, valid loss = 0.062513, valid accuracy = 98.08%, duration = 5.910606(s)
Epoch [13/20], train loss = 0.052707, train accuracy = 98.31%, valid loss = 0.060866, valid accuracy = 98.05%, duration = 5.948015(s)
Epoch [14/20], train loss = 0.050667, train accuracy = 98.36%, valid loss = 0.056812, valid accuracy = 98.48%, duration = 5.991836(s)
Epoch [15/20], train loss = 0.047504, train accuracy = 98.49%, valid loss = 0.058554, valid accuracy = 98.27%, duration = 5.973739(s)
Epoch [16/20], train loss = 0.045997, train accuracy = 98.51%, valid loss = 0.055948, valid accuracy = 98.38%, duration = 5.989984(s)
Epoch [17/20], train loss = 0.044333, train accuracy = 98.61%, valid loss = 0.060125, valid accuracy = 98.20%, duration = 5.916327(s)
Epoch [18/20], train loss = 0.041868, train accuracy = 98.56%, valid loss = 0.060557, valid accuracy = 98.32%, duration = 5.934472(s)
Epoch [19/20], train loss = 0.039762, train accuracy = 98.70%, valid loss = 0.059922, valid accuracy = 98.12%, duration = 6.026008(s)
Epoch [20/20], train loss = 0.038397, train accuracy = 98.74%, valid loss = 0.061457, valid accuracy = 98.40%, duration = 5.892190(s)
Duration for train : 121.589086(s)
<<< Train Finished >>>
Test Accraucy : 98.10%
'''