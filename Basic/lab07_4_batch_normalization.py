#-*- coding: utf-8 -*-
from __future__ import division

import tensorflow as tf
import numpy as np
import os
import shutil
import time

import load_data

x_train, x_validation, x_test, y_train, y_validation, y_test=load_data.load_mnist('./data/mnist/', seed=0, as_image=False, scaling=True)

BOARD_PATH = "./board/lab07-4_board"
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
    is_training = tf.placeholder(tf.bool, name = 'is_training')

h1 = relu_layer(tensor_op=X, output_dim=256, is_training=is_training, name='Relu_Layer1')
h2 = relu_layer(tensor_op=h1, output_dim=128, is_training=is_training, name='Relu_Layer2')
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
                               feed_dict={X:x_train[mask[s:t],:], Y:y_train[mask[s:t],:], is_training:True})
            acc_per_epoch += a
            loss_per_epoch += l
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        acc_per_epoch /= total_step*BATCH_SIZE
        loss_per_epoch /= total_step

        va, vl = sess.run([accuracy, normal_loss], feed_dict={X:x_validation, Y:y_validation, is_training:False})
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

    ta = sess.run(accuracy, feed_dict = {X:x_test, Y:y_test, is_training:False})
    print("Test Accraucy : {:.2%}".format(ta/ntest))

'''
Epoch [ 1/30], train loss = 0.219218, train accuracy = 93.51%, valid loss = 0.127199, valid accuracy = 96.23%, duration = 8.500624(s)
Epoch [ 2/30], train loss = 0.103015, train accuracy = 96.76%, valid loss = 0.112798, valid accuracy = 96.13%, duration = 8.331897(s)
Epoch [ 3/30], train loss = 0.075354, train accuracy = 97.71%, valid loss = 0.077470, valid accuracy = 97.43%, duration = 8.270750(s)
Epoch [ 4/30], train loss = 0.058328, train accuracy = 98.15%, valid loss = 0.092573, valid accuracy = 97.15%, duration = 8.285274(s)
Epoch [ 5/30], train loss = 0.047674, train accuracy = 98.46%, valid loss = 0.055822, valid accuracy = 98.23%, duration = 8.246089(s)
Epoch [ 6/30], train loss = 0.040874, train accuracy = 98.64%, valid loss = 0.071556, valid accuracy = 97.75%, duration = 8.267615(s)
Epoch [ 7/30], train loss = 0.035974, train accuracy = 98.81%, valid loss = 0.064497, valid accuracy = 97.95%, duration = 8.264611(s)
Epoch [ 8/30], train loss = 0.030159, train accuracy = 98.98%, valid loss = 0.068492, valid accuracy = 97.85%, duration = 8.290669(s)
Epoch [ 9/30], train loss = 0.028771, train accuracy = 99.03%, valid loss = 0.062606, valid accuracy = 98.20%, duration = 8.264342(s)
Epoch [10/30], train loss = 0.023332, train accuracy = 99.23%, valid loss = 0.053647, valid accuracy = 98.25%, duration = 8.342478(s)
Epoch [11/30], train loss = 0.021938, train accuracy = 99.29%, valid loss = 0.081393, valid accuracy = 97.53%, duration = 8.273199(s)
Epoch [12/30], train loss = 0.021671, train accuracy = 99.31%, valid loss = 0.051713, valid accuracy = 98.48%, duration = 8.229278(s)
Epoch [13/30], train loss = 0.019166, train accuracy = 99.37%, valid loss = 0.071326, valid accuracy = 98.15%, duration = 8.345954(s)
Epoch [14/30], train loss = 0.016377, train accuracy = 99.47%, valid loss = 0.055527, valid accuracy = 98.45%, duration = 8.333055(s)
Epoch [15/30], train loss = 0.016686, train accuracy = 99.46%, valid loss = 0.066580, valid accuracy = 98.03%, duration = 8.264043(s)
Epoch [16/30], train loss = 0.015635, train accuracy = 99.49%, valid loss = 0.065975, valid accuracy = 98.30%, duration = 8.225630(s)
Epoch [17/30], train loss = 0.016453, train accuracy = 99.44%, valid loss = 0.072315, valid accuracy = 97.85%, duration = 8.187623(s)
Epoch [18/30], train loss = 0.013620, train accuracy = 99.56%, valid loss = 0.071284, valid accuracy = 98.13%, duration = 8.198115(s)
Epoch [19/30], train loss = 0.012837, train accuracy = 99.58%, valid loss = 0.061077, valid accuracy = 98.48%, duration = 8.248388(s)
Epoch [20/30], train loss = 0.013113, train accuracy = 99.56%, valid loss = 0.072218, valid accuracy = 98.25%, duration = 8.276775(s)
Epoch [21/30], train loss = 0.010798, train accuracy = 99.64%, valid loss = 0.066171, valid accuracy = 98.23%, duration = 8.285307(s)
Epoch [22/30], train loss = 0.013672, train accuracy = 99.54%, valid loss = 0.074650, valid accuracy = 98.05%, duration = 8.304294(s)
Epoch [23/30], train loss = 0.009283, train accuracy = 99.71%, valid loss = 0.071097, valid accuracy = 98.27%, duration = 8.252367(s)
Epoch [24/30], train loss = 0.010358, train accuracy = 99.66%, valid loss = 0.084560, valid accuracy = 97.87%, duration = 8.332365(s)
Epoch [25/30], train loss = 0.012134, train accuracy = 99.58%, valid loss = 0.071764, valid accuracy = 98.25%, duration = 8.241732(s)
Epoch [26/30], train loss = 0.009326, train accuracy = 99.69%, valid loss = 0.062898, valid accuracy = 98.35%, duration = 8.263137(s)
Epoch [27/30], train loss = 0.008520, train accuracy = 99.71%, valid loss = 0.069301, valid accuracy = 98.37%, duration = 8.289776(s)
Epoch [28/30], train loss = 0.009177, train accuracy = 99.67%, valid loss = 0.091823, valid accuracy = 97.78%, duration = 8.352524(s)
Epoch [29/30], train loss = 0.008897, train accuracy = 99.72%, valid loss = 0.073324, valid accuracy = 98.10%, duration = 8.284049(s)
Epoch [30/30], train loss = 0.007555, train accuracy = 99.75%, valid loss = 0.071557, valid accuracy = 98.32%, duration = 8.296070(s)
Duration for train : 250.553248(s)
<<< Train Finished >>>
Test Accraucy : 98.28%
'''
