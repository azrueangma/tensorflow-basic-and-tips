import tensorflow as tf
import numpy as np
import load_data
import shutil
import time
import os

x_train, x_validation, x_test, y_train, y_validation, y_test, VOCAB_SIZE \
    = load_data.load_spam_data(data_dir='./data', data_file='text_data.txt', seed=0)

BOARD_PATH = "./board/lab09-1_board"
BATCH_SIZE = 32

TOTAL_EPOCH = 30
INIT_LEARNING_RATE = 0.001
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_SIZE = 50
NCLASS = 2
NUM_RNN_UNITS = 10

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)


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


def rnn_layer(tensor_op, activiation=tf.nn.tanh, name="RNN_Layer"):
    with tf.variable_scope(name):
        cell = tf.contrib.rnn.BasicRNNCell(num_units=NUM_RNN_UNITS, activation=activiation)
        output, state = tf.nn.dynamic_rnn(cell, tensor_op, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return last


with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, MAX_SEQUENCE_LENGTH], dtype=tf.int32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

embedding_mat = tf.get_variable(name='embedding', shape=[VOCAB_SIZE, EMBEDDING_SIZE],
                                initializer=tf.random_normal_initializer(stddev=1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, X)

l1 = rnn_layer(embedding_output, name="RNN_Layer1")
h1 = linear(tensor_op=l1, output_dim=NCLASS, name='FCLayer1')
logits = tf.nn.softmax(h1)

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='loss')
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
            a, l, _ = sess.run([accuracy, loss, optim],
                               feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :],learning_rate:u})
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

    ta = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
    print("Test Accraucy : {:.2%}".format(ta / ntest))

'''
Epoch [ 1/30], train loss = 0.506828, train accuracy = 79.83%, valid loss = 0.483568, valid accuracy = 83.12%, duration = 0.497267(s)
Epoch [ 2/30], train loss = 0.460430, train accuracy = 85.18%, valid loss = 0.477168, valid accuracy = 84.02%, duration = 0.416871(s)
Epoch [ 3/30], train loss = 0.451578, train accuracy = 86.49%, valid loss = 0.473766, valid accuracy = 84.20%, duration = 0.426420(s)
Epoch [ 4/30], train loss = 0.444922, train accuracy = 87.11%, valid loss = 0.471934, valid accuracy = 84.20%, duration = 0.439923(s)
Epoch [ 5/30], train loss = 0.439899, train accuracy = 87.40%, valid loss = 0.469085, valid accuracy = 84.38%, duration = 0.407431(s)
Epoch [ 6/30], train loss = 0.427125, train accuracy = 88.74%, valid loss = 0.460623, valid accuracy = 84.74%, duration = 0.417472(s)
Epoch [ 7/30], train loss = 0.402201, train accuracy = 91.22%, valid loss = 0.398437, valid accuracy = 92.10%, duration = 0.408947(s)
Epoch [ 8/30], train loss = 0.378326, train accuracy = 93.72%, valid loss = 0.387205, valid accuracy = 92.64%, duration = 0.404641(s)
Epoch [ 9/30], train loss = 0.364824, train accuracy = 94.91%, valid loss = 0.376390, valid accuracy = 93.72%, duration = 0.411701(s)
Epoch [10/30], train loss = 0.354017, train accuracy = 96.26%, valid loss = 0.374264, valid accuracy = 93.90%, duration = 0.404886(s)
Epoch [11/30], train loss = 0.348353, train accuracy = 96.64%, valid loss = 0.377556, valid accuracy = 93.18%, duration = 0.404467(s)
Epoch [12/30], train loss = 0.344230, train accuracy = 97.00%, valid loss = 0.368380, valid accuracy = 94.25%, duration = 0.400636(s)
Epoch [13/30], train loss = 0.343163, train accuracy = 97.26%, valid loss = 0.364769, valid accuracy = 94.97%, duration = 0.403440(s)
Epoch [14/30], train loss = 0.342527, train accuracy = 97.16%, valid loss = 0.368059, valid accuracy = 94.25%, duration = 0.416292(s)
Epoch [15/30], train loss = 0.339391, train accuracy = 97.57%, valid loss = 0.363413, valid accuracy = 94.97%, duration = 0.403420(s)
Epoch [16/30], train loss = 0.337066, train accuracy = 97.75%, valid loss = 0.366150, valid accuracy = 94.43%, duration = 0.412583(s)
Epoch [17/30], train loss = 0.337944, train accuracy = 97.62%, valid loss = 0.364606, valid accuracy = 94.43%, duration = 0.406003(s)
Epoch [18/30], train loss = 0.336700, train accuracy = 97.75%, valid loss = 0.368186, valid accuracy = 94.25%, duration = 0.410535(s)
Epoch [19/30], train loss = 0.336802, train accuracy = 97.68%, valid loss = 0.365084, valid accuracy = 94.43%, duration = 0.408595(s)
Epoch [20/30], train loss = 0.335611, train accuracy = 97.88%, valid loss = 0.360426, valid accuracy = 95.15%, duration = 0.409721(s)
Epoch [21/30], train loss = 0.335633, train accuracy = 97.83%, valid loss = 0.363984, valid accuracy = 94.79%, duration = 0.415439(s)
Epoch [22/30], train loss = 0.335057, train accuracy = 97.88%, valid loss = 0.367151, valid accuracy = 94.43%, duration = 0.408589(s)
Epoch [23/30], train loss = 0.334278, train accuracy = 98.01%, valid loss = 0.365076, valid accuracy = 94.43%, duration = 0.417268(s)
Epoch [24/30], train loss = 0.336208, train accuracy = 97.78%, valid loss = 0.366341, valid accuracy = 94.43%, duration = 0.404090(s)
Epoch [25/30], train loss = 0.336192, train accuracy = 97.80%, valid loss = 0.362459, valid accuracy = 94.79%, duration = 0.403974(s)
Epoch [26/30], train loss = 0.333974, train accuracy = 98.01%, valid loss = 0.369727, valid accuracy = 93.72%, duration = 0.399994(s)
Epoch [27/30], train loss = 0.333986, train accuracy = 97.99%, valid loss = 0.365095, valid accuracy = 94.43%, duration = 0.407056(s)
Epoch [28/30], train loss = 0.333204, train accuracy = 98.06%, valid loss = 0.365140, valid accuracy = 94.43%, duration = 0.417083(s)
Epoch [29/30], train loss = 0.333100, train accuracy = 98.09%, valid loss = 0.372257, valid accuracy = 93.72%, duration = 0.410926(s)
Epoch [30/30], train loss = 0.332797, train accuracy = 98.09%, valid loss = 0.367372, valid accuracy = 94.25%, duration = 0.414142(s)
Duration for train : 12.727248(s)
<<< Train Finished >>>
Test Accraucy : 96.24%
'''