import tensorflow as tf
import numpy as np
import load_data
import shutil
import time
import os

x_train, x_validation, x_test, y_train, y_validation, y_test, VOCAB_SIZE \
    = load_data.load_spam_data(data_dir='./data', data_file='text_data.txt', seed=0)

BOARD_PATH = "./board/lab09-3_board"
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


def lstm_layer(tensor_op, activiation=tf.nn.tanh, use_peepholes=False, name="LSTM_Layer"):
    with tf.variable_scope(name):
        cell = tf.contrib.rnn.LSTMCell(num_units=NUM_RNN_UNITS, use_peepholes=use_peepholes, activation=activiation)
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

l1 = lstm_layer(embedding_output, use_peepholes=True, name="LSTM_Layer1")
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
Epoch [ 1/30], train loss = 0.472987, train accuracy = 86.05%, valid loss = 0.476580, valid accuracy = 84.02%, duration = 1.322302(s)
Epoch [ 2/30], train loss = 0.444044, train accuracy = 86.96%, valid loss = 0.470032, valid accuracy = 84.20%, duration = 1.028611(s)
Epoch [ 3/30], train loss = 0.404451, train accuracy = 91.06%, valid loss = 0.382713, valid accuracy = 93.54%, duration = 0.972990(s)
Epoch [ 4/30], train loss = 0.354702, train accuracy = 96.33%, valid loss = 0.366938, valid accuracy = 95.15%, duration = 0.969757(s)
Epoch [ 5/30], train loss = 0.343842, train accuracy = 97.31%, valid loss = 0.362825, valid accuracy = 95.51%, duration = 0.992581(s)
Epoch [ 6/30], train loss = 0.340054, train accuracy = 97.57%, valid loss = 0.357578, valid accuracy = 95.51%, duration = 0.989183(s)
Epoch [ 7/30], train loss = 0.337144, train accuracy = 97.83%, valid loss = 0.356089, valid accuracy = 95.69%, duration = 0.995116(s)
Epoch [ 8/30], train loss = 0.337890, train accuracy = 97.70%, valid loss = 0.355146, valid accuracy = 95.87%, duration = 0.985210(s)
Epoch [ 9/30], train loss = 0.335494, train accuracy = 97.93%, valid loss = 0.356725, valid accuracy = 95.69%, duration = 0.963984(s)
Epoch [10/30], train loss = 0.331181, train accuracy = 98.32%, valid loss = 0.355794, valid accuracy = 95.87%, duration = 0.991150(s)
Epoch [11/30], train loss = 0.329723, train accuracy = 98.48%, valid loss = 0.353453, valid accuracy = 95.69%, duration = 0.990958(s)
Epoch [12/30], train loss = 0.329119, train accuracy = 98.50%, valid loss = 0.353106, valid accuracy = 96.05%, duration = 1.001763(s)
Epoch [13/30], train loss = 0.328817, train accuracy = 98.50%, valid loss = 0.355148, valid accuracy = 95.69%, duration = 0.975287(s)
Epoch [14/30], train loss = 0.328110, train accuracy = 98.58%, valid loss = 0.352203, valid accuracy = 96.05%, duration = 0.986533(s)
Epoch [15/30], train loss = 0.328162, train accuracy = 98.58%, valid loss = 0.351971, valid accuracy = 96.23%, duration = 0.985759(s)
Epoch [16/30], train loss = 0.327851, train accuracy = 98.61%, valid loss = 0.350518, valid accuracy = 96.41%, duration = 0.987973(s)
Epoch [17/30], train loss = 0.327570, train accuracy = 98.63%, valid loss = 0.353674, valid accuracy = 96.05%, duration = 0.989571(s)
Epoch [18/30], train loss = 0.327683, train accuracy = 98.61%, valid loss = 0.353432, valid accuracy = 96.05%, duration = 0.983298(s)
Epoch [19/30], train loss = 0.327627, train accuracy = 98.61%, valid loss = 0.352441, valid accuracy = 95.87%, duration = 0.999764(s)
Epoch [20/30], train loss = 0.327084, train accuracy = 98.66%, valid loss = 0.352669, valid accuracy = 96.05%, duration = 0.992417(s)
Epoch [21/30], train loss = 0.326268, train accuracy = 98.73%, valid loss = 0.352768, valid accuracy = 96.05%, duration = 0.998081(s)
Epoch [22/30], train loss = 0.327031, train accuracy = 98.66%, valid loss = 0.351447, valid accuracy = 96.23%, duration = 0.985943(s)
Epoch [23/30], train loss = 0.327339, train accuracy = 98.63%, valid loss = 0.350882, valid accuracy = 96.05%, duration = 0.984722(s)
Epoch [24/30], train loss = 0.326490, train accuracy = 98.71%, valid loss = 0.350906, valid accuracy = 96.23%, duration = 0.988670(s)
Epoch [25/30], train loss = 0.326714, train accuracy = 98.68%, valid loss = 0.350901, valid accuracy = 96.23%, duration = 0.969284(s)
Epoch [26/30], train loss = 0.326693, train accuracy = 98.68%, valid loss = 0.351019, valid accuracy = 96.23%, duration = 0.984522(s)
Epoch [27/30], train loss = 0.326674, train accuracy = 98.68%, valid loss = 0.351047, valid accuracy = 96.05%, duration = 1.013560(s)
Epoch [28/30], train loss = 0.326656, train accuracy = 98.68%, valid loss = 0.350888, valid accuracy = 96.23%, duration = 0.996426(s)
Epoch [29/30], train loss = 0.326635, train accuracy = 98.68%, valid loss = 0.350760, valid accuracy = 96.23%, duration = 0.971494(s)
Epoch [30/30], train loss = 0.326614, train accuracy = 98.68%, valid loss = 0.350535, valid accuracy = 96.23%, duration = 1.010330(s)
Duration for train : 30.549353(s)
<<< Train Finished >>>
Test Accraucy : 97.58%
'''