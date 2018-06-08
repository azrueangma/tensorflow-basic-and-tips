import tensorflow as tf
import numpy as np
import load_data
import shutil
import time
import os

x_train, x_validation, x_test, y_train, y_validation, y_test, VOCAB_SIZE \
    = load_data.load_spam_data(data_dir='./data', data_file='text_data.txt', seed=0)

BOARD_PATH = "./board/lab09-4_board"
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

tf.set_random_seed(0)


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


def lstm_layer(tensor_op, activation=tf.nn.tanh, use_peepholes=False, name="LSTM_Layer"):
    with tf.variable_scope(name):
        cell = tf.contrib.rnn.LSTMCell(num_units=NUM_RNN_UNITS, use_peepholes=use_peepholes, activation=activation)
        output, state = tf.nn.dynamic_rnn(cell, tensor_op, dtype=tf.float32)
        output = tf.transpose(output, [1, 0, 2])
        last = tf.gather(output, int(output.get_shape()[0]) - 1)
    return last


def _optimizer(optimizer, cost, is_clipping):
    temp = tf.cond(tf.equal(is_clipping, tf.constant(True)), lambda:tf.constant(1), lambda:tf.constant(0))
    if temp == 1:
        gvs = optimizer.compute_gradients(cost)
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        return optimizer.apply_gradients(clipped_gvs)
    else:
        return optimizer.minimize(cost)


with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, MAX_SEQUENCE_LENGTH], dtype=tf.int32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    is_clipping = tf.placeholder(tf.bool, name='is_clipping')

embedding_mat = tf.get_variable(name='embedding', shape=[VOCAB_SIZE, EMBEDDING_SIZE],
                                initializer=tf.random_normal_initializer(stddev=1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, X)

l1 = lstm_layer(embedding_output, use_peepholes=False, name="LSTM_Layer1")
h1 = linear(tensor_op=l1, output_dim=NCLASS, name='FCLayer1')
logits = tf.nn.softmax(h1)

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optim = _optimizer(optimizer, loss, is_clipping)

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
                               feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :],learning_rate:u, is_clipping:True})
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
Epoch [ 1/30], train loss = 0.486680, train accuracy = 85.64%, valid loss = 0.480409, valid accuracy = 84.02%, duration = 0.877145(s)
Epoch [ 2/30], train loss = 0.446443, train accuracy = 86.91%, valid loss = 0.470609, valid accuracy = 84.56%, duration = 0.695338(s)
Epoch [ 3/30], train loss = 0.416889, train accuracy = 89.70%, valid loss = 0.385344, valid accuracy = 94.08%, duration = 0.700472(s)
Epoch [ 4/30], train loss = 0.358459, train accuracy = 96.10%, valid loss = 0.371586, valid accuracy = 94.61%, duration = 0.705691(s)
Epoch [ 5/30], train loss = 0.342481, train accuracy = 97.47%, valid loss = 0.357806, valid accuracy = 95.87%, duration = 0.722070(s)
Epoch [ 6/30], train loss = 0.336470, train accuracy = 98.01%, valid loss = 0.357242, valid accuracy = 96.05%, duration = 0.705538(s)
Epoch [ 7/30], train loss = 0.333375, train accuracy = 98.24%, valid loss = 0.352761, valid accuracy = 96.23%, duration = 0.716857(s)
Epoch [ 8/30], train loss = 0.331380, train accuracy = 98.40%, valid loss = 0.350842, valid accuracy = 96.41%, duration = 0.720519(s)
Epoch [ 9/30], train loss = 0.330196, train accuracy = 98.48%, valid loss = 0.349967, valid accuracy = 96.23%, duration = 0.714951(s)
Epoch [10/30], train loss = 0.329029, train accuracy = 98.58%, valid loss = 0.349940, valid accuracy = 96.41%, duration = 0.698603(s)
Epoch [11/30], train loss = 0.328297, train accuracy = 98.61%, valid loss = 0.348834, valid accuracy = 96.41%, duration = 0.693119(s)
Epoch [12/30], train loss = 0.327654, train accuracy = 98.68%, valid loss = 0.349205, valid accuracy = 96.41%, duration = 0.685065(s)
Epoch [13/30], train loss = 0.326647, train accuracy = 98.76%, valid loss = 0.348737, valid accuracy = 96.41%, duration = 0.695041(s)
Epoch [14/30], train loss = 0.325855, train accuracy = 98.84%, valid loss = 0.349103, valid accuracy = 96.05%, duration = 0.680228(s)
Epoch [15/30], train loss = 0.325593, train accuracy = 98.84%, valid loss = 0.349894, valid accuracy = 96.23%, duration = 0.696791(s)
Epoch [16/30], train loss = 0.325426, train accuracy = 98.86%, valid loss = 0.350251, valid accuracy = 96.05%, duration = 0.706505(s)
Epoch [17/30], train loss = 0.325058, train accuracy = 98.89%, valid loss = 0.350528, valid accuracy = 96.05%, duration = 0.687819(s)
Epoch [18/30], train loss = 0.325237, train accuracy = 98.86%, valid loss = 0.350899, valid accuracy = 96.05%, duration = 0.702524(s)
Epoch [19/30], train loss = 0.324927, train accuracy = 98.89%, valid loss = 0.350098, valid accuracy = 96.23%, duration = 0.685976(s)
Epoch [20/30], train loss = 0.324860, train accuracy = 98.89%, valid loss = 0.350401, valid accuracy = 96.05%, duration = 0.686555(s)
Epoch [21/30], train loss = 0.324551, train accuracy = 98.92%, valid loss = 0.350468, valid accuracy = 96.05%, duration = 0.682068(s)
Epoch [22/30], train loss = 0.324764, train accuracy = 98.89%, valid loss = 0.350523, valid accuracy = 96.05%, duration = 0.677121(s)
Epoch [23/30], train loss = 0.324717, train accuracy = 98.89%, valid loss = 0.350658, valid accuracy = 96.05%, duration = 0.684875(s)
Epoch [24/30], train loss = 0.324416, train accuracy = 98.92%, valid loss = 0.350617, valid accuracy = 96.05%, duration = 0.682738(s)
Epoch [25/30], train loss = 0.324376, train accuracy = 98.92%, valid loss = 0.350680, valid accuracy = 96.05%, duration = 0.673628(s)
Epoch [26/30], train loss = 0.324595, train accuracy = 98.89%, valid loss = 0.350576, valid accuracy = 96.05%, duration = 0.690053(s)
Epoch [27/30], train loss = 0.324552, train accuracy = 98.89%, valid loss = 0.349468, valid accuracy = 96.05%, duration = 0.681059(s)
Epoch [28/30], train loss = 0.323575, train accuracy = 98.99%, valid loss = 0.347253, valid accuracy = 96.59%, duration = 0.674873(s)
Epoch [29/30], train loss = 0.322854, train accuracy = 99.07%, valid loss = 0.343952, valid accuracy = 96.77%, duration = 0.688686(s)
Epoch [30/30], train loss = 0.322468, train accuracy = 99.10%, valid loss = 0.345135, valid accuracy = 96.59%, duration = 0.690086(s)
Duration for train : 21.421423(s)
<<< Train Finished >>>
Test Accraucy : 96.86%
'''
