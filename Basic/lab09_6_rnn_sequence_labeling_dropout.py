import tensorflow as tf
import numpy as np
import load_data
import shutil
import time
import os

x_train, x_validation, x_test, y_train, y_validation, y_test, VOCAB_SIZE \
    = load_data.load_spam_data(data_dir='./data', data_file='text_data.txt', seed=0)

BOARD_PATH = "./board/lab09-8_board"
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


with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, MAX_SEQUENCE_LENGTH], dtype=tf.int32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

embedding_mat = tf.get_variable(name='embedding', shape=[VOCAB_SIZE, EMBEDDING_SIZE],
                                initializer=tf.random_normal_initializer(stddev=1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, X)

l1 = lstm_layer(embedding_output, use_peepholes=False, name="LSTM_Layer1")
d1 = tf.nn.dropout(l1, keep_prob=keep_prob)
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
                               feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :],learning_rate:u, keep_prob:0.7})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step * BATCH_SIZE
        acc_per_epoch /= total_step * BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation, keep_prob:1.0})
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

    ta = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob:1.0})
    print("Test Accraucy : {:.2%}".format(ta / ntest))

'''
Epoch [ 1/30], train loss = 0.458347, train accuracy = 86.03%, valid loss = 0.476486, valid accuracy = 84.20%, duration = 0.966373(s)
Epoch [ 2/30], train loss = 0.439943, train accuracy = 87.40%, valid loss = 0.445550, valid accuracy = 85.64%, duration = 0.732860(s)
Epoch [ 3/30], train loss = 0.375556, train accuracy = 94.11%, valid loss = 0.370075, valid accuracy = 94.97%, duration = 0.732490(s)
Epoch [ 4/30], train loss = 0.347541, train accuracy = 96.95%, valid loss = 0.368606, valid accuracy = 94.79%, duration = 0.731692(s)
Epoch [ 5/30], train loss = 0.340340, train accuracy = 97.44%, valid loss = 0.358535, valid accuracy = 95.33%, duration = 0.699957(s)
Epoch [ 6/30], train loss = 0.335841, train accuracy = 97.91%, valid loss = 0.355635, valid accuracy = 95.69%, duration = 0.691601(s)
Epoch [ 7/30], train loss = 0.332485, train accuracy = 98.19%, valid loss = 0.352472, valid accuracy = 96.23%, duration = 0.689250(s)
Epoch [ 8/30], train loss = 0.331837, train accuracy = 98.30%, valid loss = 0.348355, valid accuracy = 96.59%, duration = 0.689120(s)
Epoch [ 9/30], train loss = 0.329306, train accuracy = 98.53%, valid loss = 0.347416, valid accuracy = 96.77%, duration = 0.702570(s)
Epoch [10/30], train loss = 0.326660, train accuracy = 98.76%, valid loss = 0.346221, valid accuracy = 96.77%, duration = 0.708256(s)
Epoch [11/30], train loss = 0.325843, train accuracy = 98.81%, valid loss = 0.347384, valid accuracy = 96.59%, duration = 0.690632(s)
Epoch [12/30], train loss = 0.325486, train accuracy = 98.86%, valid loss = 0.345152, valid accuracy = 96.59%, duration = 0.690803(s)
Epoch [13/30], train loss = 0.325113, train accuracy = 98.92%, valid loss = 0.348548, valid accuracy = 96.41%, duration = 0.694453(s)
Epoch [14/30], train loss = 0.324782, train accuracy = 98.92%, valid loss = 0.347829, valid accuracy = 96.41%, duration = 0.693993(s)
Epoch [15/30], train loss = 0.324661, train accuracy = 98.92%, valid loss = 0.348385, valid accuracy = 96.41%, duration = 0.693257(s)
Epoch [16/30], train loss = 0.324554, train accuracy = 98.92%, valid loss = 0.348601, valid accuracy = 96.41%, duration = 0.698827(s)
Epoch [17/30], train loss = 0.324237, train accuracy = 98.94%, valid loss = 0.348864, valid accuracy = 96.41%, duration = 0.692100(s)
Epoch [18/30], train loss = 0.324448, train accuracy = 98.92%, valid loss = 0.348994, valid accuracy = 96.23%, duration = 0.693655(s)
Epoch [19/30], train loss = 0.324404, train accuracy = 98.92%, valid loss = 0.349044, valid accuracy = 96.23%, duration = 0.698820(s)
Epoch [20/30], train loss = 0.324366, train accuracy = 98.92%, valid loss = 0.349154, valid accuracy = 96.23%, duration = 0.689532(s)
Epoch [21/30], train loss = 0.324075, train accuracy = 98.94%, valid loss = 0.349335, valid accuracy = 96.23%, duration = 0.695010(s)
Epoch [22/30], train loss = 0.324296, train accuracy = 98.92%, valid loss = 0.349567, valid accuracy = 96.23%, duration = 0.702351(s)
Epoch [23/30], train loss = 0.324000, train accuracy = 98.94%, valid loss = 0.349779, valid accuracy = 96.05%, duration = 0.703061(s)
Epoch [24/30], train loss = 0.323947, train accuracy = 98.94%, valid loss = 0.349804, valid accuracy = 96.23%, duration = 0.697468(s)
Epoch [25/30], train loss = 0.324161, train accuracy = 98.92%, valid loss = 0.349944, valid accuracy = 96.23%, duration = 0.695522(s)
Epoch [26/30], train loss = 0.323823, train accuracy = 98.94%, valid loss = 0.349834, valid accuracy = 96.23%, duration = 0.693964(s)
Epoch [27/30], train loss = 0.324015, train accuracy = 98.92%, valid loss = 0.350154, valid accuracy = 96.23%, duration = 0.701341(s)
Epoch [28/30], train loss = 0.323957, train accuracy = 98.97%, valid loss = 0.349533, valid accuracy = 96.41%, duration = 0.687824(s)
Epoch [29/30], train loss = 0.323900, train accuracy = 98.97%, valid loss = 0.350188, valid accuracy = 96.41%, duration = 0.691595(s)
Epoch [30/30], train loss = 0.323861, train accuracy = 98.97%, valid loss = 0.349685, valid accuracy = 96.41%, duration = 0.697615(s)
Duration for train : 21.714434(s)
<<< Train Finished >>>
Test Accraucy : 97.94%
'''