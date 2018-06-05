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


with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, MAX_SEQUENCE_LENGTH], dtype=tf.int32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

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
    gvs = optimizer.compute_gradients(loss)
    clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    optim = optimizer.apply_gradients(clipped_gvs)

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
Epoch [ 1/30], train loss = 0.486212, train accuracy = 82.93%, valid loss = 0.481001, valid accuracy = 83.66%, duration = 0.983454(s)
Epoch [ 2/30], train loss = 0.445594, train accuracy = 87.14%, valid loss = 0.472510, valid accuracy = 84.38%, duration = 0.869377(s)
Epoch [ 3/30], train loss = 0.412765, train accuracy = 90.37%, valid loss = 0.394673, valid accuracy = 93.18%, duration = 0.814092(s)
Epoch [ 4/30], train loss = 0.360857, train accuracy = 95.74%, valid loss = 0.372939, valid accuracy = 94.61%, duration = 0.809755(s)
Epoch [ 5/30], train loss = 0.348183, train accuracy = 96.98%, valid loss = 0.369181, valid accuracy = 94.97%, duration = 0.877651(s)
Epoch [ 6/30], train loss = 0.341969, train accuracy = 97.47%, valid loss = 0.362072, valid accuracy = 95.15%, duration = 0.845243(s)
Epoch [ 7/30], train loss = 0.336809, train accuracy = 97.91%, valid loss = 0.354322, valid accuracy = 95.87%, duration = 0.865545(s)
Epoch [ 8/30], train loss = 0.333103, train accuracy = 98.27%, valid loss = 0.351058, valid accuracy = 96.41%, duration = 0.842646(s)
Epoch [ 9/30], train loss = 0.331120, train accuracy = 98.42%, valid loss = 0.347037, valid accuracy = 96.95%, duration = 0.798245(s)
Epoch [10/30], train loss = 0.329126, train accuracy = 98.58%, valid loss = 0.344989, valid accuracy = 96.77%, duration = 0.770855(s)
Epoch [11/30], train loss = 0.328507, train accuracy = 98.58%, valid loss = 0.346792, valid accuracy = 96.59%, duration = 0.791242(s)
Epoch [12/30], train loss = 0.326862, train accuracy = 98.76%, valid loss = 0.345999, valid accuracy = 96.77%, duration = 0.794129(s)
Epoch [13/30], train loss = 0.326506, train accuracy = 98.79%, valid loss = 0.344481, valid accuracy = 96.95%, duration = 0.787929(s)
Epoch [14/30], train loss = 0.325994, train accuracy = 98.81%, valid loss = 0.346817, valid accuracy = 96.59%, duration = 0.790047(s)
Epoch [15/30], train loss = 0.325797, train accuracy = 98.81%, valid loss = 0.345357, valid accuracy = 96.77%, duration = 0.776783(s)
Epoch [16/30], train loss = 0.325611, train accuracy = 98.81%, valid loss = 0.342566, valid accuracy = 97.13%, duration = 0.786952(s)
Epoch [17/30], train loss = 0.325068, train accuracy = 98.89%, valid loss = 0.343601, valid accuracy = 96.95%, duration = 0.963380(s)
Epoch [18/30], train loss = 0.324885, train accuracy = 98.92%, valid loss = 0.344557, valid accuracy = 96.77%, duration = 1.261034(s)
Epoch [19/30], train loss = 0.324792, train accuracy = 98.92%, valid loss = 0.348957, valid accuracy = 96.05%, duration = 0.873399(s)
Epoch [20/30], train loss = 0.324569, train accuracy = 98.94%, valid loss = 0.348036, valid accuracy = 96.23%, duration = 0.836372(s)
Epoch [21/30], train loss = 0.324428, train accuracy = 98.94%, valid loss = 0.346869, valid accuracy = 96.59%, duration = 0.872613(s)
Epoch [22/30], train loss = 0.324335, train accuracy = 98.94%, valid loss = 0.349800, valid accuracy = 96.05%, duration = 0.869927(s)
Epoch [23/30], train loss = 0.324044, train accuracy = 98.97%, valid loss = 0.349340, valid accuracy = 96.23%, duration = 0.884630(s)
Epoch [24/30], train loss = 0.324007, train accuracy = 98.97%, valid loss = 0.348037, valid accuracy = 96.41%, duration = 0.827265(s)
Epoch [25/30], train loss = 0.324224, train accuracy = 98.94%, valid loss = 0.348864, valid accuracy = 96.23%, duration = 0.832333(s)
Epoch [26/30], train loss = 0.324195, train accuracy = 98.94%, valid loss = 0.349400, valid accuracy = 96.23%, duration = 0.849554(s)
Epoch [27/30], train loss = 0.324172, train accuracy = 98.94%, valid loss = 0.349601, valid accuracy = 96.23%, duration = 0.804095(s)
Epoch [28/30], train loss = 0.324151, train accuracy = 98.94%, valid loss = 0.349713, valid accuracy = 96.23%, duration = 0.793761(s)
Epoch [29/30], train loss = 0.324130, train accuracy = 98.94%, valid loss = 0.350244, valid accuracy = 96.05%, duration = 0.802515(s)
Epoch [30/30], train loss = 0.324117, train accuracy = 98.94%, valid loss = 0.350474, valid accuracy = 96.05%, duration = 0.820426(s)
Duration for train : 25.963353(s)
<<< Train Finished >>>
Test Accraucy : 96.95%
'''
