import tensorflow as tf
import numpy as np
import load_data
import shutil
import time
import os

x_train, x_validation, x_test, y_train, y_validation, y_test, VOCAB_SIZE \
    = load_data.load_spam_data(data_dir='./data', data_file='text_data.txt', seed=0)

BOARD_PATH = "./board/lab09-7_board"
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

def lstm_cell(num_units, activation, use_peepholes):
    cell = tf.contrib.rnn.LSTMCell(num_units=num_units, use_peepholes=use_peepholes, activation=activation)
    return cell

def lstm_layer(tensor_op, activation=tf.nn.tanh, num_layer=1, use_peepholes=False, name="LSTM_Layer"):
    with tf.variable_scope(name):
        stacked_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(NUM_RNN_UNITS, activation, use_peepholes) for _ in range(num_layer)])
        output, _ = tf.nn.dynamic_rnn(stacked_cell, tensor_op, dtype=tf.float32)
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

l1 = lstm_layer(embedding_output, num_layer=3, use_peepholes=False, name="LSTM_Layer1")
h1 = linear(tensor_op=l1, output_dim=NCLASS, name='FCLayer1')
logits = tf.nn.softmax(h1)

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='loss')
    optim= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

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
Epoch [ 1/30], train loss = 0.483855, train accuracy = 86.49%, valid loss = 0.454900, valid accuracy = 85.10%, duration = 1.656429(s)
Epoch [ 2/30], train loss = 0.388181, train accuracy = 93.90%, valid loss = 0.382080, valid accuracy = 93.90%, duration = 1.327586(s)
Epoch [ 3/30], train loss = 0.355735, train accuracy = 96.51%, valid loss = 0.366349, valid accuracy = 95.15%, duration = 1.331749(s)
Epoch [ 4/30], train loss = 0.343056, train accuracy = 97.42%, valid loss = 0.364024, valid accuracy = 94.97%, duration = 1.322346(s)
Epoch [ 5/30], train loss = 0.338422, train accuracy = 97.70%, valid loss = 0.361655, valid accuracy = 94.97%, duration = 1.323588(s)
Epoch [ 6/30], train loss = 0.335232, train accuracy = 97.96%, valid loss = 0.355535, valid accuracy = 95.87%, duration = 1.313298(s)
Epoch [ 7/30], train loss = 0.334161, train accuracy = 98.06%, valid loss = 0.358271, valid accuracy = 95.33%, duration = 1.328941(s)
Epoch [ 8/30], train loss = 0.331764, train accuracy = 98.22%, valid loss = 0.353813, valid accuracy = 95.87%, duration = 1.344908(s)
Epoch [ 9/30], train loss = 0.332245, train accuracy = 98.19%, valid loss = 0.352364, valid accuracy = 96.05%, duration = 1.322268(s)
Epoch [10/30], train loss = 0.329944, train accuracy = 98.40%, valid loss = 0.354038, valid accuracy = 95.87%, duration = 1.328722(s)
Epoch [11/30], train loss = 0.329625, train accuracy = 98.40%, valid loss = 0.352861, valid accuracy = 95.87%, duration = 1.328411(s)
Epoch [12/30], train loss = 0.326765, train accuracy = 98.71%, valid loss = 0.353860, valid accuracy = 95.87%, duration = 1.336565(s)
Epoch [13/30], train loss = 0.326261, train accuracy = 98.76%, valid loss = 0.351221, valid accuracy = 96.05%, duration = 1.452471(s)
Epoch [14/30], train loss = 0.325637, train accuracy = 98.81%, valid loss = 0.349872, valid accuracy = 96.23%, duration = 1.364088(s)
Epoch [15/30], train loss = 0.325142, train accuracy = 98.86%, valid loss = 0.351874, valid accuracy = 96.05%, duration = 1.388689(s)
Epoch [16/30], train loss = 0.324962, train accuracy = 98.86%, valid loss = 0.350661, valid accuracy = 96.05%, duration = 1.371433(s)
Epoch [17/30], train loss = 0.324441, train accuracy = 98.92%, valid loss = 0.351010, valid accuracy = 96.05%, duration = 1.384334(s)
Epoch [18/30], train loss = 0.324655, train accuracy = 98.89%, valid loss = 0.351135, valid accuracy = 96.05%, duration = 1.654048(s)
Epoch [19/30], train loss = 0.324619, train accuracy = 98.89%, valid loss = 0.350903, valid accuracy = 96.05%, duration = 1.528623(s)
Epoch [20/30], train loss = 0.324585, train accuracy = 98.89%, valid loss = 0.350894, valid accuracy = 96.05%, duration = 1.446282(s)
Epoch [21/30], train loss = 0.324078, train accuracy = 98.94%, valid loss = 0.350851, valid accuracy = 96.23%, duration = 1.371589(s)
Epoch [22/30], train loss = 0.324301, train accuracy = 98.92%, valid loss = 0.351015, valid accuracy = 96.05%, duration = 1.402765(s)
Epoch [23/30], train loss = 0.324285, train accuracy = 98.92%, valid loss = 0.351069, valid accuracy = 96.05%, duration = 1.325231(s)
Epoch [24/30], train loss = 0.324014, train accuracy = 98.94%, valid loss = 0.350996, valid accuracy = 96.05%, duration = 1.330541(s)
Epoch [25/30], train loss = 0.324260, train accuracy = 98.92%, valid loss = 0.351095, valid accuracy = 96.05%, duration = 1.336769(s)
Epoch [26/30], train loss = 0.323992, train accuracy = 98.94%, valid loss = 0.350984, valid accuracy = 96.05%, duration = 1.360646(s)
Epoch [27/30], train loss = 0.324239, train accuracy = 98.92%, valid loss = 0.350783, valid accuracy = 96.05%, duration = 1.364930(s)
Epoch [28/30], train loss = 0.324228, train accuracy = 98.92%, valid loss = 0.350414, valid accuracy = 96.23%, duration = 1.323945(s)
Epoch [29/30], train loss = 0.324199, train accuracy = 98.92%, valid loss = 0.348955, valid accuracy = 96.41%, duration = 1.322856(s)
Epoch [30/30], train loss = 0.324447, train accuracy = 98.89%, valid loss = 0.351370, valid accuracy = 96.05%, duration = 1.336988(s)
Duration for train : 41.802052(s)
<<< Train Finished >>>
Test Accraucy : 97.31%
'''