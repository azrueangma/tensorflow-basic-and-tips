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

GLOBAL_STEP = 0

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


def relu_layer(tensor_op, output_dim, weight_decay=False, regularizer=None,
               keep_prob=1.0, is_training=False, with_W=False, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name='b', shape=[output_dim], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name='pre_op')
        bn = tf.contrib.layers.batch_norm(pre_activation,
                                          is_training=is_training,
                                          updates_collections=None)
        h = tf.nn.relu(bn, name='tanh_op')
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
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    is_training = tf.placeholder(tf.bool, name='is_training')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

embedding_mat = tf.get_variable(name='embedding', shape=[VOCAB_SIZE, EMBEDDING_SIZE],
                                initializer=tf.random_normal_initializer(stddev=1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, X)

l1 = lstm_layer(embedding_output, num_layer=3, use_peepholes=False, name="LSTM_Layers")
h1 = relu_layer(l1, NUM_RNN_UNITS*2, keep_prob=keep_prob, is_training=is_training, name='FCLayer1')
h2 = linear(h1, output_dim=NCLASS, name='FCLayer2')
logits = tf.nn.softmax(h2)

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name='hypothesis')
    loss=tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot), name='loss')
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    if GLOBAL_STEP>=20:
        gvs = optimizer.compute_gradients(loss)
        clipped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        optim = optimizer.apply_gradients(clipped_gvs)
    else:
        optim = optimizer.minimize(loss)

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
                               feed_dict={X: x_train[mask[s:t], :], Y: y_train[mask[s:t], :],learning_rate:u,
                                          keep_prob:0.5, is_training:True})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step * BATCH_SIZE
        acc_per_epoch /= total_step * BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation,
                                                       keep_prob:1.0, is_training:False})
        epoch_valid_acc = va / len(x_validation)
        epoch_valid_loss = vl / len(x_validation)

        s = sess.run(merged, feed_dict={avg_train_loss: loss_per_epoch, avg_train_acc: acc_per_epoch,
                                        avg_validation_loss: epoch_valid_loss, avg_validation_acc: epoch_valid_acc})
        writer.add_summary(s, global_step=epoch)

        u = u*0.95
        GLOBAL_STEP += 1
        if (epoch + 1) % 1 == 0:
            print("Epoch [{:2d}/{:2d}], train loss = {:.6f}, train accuracy = {:.2%}, "
                  "valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)"
                  .format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc,
                          epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

    ta = sess.run(accuracy, feed_dict={X: x_test, Y: y_test, keep_prob:1.0, is_training:False})
    print("Test Accraucy : {:.2%}".format(ta / ntest))

'''
Epoch [ 1/30], train loss = 0.607243, train accuracy = 75.41%, valid loss = 0.594640, valid accuracy = 83.84%, duration = 1.758112(s)
Epoch [ 2/30], train loss = 0.481611, train accuracy = 88.51%, valid loss = 0.538065, valid accuracy = 83.84%, duration = 1.416298(s)
Epoch [ 3/30], train loss = 0.410500, train accuracy = 93.65%, valid loss = 0.494813, valid accuracy = 83.84%, duration = 1.376373(s)
Epoch [ 4/30], train loss = 0.375817, train accuracy = 96.02%, valid loss = 0.460375, valid accuracy = 84.02%, duration = 1.369424(s)
Epoch [ 5/30], train loss = 0.361132, train accuracy = 96.93%, valid loss = 0.437578, valid accuracy = 86.71%, duration = 1.386466(s)
Epoch [ 6/30], train loss = 0.349041, train accuracy = 97.70%, valid loss = 0.417269, valid accuracy = 90.48%, duration = 1.398835(s)
Epoch [ 7/30], train loss = 0.344110, train accuracy = 97.86%, valid loss = 0.385705, valid accuracy = 94.43%, duration = 1.355973(s)
Epoch [ 8/30], train loss = 0.339800, train accuracy = 98.09%, valid loss = 0.372637, valid accuracy = 95.51%, duration = 1.341260(s)
Epoch [ 9/30], train loss = 0.335431, train accuracy = 98.32%, valid loss = 0.368054, valid accuracy = 96.05%, duration = 1.357013(s)
Epoch [10/30], train loss = 0.333277, train accuracy = 98.35%, valid loss = 0.363482, valid accuracy = 96.23%, duration = 1.374993(s)
Epoch [11/30], train loss = 0.328520, train accuracy = 98.89%, valid loss = 0.359056, valid accuracy = 96.41%, duration = 1.349257(s)
Epoch [12/30], train loss = 0.328963, train accuracy = 98.79%, valid loss = 0.362919, valid accuracy = 96.05%, duration = 1.354357(s)
Epoch [13/30], train loss = 0.327993, train accuracy = 98.84%, valid loss = 0.346173, valid accuracy = 97.49%, duration = 1.368881(s)
Epoch [14/30], train loss = 0.324871, train accuracy = 99.04%, valid loss = 0.345010, valid accuracy = 97.31%, duration = 1.361458(s)
Epoch [15/30], train loss = 0.325485, train accuracy = 99.04%, valid loss = 0.343386, valid accuracy = 97.31%, duration = 1.360478(s)
Epoch [16/30], train loss = 0.324407, train accuracy = 99.12%, valid loss = 0.342727, valid accuracy = 97.49%, duration = 1.412186(s)
Epoch [17/30], train loss = 0.322898, train accuracy = 99.30%, valid loss = 0.340571, valid accuracy = 97.67%, duration = 1.445783(s)
Epoch [18/30], train loss = 0.323205, train accuracy = 99.17%, valid loss = 0.339468, valid accuracy = 97.49%, duration = 1.451334(s)
Epoch [19/30], train loss = 0.323490, train accuracy = 99.15%, valid loss = 0.337701, valid accuracy = 97.67%, duration = 1.442840(s)
Epoch [20/30], train loss = 0.323485, train accuracy = 99.12%, valid loss = 0.339116, valid accuracy = 97.49%, duration = 1.426095(s)
Epoch [21/30], train loss = 0.322536, train accuracy = 99.20%, valid loss = 0.338149, valid accuracy = 97.67%, duration = 1.456306(s)
Epoch [22/30], train loss = 0.322058, train accuracy = 99.28%, valid loss = 0.338268, valid accuracy = 97.67%, duration = 1.450133(s)
Epoch [23/30], train loss = 0.322118, train accuracy = 99.23%, valid loss = 0.338061, valid accuracy = 97.67%, duration = 1.358750(s)
Epoch [24/30], train loss = 0.322113, train accuracy = 99.25%, valid loss = 0.337178, valid accuracy = 97.85%, duration = 1.356240(s)
Epoch [25/30], train loss = 0.325442, train accuracy = 98.86%, valid loss = 0.350876, valid accuracy = 96.05%, duration = 1.351650(s)
Epoch [26/30], train loss = 0.322209, train accuracy = 99.23%, valid loss = 0.337211, valid accuracy = 97.49%, duration = 1.349526(s)
Epoch [27/30], train loss = 0.322386, train accuracy = 99.15%, valid loss = 0.336327, valid accuracy = 97.67%, duration = 1.368204(s)
Epoch [28/30], train loss = 0.321920, train accuracy = 99.25%, valid loss = 0.337210, valid accuracy = 97.67%, duration = 1.343893(s)
Epoch [29/30], train loss = 0.321467, train accuracy = 99.25%, valid loss = 0.335978, valid accuracy = 97.67%, duration = 1.343708(s)
Epoch [30/30], train loss = 0.322013, train accuracy = 99.20%, valid loss = 0.336507, valid accuracy = 97.67%, duration = 1.345457(s)
Duration for train : 42.351785(s)
<<< Train Finished >>>
Test Accraucy : 97.94%
'''