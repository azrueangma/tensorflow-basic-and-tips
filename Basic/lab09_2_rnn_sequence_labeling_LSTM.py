import tensorflow as tf
import numpy as np
import load_data
import shutil
import time
import os

x_train, x_validation, x_test, y_train, y_validation, y_test, VOCAB_SIZE \
    = load_data.load_spam_data(data_dir='./data', data_file='text_data.txt', seed=0)

BOARD_PATH = "./board/lab09-2_board"
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


def lstm_layer(tensor_op, activation=tf.nn.tanh, name="LSTM_Layer"):
    with tf.variable_scope(name):
        cell = tf.contrib.rnn.LSTMCell(num_units=NUM_RNN_UNITS, activation=activation)
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

l1 = lstm_layer(embedding_output, name="LSTM_Layer1")
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
Epoch [ 1/30], train loss = 0.487785, train accuracy = 82.90%, valid loss = 0.481746, valid accuracy = 83.66%, duration = 0.970089(s)
Epoch [ 2/30], train loss = 0.445551, train accuracy = 87.11%, valid loss = 0.471442, valid accuracy = 84.38%, duration = 0.766803(s)
Epoch [ 3/30], train loss = 0.413851, train accuracy = 90.21%, valid loss = 0.396058, valid accuracy = 92.46%, duration = 0.761836(s)
Epoch [ 4/30], train loss = 0.361742, train accuracy = 95.56%, valid loss = 0.373348, valid accuracy = 94.25%, duration = 0.768600(s)
Epoch [ 5/30], train loss = 0.348401, train accuracy = 96.95%, valid loss = 0.368274, valid accuracy = 94.79%, duration = 0.762522(s)
Epoch [ 6/30], train loss = 0.342802, train accuracy = 97.29%, valid loss = 0.359926, valid accuracy = 95.87%, duration = 0.759380(s)
Epoch [ 7/30], train loss = 0.337310, train accuracy = 97.86%, valid loss = 0.353773, valid accuracy = 96.05%, duration = 0.768160(s)
Epoch [ 8/30], train loss = 0.333187, train accuracy = 98.32%, valid loss = 0.351862, valid accuracy = 96.05%, duration = 0.755717(s)
Epoch [ 9/30], train loss = 0.330717, train accuracy = 98.45%, valid loss = 0.349585, valid accuracy = 96.59%, duration = 0.744207(s)
Epoch [10/30], train loss = 0.328827, train accuracy = 98.61%, valid loss = 0.347557, valid accuracy = 96.59%, duration = 0.761326(s)
Epoch [11/30], train loss = 0.327443, train accuracy = 98.79%, valid loss = 0.346492, valid accuracy = 96.59%, duration = 0.751198(s)
Epoch [12/30], train loss = 0.326638, train accuracy = 98.81%, valid loss = 0.347579, valid accuracy = 96.41%, duration = 0.764035(s)
Epoch [13/30], train loss = 0.326044, train accuracy = 98.86%, valid loss = 0.347958, valid accuracy = 96.41%, duration = 0.762824(s)
Epoch [14/30], train loss = 0.325155, train accuracy = 98.92%, valid loss = 0.345662, valid accuracy = 96.77%, duration = 0.765924(s)
Epoch [15/30], train loss = 0.324923, train accuracy = 98.92%, valid loss = 0.346394, valid accuracy = 96.59%, duration = 0.747593(s)
Epoch [16/30], train loss = 0.324567, train accuracy = 98.94%, valid loss = 0.345308, valid accuracy = 96.77%, duration = 0.750947(s)
Epoch [17/30], train loss = 0.324110, train accuracy = 98.99%, valid loss = 0.345663, valid accuracy = 96.77%, duration = 0.761392(s)
Epoch [18/30], train loss = 0.324242, train accuracy = 98.97%, valid loss = 0.346165, valid accuracy = 96.59%, duration = 0.760586(s)
Epoch [19/30], train loss = 0.323883, train accuracy = 98.99%, valid loss = 0.345744, valid accuracy = 96.77%, duration = 0.751531(s)
Epoch [20/30], train loss = 0.324048, train accuracy = 98.97%, valid loss = 0.346095, valid accuracy = 96.59%, duration = 0.758950(s)
Epoch [21/30], train loss = 0.323977, train accuracy = 98.97%, valid loss = 0.345922, valid accuracy = 96.77%, duration = 0.767579(s)
Epoch [22/30], train loss = 0.323912, train accuracy = 98.97%, valid loss = 0.345716, valid accuracy = 96.77%, duration = 0.745300(s)
Epoch [23/30], train loss = 0.323603, train accuracy = 99.02%, valid loss = 0.345965, valid accuracy = 96.77%, duration = 0.770595(s)
Epoch [24/30], train loss = 0.323809, train accuracy = 98.99%, valid loss = 0.346129, valid accuracy = 96.59%, duration = 0.752945(s)
Epoch [25/30], train loss = 0.323776, train accuracy = 98.99%, valid loss = 0.345951, valid accuracy = 96.59%, duration = 0.755930(s)
Epoch [26/30], train loss = 0.323739, train accuracy = 98.99%, valid loss = 0.346082, valid accuracy = 96.59%, duration = 0.794657(s)
Epoch [27/30], train loss = 0.323708, train accuracy = 98.99%, valid loss = 0.346028, valid accuracy = 96.59%, duration = 0.757102(s)
Epoch [28/30], train loss = 0.323675, train accuracy = 98.99%, valid loss = 0.345514, valid accuracy = 96.77%, duration = 0.797148(s)
Epoch [29/30], train loss = 0.323816, train accuracy = 98.97%, valid loss = 0.345522, valid accuracy = 96.59%, duration = 0.757190(s)
Epoch [30/30], train loss = 0.323683, train accuracy = 98.99%, valid loss = 0.345254, valid accuracy = 96.59%, duration = 0.767322(s)
Duration for train : 23.501528(s)
<<< Train Finished >>>
Test Accraucy : 97.40%
'''
