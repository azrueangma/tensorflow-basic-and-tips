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


with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape=[None, MAX_SEQUENCE_LENGTH], dtype=tf.int32, name='X')
    Y = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name='Y_one_hot')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

embedding_mat = tf.get_variable(name='embedding', shape=[VOCAB_SIZE, EMBEDDING_SIZE],
                                initializer=tf.random_normal_initializer(stddev=1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, X)

cell = tf.contrib.rnn.BasicRNNCell(num_units = NUM_RNN_UNITS)
output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.transpose(output, [1, 0, 2])
last = tf.gather(output, int(output.get_shape()[0])-1)

h1 = linear(tensor_op=last, output_dim=2, name='FCLayer1')
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
Epoch [ 1/30], train loss = 0.502206, train accuracy = 82.33%, valid loss = 0.495809, valid accuracy = 81.51%, duration = 0.584092(s)
Epoch [ 2/30], train loss = 0.464968, train accuracy = 85.36%, valid loss = 0.486965, valid accuracy = 82.59%, duration = 0.417329(s)
Epoch [ 3/30], train loss = 0.455104, train accuracy = 86.26%, valid loss = 0.480355, valid accuracy = 83.48%, duration = 0.423419(s)
Epoch [ 4/30], train loss = 0.448747, train accuracy = 86.65%, valid loss = 0.476055, valid accuracy = 84.02%, duration = 0.424771(s)
Epoch [ 5/30], train loss = 0.444557, train accuracy = 86.91%, valid loss = 0.472758, valid accuracy = 84.20%, duration = 0.416275(s)
Epoch [ 6/30], train loss = 0.440424, train accuracy = 87.53%, valid loss = 0.470258, valid accuracy = 84.56%, duration = 0.419210(s)
Epoch [ 7/30], train loss = 0.436919, train accuracy = 87.81%, valid loss = 0.467729, valid accuracy = 84.56%, duration = 0.431674(s)
Epoch [ 8/30], train loss = 0.433233, train accuracy = 88.17%, valid loss = 0.461816, valid accuracy = 85.10%, duration = 0.424159(s)
Epoch [ 9/30], train loss = 0.420008, train accuracy = 89.05%, valid loss = 0.442589, valid accuracy = 86.36%, duration = 0.430879(s)
Epoch [10/30], train loss = 0.384974, train accuracy = 93.03%, valid loss = 0.388007, valid accuracy = 92.46%, duration = 0.414749(s)
Epoch [11/30], train loss = 0.371591, train accuracy = 94.34%, valid loss = 0.413718, valid accuracy = 90.31%, duration = 0.426602(s)
Epoch [12/30], train loss = 0.379638, train accuracy = 93.60%, valid loss = 0.392344, valid accuracy = 92.10%, duration = 0.417529(s)
Epoch [13/30], train loss = 0.392708, train accuracy = 92.15%, valid loss = 0.389698, valid accuracy = 92.64%, duration = 0.426490(s)
Epoch [14/30], train loss = 0.384194, train accuracy = 92.92%, valid loss = 0.421856, valid accuracy = 89.05%, duration = 0.419066(s)
Epoch [15/30], train loss = 0.385186, train accuracy = 92.90%, valid loss = 0.406740, valid accuracy = 90.48%, duration = 0.430515(s)
Epoch [16/30], train loss = 0.408362, train accuracy = 90.50%, valid loss = 0.434029, valid accuracy = 87.79%, duration = 0.420864(s)
Epoch [17/30], train loss = 0.392789, train accuracy = 92.07%, valid loss = 0.409683, valid accuracy = 90.13%, duration = 0.420899(s)
Epoch [18/30], train loss = 0.380906, train accuracy = 93.29%, valid loss = 0.385970, valid accuracy = 92.28%, duration = 0.425250(s)
Epoch [19/30], train loss = 0.358766, train accuracy = 95.58%, valid loss = 0.385818, valid accuracy = 92.82%, duration = 0.489952(s)
Epoch [20/30], train loss = 0.367105, train accuracy = 94.71%, valid loss = 0.394282, valid accuracy = 91.74%, duration = 0.432757(s)
Epoch [21/30], train loss = 0.371827, train accuracy = 94.24%, valid loss = 0.399128, valid accuracy = 91.38%, duration = 0.469694(s)
Epoch [22/30], train loss = 0.380695, train accuracy = 93.31%, valid loss = 0.398887, valid accuracy = 91.38%, duration = 0.437583(s)
Epoch [23/30], train loss = 0.368911, train accuracy = 94.50%, valid loss = 0.390305, valid accuracy = 92.10%, duration = 0.453881(s)
Epoch [24/30], train loss = 0.368750, train accuracy = 94.50%, valid loss = 0.391572, valid accuracy = 91.92%, duration = 0.459434(s)
Epoch [25/30], train loss = 0.369343, train accuracy = 94.42%, valid loss = 0.394975, valid accuracy = 91.56%, duration = 0.466690(s)
Epoch [26/30], train loss = 0.363942, train accuracy = 94.96%, valid loss = 0.402869, valid accuracy = 91.02%, duration = 0.469521(s)
Epoch [27/30], train loss = 0.383216, train accuracy = 93.00%, valid loss = 0.406432, valid accuracy = 90.48%, duration = 0.445495(s)
Epoch [28/30], train loss = 0.384653, train accuracy = 92.87%, valid loss = 0.408893, valid accuracy = 90.31%, duration = 0.448472(s)
Epoch [29/30], train loss = 0.370716, train accuracy = 94.27%, valid loss = 0.392377, valid accuracy = 91.92%, duration = 0.451763(s)
Epoch [30/30], train loss = 0.369026, train accuracy = 94.45%, valid loss = 0.394747, valid accuracy = 91.74%, duration = 0.412313(s)
Duration for train : 13.537435(s)
<<< Train Finished >>>
Test Accraucy : 92.92%
'''