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


def rnn_layer(tensor_op, activation=tf.nn.tanh, name="RNN_Layer"):
    with tf.variable_scope(name):
        cell = tf.contrib.rnn.BasicRNNCell(num_units=NUM_RNN_UNITS, activation=activation)
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
Epoch [ 1/30], train loss = 0.465632, train accuracy = 84.76%, valid loss = 0.488593, valid accuracy = 81.87%, duration = 0.654861(s)
Epoch [ 2/30], train loss = 0.450950, train accuracy = 86.23%, valid loss = 0.481166, valid accuracy = 83.12%, duration = 0.469807(s)
Epoch [ 3/30], train loss = 0.443921, train accuracy = 87.09%, valid loss = 0.477159, valid accuracy = 83.84%, duration = 0.439299(s)
Epoch [ 4/30], train loss = 0.439832, train accuracy = 87.47%, valid loss = 0.474711, valid accuracy = 83.84%, duration = 0.454218(s)
Epoch [ 5/30], train loss = 0.436733, train accuracy = 87.81%, valid loss = 0.473038, valid accuracy = 84.02%, duration = 0.444866(s)
Epoch [ 6/30], train loss = 0.432638, train accuracy = 88.12%, valid loss = 0.464846, valid accuracy = 84.20%, duration = 0.458310(s)
Epoch [ 7/30], train loss = 0.405316, train accuracy = 90.86%, valid loss = 0.390491, valid accuracy = 92.82%, duration = 0.512067(s)
Epoch [ 8/30], train loss = 0.374437, train accuracy = 93.93%, valid loss = 0.370262, valid accuracy = 94.61%, duration = 0.445016(s)
Epoch [ 9/30], train loss = 0.360213, train accuracy = 95.38%, valid loss = 0.367799, valid accuracy = 94.97%, duration = 0.478709(s)
Epoch [10/30], train loss = 0.354307, train accuracy = 95.95%, valid loss = 0.363806, valid accuracy = 95.15%, duration = 0.447401(s)
Epoch [11/30], train loss = 0.352379, train accuracy = 96.20%, valid loss = 0.362975, valid accuracy = 95.33%, duration = 0.499718(s)
Epoch [12/30], train loss = 0.349682, train accuracy = 96.41%, valid loss = 0.360645, valid accuracy = 95.33%, duration = 0.655139(s)
Epoch [13/30], train loss = 0.347821, train accuracy = 96.67%, valid loss = 0.361180, valid accuracy = 95.15%, duration = 0.572140(s)
Epoch [14/30], train loss = 0.346424, train accuracy = 96.75%, valid loss = 0.361166, valid accuracy = 94.97%, duration = 0.517355(s)
Epoch [15/30], train loss = 0.346981, train accuracy = 96.72%, valid loss = 0.360470, valid accuracy = 95.15%, duration = 0.660420(s)
Epoch [16/30], train loss = 0.344745, train accuracy = 96.90%, valid loss = 0.358419, valid accuracy = 95.69%, duration = 0.574936(s)
Epoch [17/30], train loss = 0.339821, train accuracy = 97.42%, valid loss = 0.360455, valid accuracy = 95.15%, duration = 0.523869(s)
Epoch [18/30], train loss = 0.343007, train accuracy = 97.00%, valid loss = 0.354924, valid accuracy = 95.87%, duration = 0.636345(s)
Epoch [19/30], train loss = 0.341410, train accuracy = 97.21%, valid loss = 0.359967, valid accuracy = 95.33%, duration = 0.567713(s)
Epoch [20/30], train loss = 0.339445, train accuracy = 97.44%, valid loss = 0.357053, valid accuracy = 95.69%, duration = 0.549353(s)
Epoch [21/30], train loss = 0.339283, train accuracy = 97.44%, valid loss = 0.353122, valid accuracy = 95.87%, duration = 0.617551(s)
Epoch [22/30], train loss = 0.340935, train accuracy = 97.26%, valid loss = 0.357436, valid accuracy = 95.69%, duration = 0.636549(s)
Epoch [23/30], train loss = 0.339280, train accuracy = 97.47%, valid loss = 0.352488, valid accuracy = 96.05%, duration = 0.737218(s)
Epoch [24/30], train loss = 0.337958, train accuracy = 97.57%, valid loss = 0.353391, valid accuracy = 96.05%, duration = 0.542734(s)
Epoch [25/30], train loss = 0.337132, train accuracy = 97.65%, valid loss = 0.353809, valid accuracy = 96.05%, duration = 0.576482(s)
Epoch [26/30], train loss = 0.338659, train accuracy = 97.49%, valid loss = 0.352916, valid accuracy = 96.05%, duration = 0.564378(s)
Epoch [27/30], train loss = 0.337856, train accuracy = 97.60%, valid loss = 0.354375, valid accuracy = 95.87%, duration = 0.505826(s)
Epoch [28/30], train loss = 0.338191, train accuracy = 97.55%, valid loss = 0.354212, valid accuracy = 96.05%, duration = 0.682861(s)
Epoch [29/30], train loss = 0.336669, train accuracy = 97.70%, valid loss = 0.353666, valid accuracy = 96.05%, duration = 0.531216(s)
Epoch [30/30], train loss = 0.336877, train accuracy = 97.68%, valid loss = 0.354049, valid accuracy = 96.05%, duration = 0.447371(s)
Duration for train : 16.774309(s)
<<< Train Finished >>>
Test Accraucy : 96.42%
'''
