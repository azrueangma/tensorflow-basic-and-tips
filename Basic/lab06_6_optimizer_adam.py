import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-6_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)

def linear(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def relu_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = relu_layer(X, 256, 'Relu_Layer1')
h2 = relu_layer(h1, 128, 'Relu_Layer2')
logits = linear(h2, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    optim = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

with tf.variable_scope("Prediction"):
    predict = tf.argmax(hypothesis, axis=1)

with tf.variable_scope("Accuracy"):
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, tf.argmax(Y_one_hot, axis = 1)), tf.float32))

with tf.variable_scope("Summary"):
    avg_train_loss = tf.placeholder(tf.float32)
    loss_train_avg  = tf.summary.scalar('avg_train_loss', avg_train_loss)
    avg_train_acc = tf.placeholder(tf.float32)
    acc_train_avg = tf.summary.scalar('avg_train_acc', avg_train_acc)
    avg_validation_loss = tf.placeholder(tf.float32)
    loss_validation_avg = tf.summary.scalar('avg_validation_loss', avg_validation_loss)
    avg_validation_acc = tf.placeholder(tf.float32)
    acc_validation_avg = tf.summary.scalar('avg_validation_acc', avg_validation_acc)
    merged = tf.summary.merge_all()

init_op = tf.global_variables_initializer()
total_step = int(ntrain/BATCH_SIZE)
print("Total step : ", total_step)
with tf.Session() as sess:
    if os.path.exists(BOARD_PATH):
        shutil.rmtree(BOARD_PATH)
    writer = tf.summary.FileWriter(BOARD_PATH)
    writer.add_graph(sess.graph)

    sess.run(init_op)

    train_start_time = time.perf_counter()
    for epoch in range(TOTAL_EPOCH):
        loss_per_epoch = 0
        acc_per_epoch = 0

        np.random.seed(epoch)
        mask = np.random.permutation(len(x_train))

        epoch_start_time = time.perf_counter()
        for step in range(total_step):
            s = BATCH_SIZE*step
            t = BATCH_SIZE*(step+1)
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t],:], Y: y_train[mask[s:t],:]})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step*BATCH_SIZE
        acc_per_epoch /= total_step*BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation})
        epoch_valid_acc = va / len(x_validation)
        epoch_valid_loss = vl / len(x_validation)

        s = sess.run(merged, feed_dict = {avg_train_loss:loss_per_epoch, avg_train_acc:acc_per_epoch,
                                          avg_validation_loss:epoch_valid_loss, avg_validation_acc:epoch_valid_acc})
        writer.add_summary(s, global_step = epoch)

        if (epoch+1) % 1 == 0:

            print("Epoch [{:2d}/{:2d}], train loss = {:.6f}, train accuracy = {:.2%}, "
                  "valid loss = {:.6f}, valid accuracy = {:.2%}, duration = {:.6f}(s)"
                  .format(epoch + 1, TOTAL_EPOCH, loss_per_epoch, acc_per_epoch, epoch_valid_loss, epoch_valid_acc, epoch_duration))

    train_end_time = time.perf_counter()
    train_duration = train_end_time - train_start_time
    print("Duration for train : {:.6f}(s)".format(train_duration))
    print("<<< Train Finished >>>")

    ta = sess.run(accuracy, feed_dict = {X:x_test, Y:y_test})
    print("Test Accraucy : {:.2%}".format(ta/ntest))

'''
Epoch [ 1/30], train loss = 0.215350, train accuracy = 93.61%, valid loss = 0.106351, valid accuracy = 96.78%, duration = 2.931922(s)
Epoch [ 2/30], train loss = 0.086820, train accuracy = 97.37%, valid loss = 0.090322, valid accuracy = 97.25%, duration = 2.718806(s)
Epoch [ 3/30], train loss = 0.059870, train accuracy = 98.15%, valid loss = 0.083493, valid accuracy = 97.73%, duration = 3.522585(s)
Epoch [ 4/30], train loss = 0.044059, train accuracy = 98.57%, valid loss = 0.091363, valid accuracy = 97.58%, duration = 2.677199(s)
Epoch [ 5/30], train loss = 0.034477, train accuracy = 98.88%, valid loss = 0.085602, valid accuracy = 97.82%, duration = 2.920313(s)
Epoch [ 6/30], train loss = 0.029261, train accuracy = 99.09%, valid loss = 0.073681, valid accuracy = 97.97%, duration = 2.321530(s)
Epoch [ 7/30], train loss = 0.025332, train accuracy = 99.21%, valid loss = 0.100420, valid accuracy = 97.35%, duration = 2.547514(s)
Epoch [ 8/30], train loss = 0.021600, train accuracy = 99.23%, valid loss = 0.091599, valid accuracy = 97.67%, duration = 2.361494(s)
Epoch [ 9/30], train loss = 0.017314, train accuracy = 99.42%, valid loss = 0.083884, valid accuracy = 97.87%, duration = 2.454028(s)
Epoch [10/30], train loss = 0.014752, train accuracy = 99.53%, valid loss = 0.090788, valid accuracy = 97.95%, duration = 2.338350(s)
Epoch [11/30], train loss = 0.017574, train accuracy = 99.45%, valid loss = 0.101866, valid accuracy = 97.73%, duration = 2.430014(s)
Epoch [12/30], train loss = 0.013788, train accuracy = 99.57%, valid loss = 0.112398, valid accuracy = 97.82%, duration = 2.362656(s)
Epoch [13/30], train loss = 0.015321, train accuracy = 99.50%, valid loss = 0.102829, valid accuracy = 98.12%, duration = 2.475057(s)
Epoch [14/30], train loss = 0.011920, train accuracy = 99.64%, valid loss = 0.094639, valid accuracy = 97.93%, duration = 2.315481(s)
Epoch [15/30], train loss = 0.010887, train accuracy = 99.65%, valid loss = 0.104724, valid accuracy = 98.03%, duration = 2.452630(s)
Epoch [16/30], train loss = 0.011361, train accuracy = 99.63%, valid loss = 0.128765, valid accuracy = 97.72%, duration = 2.302163(s)
Epoch [17/30], train loss = 0.012343, train accuracy = 99.62%, valid loss = 0.103037, valid accuracy = 98.10%, duration = 2.438337(s)
Epoch [18/30], train loss = 0.008342, train accuracy = 99.73%, valid loss = 0.111229, valid accuracy = 98.02%, duration = 2.393246(s)
Epoch [19/30], train loss = 0.010932, train accuracy = 99.68%, valid loss = 0.103243, valid accuracy = 98.18%, duration = 2.533140(s)
Epoch [20/30], train loss = 0.008796, train accuracy = 99.73%, valid loss = 0.125116, valid accuracy = 97.82%, duration = 2.403552(s)
Epoch [21/30], train loss = 0.008723, train accuracy = 99.75%, valid loss = 0.103914, valid accuracy = 98.30%, duration = 2.520469(s)
Epoch [22/30], train loss = 0.009324, train accuracy = 99.75%, valid loss = 0.135413, valid accuracy = 97.85%, duration = 2.367390(s)
Epoch [23/30], train loss = 0.010543, train accuracy = 99.68%, valid loss = 0.133185, valid accuracy = 98.12%, duration = 2.442824(s)
Epoch [24/30], train loss = 0.008352, train accuracy = 99.77%, valid loss = 0.113460, valid accuracy = 98.27%, duration = 2.355825(s)
Epoch [25/30], train loss = 0.008537, train accuracy = 99.73%, valid loss = 0.125984, valid accuracy = 98.15%, duration = 2.842325(s)
Epoch [26/30], train loss = 0.008221, train accuracy = 99.78%, valid loss = 0.125521, valid accuracy = 98.00%, duration = 2.298555(s)
Epoch [27/30], train loss = 0.007750, train accuracy = 99.73%, valid loss = 0.128022, valid accuracy = 98.03%, duration = 2.564007(s)
Epoch [28/30], train loss = 0.008960, train accuracy = 99.75%, valid loss = 0.127833, valid accuracy = 97.98%, duration = 2.613819(s)
Epoch [29/30], train loss = 0.006019, train accuracy = 99.84%, valid loss = 0.130919, valid accuracy = 97.92%, duration = 2.535497(s)
Epoch [30/30], train loss = 0.007679, train accuracy = 99.79%, valid loss = 0.151782, valid accuracy = 98.10%, duration = 2.409834(s)
Duration for train : 76.789371(s)
<<< Train Finished >>>
Test Accraucy : 97.98%
'''
