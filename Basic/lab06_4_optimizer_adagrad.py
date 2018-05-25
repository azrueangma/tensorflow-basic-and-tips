import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-4_board"
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
    optim = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(loss)

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
Epoch [ 1/30], train loss = 0.492277, train accuracy = 87.98%, valid loss = 0.312917, valid accuracy = 91.57%, duration = 2.245253(s)
Epoch [ 2/30], train loss = 0.298068, train accuracy = 91.77%, valid loss = 0.264933, valid accuracy = 92.70%, duration = 2.155810(s)
Epoch [ 3/30], train loss = 0.263305, train accuracy = 92.68%, valid loss = 0.239879, valid accuracy = 93.50%, duration = 2.192810(s)
Epoch [ 4/30], train loss = 0.242808, train accuracy = 93.26%, valid loss = 0.225481, valid accuracy = 93.93%, duration = 2.212288(s)
Epoch [ 5/30], train loss = 0.228398, train accuracy = 93.61%, valid loss = 0.213383, valid accuracy = 94.30%, duration = 2.478765(s)
Epoch [ 6/30], train loss = 0.217218, train accuracy = 93.91%, valid loss = 0.204582, valid accuracy = 94.43%, duration = 2.353402(s)
Epoch [ 7/30], train loss = 0.207847, train accuracy = 94.19%, valid loss = 0.197594, valid accuracy = 94.65%, duration = 2.100367(s)
Epoch [ 8/30], train loss = 0.200307, train accuracy = 94.39%, valid loss = 0.190755, valid accuracy = 94.73%, duration = 2.161755(s)
Epoch [ 9/30], train loss = 0.193566, train accuracy = 94.62%, valid loss = 0.185827, valid accuracy = 94.88%, duration = 2.095429(s)
Epoch [10/30], train loss = 0.187708, train accuracy = 94.75%, valid loss = 0.180767, valid accuracy = 94.88%, duration = 2.219293(s)
Epoch [11/30], train loss = 0.182624, train accuracy = 94.88%, valid loss = 0.176458, valid accuracy = 94.92%, duration = 2.201566(s)
Epoch [12/30], train loss = 0.177896, train accuracy = 95.02%, valid loss = 0.172683, valid accuracy = 95.03%, duration = 2.052592(s)
Epoch [13/30], train loss = 0.173489, train accuracy = 95.11%, valid loss = 0.169321, valid accuracy = 95.08%, duration = 2.079079(s)
Epoch [14/30], train loss = 0.169644, train accuracy = 95.22%, valid loss = 0.165948, valid accuracy = 95.25%, duration = 2.104238(s)
Epoch [15/30], train loss = 0.166070, train accuracy = 95.30%, valid loss = 0.163521, valid accuracy = 95.32%, duration = 2.181695(s)
Epoch [16/30], train loss = 0.162708, train accuracy = 95.46%, valid loss = 0.160361, valid accuracy = 95.50%, duration = 2.167028(s)
Epoch [17/30], train loss = 0.159463, train accuracy = 95.56%, valid loss = 0.157742, valid accuracy = 95.38%, duration = 2.177746(s)
Epoch [18/30], train loss = 0.156566, train accuracy = 95.66%, valid loss = 0.155039, valid accuracy = 95.57%, duration = 2.400852(s)
Epoch [19/30], train loss = 0.153931, train accuracy = 95.68%, valid loss = 0.152919, valid accuracy = 95.73%, duration = 2.360259(s)
Epoch [20/30], train loss = 0.151288, train accuracy = 95.77%, valid loss = 0.150889, valid accuracy = 95.75%, duration = 2.174369(s)
Epoch [21/30], train loss = 0.148852, train accuracy = 95.86%, valid loss = 0.149222, valid accuracy = 95.80%, duration = 2.279370(s)
Epoch [22/30], train loss = 0.146592, train accuracy = 95.89%, valid loss = 0.147173, valid accuracy = 95.92%, duration = 2.324988(s)
Epoch [23/30], train loss = 0.144273, train accuracy = 95.96%, valid loss = 0.145796, valid accuracy = 95.85%, duration = 2.176259(s)
Epoch [24/30], train loss = 0.142374, train accuracy = 96.03%, valid loss = 0.143794, valid accuracy = 95.92%, duration = 2.056428(s)
Epoch [25/30], train loss = 0.140317, train accuracy = 96.07%, valid loss = 0.142112, valid accuracy = 96.00%, duration = 2.069946(s)
Epoch [26/30], train loss = 0.138445, train accuracy = 96.14%, valid loss = 0.140757, valid accuracy = 96.07%, duration = 2.052994(s)
Epoch [27/30], train loss = 0.136610, train accuracy = 96.18%, valid loss = 0.139314, valid accuracy = 96.12%, duration = 2.084365(s)
Epoch [28/30], train loss = 0.134909, train accuracy = 96.23%, valid loss = 0.138147, valid accuracy = 96.13%, duration = 2.082951(s)
Epoch [29/30], train loss = 0.133181, train accuracy = 96.29%, valid loss = 0.136529, valid accuracy = 96.20%, duration = 2.056093(s)
Epoch [30/30], train loss = 0.131595, train accuracy = 96.33%, valid loss = 0.135759, valid accuracy = 96.28%, duration = 2.103203(s)
Duration for train : 66.377276(s)
<<< Train Finished >>>
Test Accraucy : 95.99%
'''
