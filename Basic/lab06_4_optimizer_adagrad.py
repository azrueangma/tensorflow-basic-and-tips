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
Epoch [ 1/30], train loss = 0.498613, train accuracy = 87.49%, valid loss = 0.316937, valid accuracy = 91.35%, duration = 2.207448(s)
Epoch [ 2/30], train loss = 0.301711, train accuracy = 91.64%, valid loss = 0.267345, valid accuracy = 92.67%, duration = 2.046917(s)
Epoch [ 3/30], train loss = 0.265786, train accuracy = 92.58%, valid loss = 0.241227, valid accuracy = 93.45%, duration = 2.042649(s)
Epoch [ 4/30], train loss = 0.244342, train accuracy = 93.17%, valid loss = 0.226068, valid accuracy = 93.92%, duration = 2.058137(s)
Epoch [ 5/30], train loss = 0.229401, train accuracy = 93.63%, valid loss = 0.213602, valid accuracy = 94.28%, duration = 2.055386(s)
Epoch [ 6/30], train loss = 0.217915, train accuracy = 93.94%, valid loss = 0.204360, valid accuracy = 94.35%, duration = 2.052276(s)
Epoch [ 7/30], train loss = 0.208286, train accuracy = 94.18%, valid loss = 0.197029, valid accuracy = 94.53%, duration = 2.037979(s)
Epoch [ 8/30], train loss = 0.200595, train accuracy = 94.36%, valid loss = 0.190361, valid accuracy = 94.87%, duration = 2.064999(s)
Epoch [ 9/30], train loss = 0.193798, train accuracy = 94.55%, valid loss = 0.185271, valid accuracy = 94.93%, duration = 2.036368(s)
Epoch [10/30], train loss = 0.187892, train accuracy = 94.73%, valid loss = 0.180250, valid accuracy = 95.05%, duration = 2.033277(s)
Epoch [11/30], train loss = 0.182761, train accuracy = 94.92%, valid loss = 0.175913, valid accuracy = 95.17%, duration = 2.066320(s)
Epoch [12/30], train loss = 0.177993, train accuracy = 94.99%, valid loss = 0.172080, valid accuracy = 95.27%, duration = 2.077037(s)
Epoch [13/30], train loss = 0.173593, train accuracy = 95.14%, valid loss = 0.168883, valid accuracy = 95.30%, duration = 2.277639(s)
Epoch [14/30], train loss = 0.169751, train accuracy = 95.26%, valid loss = 0.165278, valid accuracy = 95.52%, duration = 2.159889(s)
Epoch [15/30], train loss = 0.166234, train accuracy = 95.31%, valid loss = 0.163157, valid accuracy = 95.55%, duration = 2.108204(s)
Epoch [16/30], train loss = 0.162909, train accuracy = 95.39%, valid loss = 0.159864, valid accuracy = 95.62%, duration = 2.091923(s)
Epoch [17/30], train loss = 0.159690, train accuracy = 95.52%, valid loss = 0.157799, valid accuracy = 95.63%, duration = 2.137502(s)
Epoch [18/30], train loss = 0.156853, train accuracy = 95.60%, valid loss = 0.154813, valid accuracy = 95.77%, duration = 2.083616(s)
Epoch [19/30], train loss = 0.154252, train accuracy = 95.67%, valid loss = 0.152529, valid accuracy = 95.78%, duration = 2.080272(s)
Epoch [20/30], train loss = 0.151654, train accuracy = 95.76%, valid loss = 0.150503, valid accuracy = 96.00%, duration = 2.084138(s)
Epoch [21/30], train loss = 0.149287, train accuracy = 95.79%, valid loss = 0.148973, valid accuracy = 96.00%, duration = 2.068114(s)
Epoch [22/30], train loss = 0.147043, train accuracy = 95.89%, valid loss = 0.146918, valid accuracy = 95.93%, duration = 2.101123(s)
Epoch [23/30], train loss = 0.144811, train accuracy = 95.97%, valid loss = 0.145452, valid accuracy = 96.00%, duration = 2.085163(s)
Epoch [24/30], train loss = 0.142921, train accuracy = 95.99%, valid loss = 0.143621, valid accuracy = 96.00%, duration = 2.059473(s)
Epoch [25/30], train loss = 0.140911, train accuracy = 96.09%, valid loss = 0.141895, valid accuracy = 96.12%, duration = 2.079561(s)
Epoch [26/30], train loss = 0.139106, train accuracy = 96.15%, valid loss = 0.140526, valid accuracy = 96.10%, duration = 2.093593(s)
Epoch [27/30], train loss = 0.137277, train accuracy = 96.19%, valid loss = 0.139075, valid accuracy = 96.08%, duration = 2.087332(s)
Epoch [28/30], train loss = 0.135639, train accuracy = 96.26%, valid loss = 0.138109, valid accuracy = 96.13%, duration = 2.075807(s)
Epoch [29/30], train loss = 0.133947, train accuracy = 96.29%, valid loss = 0.136512, valid accuracy = 96.17%, duration = 2.096843(s)
Epoch [30/30], train loss = 0.132388, train accuracy = 96.39%, valid loss = 0.135892, valid accuracy = 96.15%, duration = 2.074825(s)
Duration for train : 63.548097(s)
<<< Train Finished >>>
Test Accraucy : 95.96%
'''
