import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-4_board"
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
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def leaky_relu_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.leaky_relu(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = leaky_relu_layer(X, 256, 'Relu_Layer1')
h2 = leaky_relu_layer(h1, 128, 'Relu_Layer2')
logits = linear(h2, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

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
Epoch [ 1/30], train loss = 13.614806, train accuracy = 87.24%, valid loss = 3.288014, valid accuracy = 92.17%, duration = 2.488672(s)
Epoch [ 2/30], train loss = 2.687083, train accuracy = 92.28%, valid loss = 2.336443, valid accuracy = 92.62%, duration = 2.046392(s)
Epoch [ 3/30], train loss = 1.678395, train accuracy = 93.61%, valid loss = 1.880088, valid accuracy = 93.55%, duration = 2.011256(s)
Epoch [ 4/30], train loss = 1.272709, train accuracy = 94.42%, valid loss = 1.615157, valid accuracy = 93.37%, duration = 2.016850(s)
Epoch [ 5/30], train loss = 0.991803, train accuracy = 95.08%, valid loss = 1.707514, valid accuracy = 92.98%, duration = 2.016667(s)
Epoch [ 6/30], train loss = 0.805703, train accuracy = 95.75%, valid loss = 1.372004, valid accuracy = 94.25%, duration = 2.012291(s)
Epoch [ 7/30], train loss = 0.672681, train accuracy = 96.10%, valid loss = 1.356274, valid accuracy = 94.63%, duration = 2.006246(s)
Epoch [ 8/30], train loss = 0.585571, train accuracy = 96.58%, valid loss = 1.191707, valid accuracy = 95.02%, duration = 2.021147(s)
Epoch [ 9/30], train loss = 0.482270, train accuracy = 96.89%, valid loss = 1.325296, valid accuracy = 94.68%, duration = 2.019766(s)
Epoch [10/30], train loss = 0.418534, train accuracy = 97.20%, valid loss = 1.134018, valid accuracy = 95.33%, duration = 2.000443(s)
Epoch [11/30], train loss = 0.353646, train accuracy = 97.58%, valid loss = 1.303328, valid accuracy = 94.72%, duration = 2.038751(s)
Epoch [12/30], train loss = 0.315384, train accuracy = 97.63%, valid loss = 1.198162, valid accuracy = 95.30%, duration = 1.999560(s)
Epoch [13/30], train loss = 0.279684, train accuracy = 97.86%, valid loss = 1.093159, valid accuracy = 95.57%, duration = 2.017409(s)
Epoch [14/30], train loss = 0.239981, train accuracy = 98.15%, valid loss = 1.092113, valid accuracy = 95.70%, duration = 2.033028(s)
Epoch [15/30], train loss = 0.220484, train accuracy = 98.22%, valid loss = 1.253691, valid accuracy = 95.37%, duration = 2.009068(s)
Epoch [16/30], train loss = 0.177941, train accuracy = 98.51%, valid loss = 1.316005, valid accuracy = 95.25%, duration = 2.013316(s)
Epoch [17/30], train loss = 0.173792, train accuracy = 98.48%, valid loss = 1.251598, valid accuracy = 95.42%, duration = 2.014124(s)
Epoch [18/30], train loss = 0.145550, train accuracy = 98.70%, valid loss = 1.235831, valid accuracy = 95.48%, duration = 2.182291(s)
Epoch [19/30], train loss = 0.114173, train accuracy = 98.93%, valid loss = 1.194559, valid accuracy = 95.97%, duration = 2.042378(s)
Epoch [20/30], train loss = 0.110366, train accuracy = 98.94%, valid loss = 1.192349, valid accuracy = 95.85%, duration = 2.070725(s)
Epoch [21/30], train loss = 0.104303, train accuracy = 99.00%, valid loss = 1.165619, valid accuracy = 95.95%, duration = 2.027915(s)
Epoch [22/30], train loss = 0.078864, train accuracy = 99.14%, valid loss = 1.145740, valid accuracy = 96.07%, duration = 2.016712(s)
Epoch [23/30], train loss = 0.079397, train accuracy = 99.16%, valid loss = 1.204915, valid accuracy = 95.93%, duration = 2.034401(s)
Epoch [24/30], train loss = 0.069963, train accuracy = 99.25%, valid loss = 1.123080, valid accuracy = 96.32%, duration = 1.994460(s)
Epoch [25/30], train loss = 0.058037, train accuracy = 99.34%, valid loss = 1.101357, valid accuracy = 96.38%, duration = 2.009870(s)
Epoch [26/30], train loss = 0.039002, train accuracy = 99.53%, valid loss = 1.178153, valid accuracy = 96.13%, duration = 2.018315(s)
Epoch [27/30], train loss = 0.035446, train accuracy = 99.56%, valid loss = 1.316463, valid accuracy = 95.77%, duration = 1.989244(s)
Epoch [28/30], train loss = 0.031456, train accuracy = 99.60%, valid loss = 1.257887, valid accuracy = 96.20%, duration = 2.018757(s)
Epoch [29/30], train loss = 0.034304, train accuracy = 99.60%, valid loss = 1.265546, valid accuracy = 95.85%, duration = 2.030413(s)
Epoch [30/30], train loss = 0.031104, train accuracy = 99.61%, valid loss = 1.290509, valid accuracy = 96.17%, duration = 2.175666(s)
Duration for train : 62.462923(s)
<<< Train Finished >>>
Test Accraucy : 95.65%
'''
