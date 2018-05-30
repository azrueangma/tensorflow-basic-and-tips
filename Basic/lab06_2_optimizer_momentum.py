import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-2_board"
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
    optim = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.1).minimize(loss)

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
Epoch [ 1/30], train loss = 0.378397, train accuracy = 89.34%, valid loss = 0.197108, valid accuracy = 94.12%, duration = 2.397975(s)
Epoch [ 2/30], train loss = 0.175017, train accuracy = 94.92%, valid loss = 0.143032, valid accuracy = 95.77%, duration = 2.125117(s)
Epoch [ 3/30], train loss = 0.125044, train accuracy = 96.41%, valid loss = 0.109329, valid accuracy = 96.77%, duration = 2.168108(s)
Epoch [ 4/30], train loss = 0.096642, train accuracy = 97.22%, valid loss = 0.100958, valid accuracy = 96.85%, duration = 2.592238(s)
Epoch [ 5/30], train loss = 0.078327, train accuracy = 97.74%, valid loss = 0.090872, valid accuracy = 97.17%, duration = 2.464637(s)
Epoch [ 6/30], train loss = 0.065225, train accuracy = 98.13%, valid loss = 0.081171, valid accuracy = 97.38%, duration = 2.263136(s)
Epoch [ 7/30], train loss = 0.054602, train accuracy = 98.48%, valid loss = 0.074685, valid accuracy = 97.68%, duration = 2.325969(s)
Epoch [ 8/30], train loss = 0.046540, train accuracy = 98.71%, valid loss = 0.072295, valid accuracy = 97.72%, duration = 2.186517(s)
Epoch [ 9/30], train loss = 0.038933, train accuracy = 98.93%, valid loss = 0.076373, valid accuracy = 97.60%, duration = 2.197264(s)
Epoch [10/30], train loss = 0.034327, train accuracy = 99.10%, valid loss = 0.065705, valid accuracy = 98.05%, duration = 2.014360(s)
Epoch [11/30], train loss = 0.028708, train accuracy = 99.27%, valid loss = 0.066996, valid accuracy = 97.90%, duration = 2.009959(s)
Epoch [12/30], train loss = 0.025068, train accuracy = 99.36%, valid loss = 0.067961, valid accuracy = 97.80%, duration = 2.013086(s)
Epoch [13/30], train loss = 0.021068, train accuracy = 99.52%, valid loss = 0.069390, valid accuracy = 97.72%, duration = 2.195876(s)
Epoch [14/30], train loss = 0.017680, train accuracy = 99.62%, valid loss = 0.062258, valid accuracy = 98.27%, duration = 2.248431(s)
Epoch [15/30], train loss = 0.015255, train accuracy = 99.70%, valid loss = 0.064107, valid accuracy = 97.88%, duration = 2.067746(s)
Epoch [16/30], train loss = 0.012717, train accuracy = 99.77%, valid loss = 0.061555, valid accuracy = 98.17%, duration = 2.088743(s)
Epoch [17/30], train loss = 0.011449, train accuracy = 99.82%, valid loss = 0.067219, valid accuracy = 97.93%, duration = 2.152746(s)
Epoch [18/30], train loss = 0.009458, train accuracy = 99.87%, valid loss = 0.065464, valid accuracy = 98.10%, duration = 2.232762(s)
Epoch [19/30], train loss = 0.008094, train accuracy = 99.90%, valid loss = 0.064060, valid accuracy = 97.92%, duration = 2.219288(s)
Epoch [20/30], train loss = 0.007088, train accuracy = 99.92%, valid loss = 0.063672, valid accuracy = 98.22%, duration = 2.145683(s)
Epoch [21/30], train loss = 0.006156, train accuracy = 99.94%, valid loss = 0.063025, valid accuracy = 98.08%, duration = 2.231888(s)
Epoch [22/30], train loss = 0.005157, train accuracy = 99.96%, valid loss = 0.064111, valid accuracy = 98.08%, duration = 2.186918(s)
Epoch [23/30], train loss = 0.004391, train accuracy = 99.98%, valid loss = 0.065788, valid accuracy = 98.18%, duration = 2.476418(s)
Epoch [24/30], train loss = 0.003865, train accuracy = 99.99%, valid loss = 0.063976, valid accuracy = 98.18%, duration = 2.398896(s)
Epoch [25/30], train loss = 0.003503, train accuracy = 99.99%, valid loss = 0.065274, valid accuracy = 98.15%, duration = 2.091249(s)
Epoch [26/30], train loss = 0.003090, train accuracy = 99.99%, valid loss = 0.066764, valid accuracy = 97.98%, duration = 2.010595(s)
Epoch [27/30], train loss = 0.002836, train accuracy = 99.99%, valid loss = 0.066045, valid accuracy = 98.00%, duration = 2.072700(s)
Epoch [28/30], train loss = 0.002540, train accuracy = 99.99%, valid loss = 0.065804, valid accuracy = 98.13%, duration = 2.039392(s)
Epoch [29/30], train loss = 0.002399, train accuracy = 99.99%, valid loss = 0.066777, valid accuracy = 98.02%, duration = 2.101148(s)
Epoch [30/30], train loss = 0.002128, train accuracy = 100.00%, valid loss = 0.067340, valid accuracy = 98.02%, duration = 2.025704(s)
Duration for train : 66.716937(s)
<<< Train Finished >>>
Test Accraucy : 98.14%
'''
