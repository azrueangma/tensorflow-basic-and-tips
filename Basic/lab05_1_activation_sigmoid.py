import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-5_board"
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
Epoch [ 1/30], train loss = 0.392748, train accuracy = 89.02%, valid loss = 0.204767, valid accuracy = 94.00%, duration = 1.951825(s)
Epoch [ 2/30], train loss = 0.184482, train accuracy = 94.66%, valid loss = 0.150737, valid accuracy = 95.53%, duration = 1.893577(s)
Epoch [ 3/30], train loss = 0.133569, train accuracy = 96.18%, valid loss = 0.114333, valid accuracy = 96.47%, duration = 1.942605(s)
Epoch [ 4/30], train loss = 0.103837, train accuracy = 97.04%, valid loss = 0.104686, valid accuracy = 96.80%, duration = 1.877139(s)
Epoch [ 5/30], train loss = 0.084423, train accuracy = 97.56%, valid loss = 0.094070, valid accuracy = 97.07%, duration = 1.914646(s)
Epoch [ 6/30], train loss = 0.070685, train accuracy = 97.97%, valid loss = 0.084815, valid accuracy = 97.32%, duration = 1.871361(s)
Epoch [ 7/30], train loss = 0.059609, train accuracy = 98.28%, valid loss = 0.077115, valid accuracy = 97.58%, duration = 1.931546(s)
Epoch [ 8/30], train loss = 0.051065, train accuracy = 98.60%, valid loss = 0.074373, valid accuracy = 97.77%, duration = 2.059228(s)
Epoch [ 9/30], train loss = 0.043156, train accuracy = 98.80%, valid loss = 0.077159, valid accuracy = 97.60%, duration = 1.986854(s)
Epoch [10/30], train loss = 0.038293, train accuracy = 98.95%, valid loss = 0.067637, valid accuracy = 98.00%, duration = 2.022298(s)
Epoch [11/30], train loss = 0.032489, train accuracy = 99.17%, valid loss = 0.069274, valid accuracy = 97.83%, duration = 1.982176(s)
Epoch [12/30], train loss = 0.028579, train accuracy = 99.27%, valid loss = 0.067888, valid accuracy = 97.85%, duration = 2.039181(s)
Epoch [13/30], train loss = 0.024287, train accuracy = 99.44%, valid loss = 0.069031, valid accuracy = 97.80%, duration = 1.981762(s)
Epoch [14/30], train loss = 0.020861, train accuracy = 99.54%, valid loss = 0.062350, valid accuracy = 98.22%, duration = 1.938181(s)
Epoch [15/30], train loss = 0.018319, train accuracy = 99.64%, valid loss = 0.064362, valid accuracy = 98.05%, duration = 1.947496(s)
Epoch [16/30], train loss = 0.015415, train accuracy = 99.71%, valid loss = 0.062098, valid accuracy = 98.22%, duration = 2.162339(s)
Epoch [17/30], train loss = 0.013760, train accuracy = 99.76%, valid loss = 0.067525, valid accuracy = 97.87%, duration = 1.968466(s)
Epoch [18/30], train loss = 0.011807, train accuracy = 99.81%, valid loss = 0.066973, valid accuracy = 98.00%, duration = 1.952567(s)
Epoch [19/30], train loss = 0.010390, train accuracy = 99.84%, valid loss = 0.064263, valid accuracy = 98.02%, duration = 1.958046(s)
Epoch [20/30], train loss = 0.009053, train accuracy = 99.88%, valid loss = 0.064352, valid accuracy = 98.08%, duration = 1.934902(s)
Epoch [21/30], train loss = 0.007943, train accuracy = 99.91%, valid loss = 0.063609, valid accuracy = 98.10%, duration = 1.891580(s)
Epoch [22/30], train loss = 0.006844, train accuracy = 99.93%, valid loss = 0.065340, valid accuracy = 98.03%, duration = 1.896939(s)
Epoch [23/30], train loss = 0.005881, train accuracy = 99.96%, valid loss = 0.066473, valid accuracy = 98.17%, duration = 1.887852(s)
Epoch [24/30], train loss = 0.005257, train accuracy = 99.96%, valid loss = 0.064827, valid accuracy = 98.07%, duration = 1.920076(s)
Epoch [25/30], train loss = 0.004619, train accuracy = 99.97%, valid loss = 0.065732, valid accuracy = 98.10%, duration = 1.937090(s)
Epoch [26/30], train loss = 0.004145, train accuracy = 99.99%, valid loss = 0.066761, valid accuracy = 98.10%, duration = 1.863111(s)
Epoch [27/30], train loss = 0.003782, train accuracy = 99.98%, valid loss = 0.066868, valid accuracy = 98.07%, duration = 1.881350(s)
Epoch [28/30], train loss = 0.003294, train accuracy = 99.99%, valid loss = 0.066709, valid accuracy = 98.12%, duration = 2.085801(s)
Epoch [29/30], train loss = 0.003073, train accuracy = 99.99%, valid loss = 0.067742, valid accuracy = 97.98%, duration = 2.035180(s)
Epoch [30/30], train loss = 0.002729, train accuracy = 100.00%, valid loss = 0.067841, valid accuracy = 98.08%, duration = 2.064063(s)
Duration for train : 59.731889(s)
<<< Train Finished >>>
Test Accraucy : 98.07%
'''
