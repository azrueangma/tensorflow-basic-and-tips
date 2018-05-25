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

TOTAL_EPOCH = 20

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
h3 = relu_layer(h2, 64, 'Relu_Layer3')
logits = linear(h1, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    optim = tf.train.MomentumOptimizer(learning_rate = 0.001, momentum = 0.1).minimize(loss)

with tf.variable_scope("Pred_and_Acc"):
    predict = tf.argmax(hypothesis, axis=1)
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
Total step :  1687
Epoch [ 1/20], train loss = 0.410928, train accuracy = 88.94%, valid loss = 0.249082, valid accuracy = 92.97%, duration = 1.586521(s)
Epoch [ 2/20], train loss = 0.232273, train accuracy = 93.40%, valid loss = 0.189742, valid accuracy = 94.82%, duration = 1.645492(s)
Epoch [ 3/20], train loss = 0.180928, train accuracy = 94.90%, valid loss = 0.151483, valid accuracy = 95.80%, duration = 1.658888(s)
Epoch [ 4/20], train loss = 0.147942, train accuracy = 95.82%, valid loss = 0.134953, valid accuracy = 96.28%, duration = 1.542642(s)
Epoch [ 5/20], train loss = 0.125062, train accuracy = 96.58%, valid loss = 0.121068, valid accuracy = 96.42%, duration = 1.765895(s)
Epoch [ 6/20], train loss = 0.108751, train accuracy = 97.01%, valid loss = 0.107464, valid accuracy = 96.82%, duration = 1.786386(s)
Epoch [ 7/20], train loss = 0.095497, train accuracy = 97.40%, valid loss = 0.102065, valid accuracy = 96.93%, duration = 1.542889(s)
Epoch [ 8/20], train loss = 0.085541, train accuracy = 97.65%, valid loss = 0.095269, valid accuracy = 97.15%, duration = 1.670200(s)
Epoch [ 9/20], train loss = 0.076736, train accuracy = 97.89%, valid loss = 0.091265, valid accuracy = 97.30%, duration = 1.537091(s)
Epoch [10/20], train loss = 0.070040, train accuracy = 98.11%, valid loss = 0.082302, valid accuracy = 97.57%, duration = 1.612395(s)
Epoch [11/20], train loss = 0.063974, train accuracy = 98.30%, valid loss = 0.082550, valid accuracy = 97.35%, duration = 1.819102(s)
Epoch [12/20], train loss = 0.058683, train accuracy = 98.47%, valid loss = 0.076842, valid accuracy = 97.65%, duration = 1.604003(s)
Epoch [13/20], train loss = 0.054037, train accuracy = 98.56%, valid loss = 0.076272, valid accuracy = 97.58%, duration = 1.560980(s)
Epoch [14/20], train loss = 0.049813, train accuracy = 98.71%, valid loss = 0.072958, valid accuracy = 97.75%, duration = 1.868105(s)
Epoch [15/20], train loss = 0.046263, train accuracy = 98.78%, valid loss = 0.073384, valid accuracy = 97.63%, duration = 1.600996(s)
Epoch [16/20], train loss = 0.042791, train accuracy = 98.88%, valid loss = 0.068882, valid accuracy = 97.95%, duration = 1.599476(s)
Epoch [17/20], train loss = 0.039967, train accuracy = 99.00%, valid loss = 0.071216, valid accuracy = 97.68%, duration = 1.669918(s)
Epoch [18/20], train loss = 0.037388, train accuracy = 99.08%, valid loss = 0.068723, valid accuracy = 97.70%, duration = 1.533915(s)
Epoch [19/20], train loss = 0.035062, train accuracy = 99.13%, valid loss = 0.065614, valid accuracy = 97.83%, duration = 1.489808(s)
Epoch [20/20], train loss = 0.032501, train accuracy = 99.25%, valid loss = 0.066377, valid accuracy = 97.92%, duration = 1.510935(s)
Duration for train : 33.132989(s)
<<< Train Finished >>>
Test Accraucy : 97.92%
'''
