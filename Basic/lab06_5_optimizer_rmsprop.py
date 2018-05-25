import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-5_board"
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
    optim = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

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
Epoch [ 1/30], train loss = 0.241427, train accuracy = 92.85%, valid loss = 0.103817, valid accuracy = 96.73%, duration = 2.412888(s)
Epoch [ 2/30], train loss = 0.096001, train accuracy = 97.17%, valid loss = 0.076999, valid accuracy = 97.85%, duration = 2.353587(s)
Epoch [ 3/30], train loss = 0.071158, train accuracy = 97.94%, valid loss = 0.084586, valid accuracy = 97.80%, duration = 2.363700(s)
Epoch [ 4/30], train loss = 0.057313, train accuracy = 98.35%, valid loss = 0.084081, valid accuracy = 97.88%, duration = 2.318659(s)
Epoch [ 5/30], train loss = 0.046245, train accuracy = 98.82%, valid loss = 0.088938, valid accuracy = 98.07%, duration = 2.320213(s)
Epoch [ 6/30], train loss = 0.040825, train accuracy = 98.94%, valid loss = 0.100651, valid accuracy = 97.65%, duration = 2.262049(s)
Epoch [ 7/30], train loss = 0.033553, train accuracy = 99.09%, valid loss = 0.109298, valid accuracy = 98.05%, duration = 2.420105(s)
Epoch [ 8/30], train loss = 0.029705, train accuracy = 99.18%, valid loss = 0.110102, valid accuracy = 98.07%, duration = 2.405832(s)
Epoch [ 9/30], train loss = 0.026193, train accuracy = 99.31%, valid loss = 0.115050, valid accuracy = 98.05%, duration = 2.652251(s)
Epoch [10/30], train loss = 0.022338, train accuracy = 99.42%, valid loss = 0.148437, valid accuracy = 97.63%, duration = 2.777555(s)
Epoch [11/30], train loss = 0.022140, train accuracy = 99.42%, valid loss = 0.144219, valid accuracy = 97.85%, duration = 2.612223(s)
Epoch [12/30], train loss = 0.018794, train accuracy = 99.51%, valid loss = 0.170245, valid accuracy = 98.05%, duration = 2.300327(s)
Epoch [13/30], train loss = 0.017731, train accuracy = 99.54%, valid loss = 0.141303, valid accuracy = 98.15%, duration = 2.408960(s)
Epoch [14/30], train loss = 0.015267, train accuracy = 99.64%, valid loss = 0.163647, valid accuracy = 98.10%, duration = 2.327752(s)
Epoch [15/30], train loss = 0.014291, train accuracy = 99.66%, valid loss = 0.166425, valid accuracy = 98.02%, duration = 2.298824(s)
Epoch [16/30], train loss = 0.013376, train accuracy = 99.70%, valid loss = 0.192583, valid accuracy = 97.92%, duration = 2.303622(s)
Epoch [17/30], train loss = 0.012934, train accuracy = 99.68%, valid loss = 0.189960, valid accuracy = 97.97%, duration = 2.334790(s)
Epoch [18/30], train loss = 0.011358, train accuracy = 99.70%, valid loss = 0.199359, valid accuracy = 98.10%, duration = 2.315284(s)
Epoch [19/30], train loss = 0.011117, train accuracy = 99.73%, valid loss = 0.177550, valid accuracy = 98.27%, duration = 2.335351(s)
Epoch [20/30], train loss = 0.009777, train accuracy = 99.77%, valid loss = 0.213369, valid accuracy = 97.70%, duration = 2.291833(s)
Epoch [21/30], train loss = 0.007951, train accuracy = 99.81%, valid loss = 0.197483, valid accuracy = 98.22%, duration = 2.308565(s)
Epoch [22/30], train loss = 0.009759, train accuracy = 99.79%, valid loss = 0.210251, valid accuracy = 97.98%, duration = 2.262918(s)
Epoch [23/30], train loss = 0.009428, train accuracy = 99.80%, valid loss = 0.274447, valid accuracy = 97.97%, duration = 2.354780(s)
Epoch [24/30], train loss = 0.008428, train accuracy = 99.82%, valid loss = 0.239048, valid accuracy = 98.02%, duration = 2.307826(s)
Epoch [25/30], train loss = 0.008867, train accuracy = 99.83%, valid loss = 0.211760, valid accuracy = 98.27%, duration = 2.348239(s)
Epoch [26/30], train loss = 0.005795, train accuracy = 99.87%, valid loss = 0.235043, valid accuracy = 98.13%, duration = 2.475296(s)
Epoch [27/30], train loss = 0.004705, train accuracy = 99.89%, valid loss = 0.262197, valid accuracy = 98.28%, duration = 2.346701(s)
Epoch [28/30], train loss = 0.006468, train accuracy = 99.86%, valid loss = 0.251852, valid accuracy = 98.10%, duration = 2.292846(s)
Epoch [29/30], train loss = 0.005144, train accuracy = 99.89%, valid loss = 0.278634, valid accuracy = 97.90%, duration = 2.277360(s)
Epoch [30/30], train loss = 0.006187, train accuracy = 99.87%, valid loss = 0.270442, valid accuracy = 98.03%, duration = 2.256593(s)
Duration for train : 72.036336(s)
<<< Train Finished >>>
Test Accraucy : 98.05%
'''
