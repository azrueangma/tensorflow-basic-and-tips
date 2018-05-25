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
Epoch [ 1/30], train loss = 0.213735, train accuracy = 93.57%, valid loss = 0.109157, valid accuracy = 96.43%, duration = 2.962748(s)
Epoch [ 2/30], train loss = 0.085296, train accuracy = 97.36%, valid loss = 0.098166, valid accuracy = 96.90%, duration = 2.641274(s)
Epoch [ 3/30], train loss = 0.060438, train accuracy = 98.10%, valid loss = 0.077855, valid accuracy = 97.65%, duration = 2.654980(s)
Epoch [ 4/30], train loss = 0.041866, train accuracy = 98.66%, valid loss = 0.086001, valid accuracy = 97.57%, duration = 2.394998(s)
Epoch [ 5/30], train loss = 0.035443, train accuracy = 98.88%, valid loss = 0.085433, valid accuracy = 97.58%, duration = 2.377375(s)
Epoch [ 6/30], train loss = 0.028238, train accuracy = 99.11%, valid loss = 0.074392, valid accuracy = 97.88%, duration = 2.427880(s)
Epoch [ 7/30], train loss = 0.023589, train accuracy = 99.21%, valid loss = 0.075423, valid accuracy = 98.03%, duration = 2.414600(s)
Epoch [ 8/30], train loss = 0.021337, train accuracy = 99.28%, valid loss = 0.086280, valid accuracy = 97.82%, duration = 2.836345(s)
Epoch [ 9/30], train loss = 0.020348, train accuracy = 99.38%, valid loss = 0.102846, valid accuracy = 97.65%, duration = 2.618658(s)
Epoch [10/30], train loss = 0.016758, train accuracy = 99.44%, valid loss = 0.094068, valid accuracy = 97.92%, duration = 2.820987(s)
Epoch [11/30], train loss = 0.013501, train accuracy = 99.53%, valid loss = 0.080476, valid accuracy = 98.08%, duration = 2.411096(s)
Epoch [12/30], train loss = 0.015117, train accuracy = 99.53%, valid loss = 0.101463, valid accuracy = 97.97%, duration = 2.398073(s)
Epoch [13/30], train loss = 0.015709, train accuracy = 99.54%, valid loss = 0.090927, valid accuracy = 97.77%, duration = 2.531626(s)
Epoch [14/30], train loss = 0.011842, train accuracy = 99.61%, valid loss = 0.076369, valid accuracy = 98.30%, duration = 2.413743(s)
Epoch [15/30], train loss = 0.010004, train accuracy = 99.67%, valid loss = 0.092016, valid accuracy = 98.08%, duration = 2.434885(s)
Epoch [16/30], train loss = 0.012560, train accuracy = 99.62%, valid loss = 0.110126, valid accuracy = 98.02%, duration = 2.481502(s)
Epoch [17/30], train loss = 0.010256, train accuracy = 99.70%, valid loss = 0.109444, valid accuracy = 98.33%, duration = 2.567255(s)
Epoch [18/30], train loss = 0.009650, train accuracy = 99.68%, valid loss = 0.095525, valid accuracy = 98.05%, duration = 2.344650(s)
Epoch [19/30], train loss = 0.012532, train accuracy = 99.65%, valid loss = 0.094103, valid accuracy = 98.20%, duration = 2.520452(s)
Epoch [20/30], train loss = 0.008858, train accuracy = 99.73%, valid loss = 0.117161, valid accuracy = 97.88%, duration = 2.866034(s)
Epoch [21/30], train loss = 0.008994, train accuracy = 99.75%, valid loss = 0.097378, valid accuracy = 98.40%, duration = 2.562642(s)
Epoch [22/30], train loss = 0.007633, train accuracy = 99.78%, valid loss = 0.115618, valid accuracy = 98.13%, duration = 2.527980(s)
Epoch [23/30], train loss = 0.007557, train accuracy = 99.74%, valid loss = 0.133337, valid accuracy = 98.08%, duration = 2.425477(s)
Epoch [24/30], train loss = 0.009209, train accuracy = 99.72%, valid loss = 0.097057, valid accuracy = 98.30%, duration = 2.607092(s)
Epoch [25/30], train loss = 0.007880, train accuracy = 99.80%, valid loss = 0.103123, valid accuracy = 98.33%, duration = 2.494292(s)
Epoch [26/30], train loss = 0.010115, train accuracy = 99.68%, valid loss = 0.121100, valid accuracy = 98.25%, duration = 2.522030(s)
Epoch [27/30], train loss = 0.006335, train accuracy = 99.79%, valid loss = 0.131420, valid accuracy = 98.20%, duration = 2.531390(s)
Epoch [28/30], train loss = 0.008148, train accuracy = 99.73%, valid loss = 0.131089, valid accuracy = 98.23%, duration = 2.462994(s)
Epoch [29/30], train loss = 0.008172, train accuracy = 99.79%, valid loss = 0.115839, valid accuracy = 98.20%, duration = 2.525103(s)
Epoch [30/30], train loss = 0.007014, train accuracy = 99.81%, valid loss = 0.127447, valid accuracy = 98.12%, duration = 2.462534(s)
Duration for train : 77.212637(s)
<<< Train Finished >>>
Test Accraucy : 97.91%
'''
