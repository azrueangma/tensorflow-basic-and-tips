import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-1_board"
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
Epoch [ 1/30], train loss = 0.391200, train accuracy = 89.19%, valid loss = 0.206905, valid accuracy = 93.83%, duration = 2.047256(s)
Epoch [ 2/30], train loss = 0.189825, train accuracy = 94.48%, valid loss = 0.155754, valid accuracy = 95.47%, duration = 2.039024(s)
Epoch [ 3/30], train loss = 0.138167, train accuracy = 96.02%, valid loss = 0.115566, valid accuracy = 96.67%, duration = 1.976950(s)
Epoch [ 4/30], train loss = 0.108167, train accuracy = 96.86%, valid loss = 0.106460, valid accuracy = 96.77%, duration = 1.920490(s)
Epoch [ 5/30], train loss = 0.088485, train accuracy = 97.45%, valid loss = 0.094490, valid accuracy = 97.12%, duration = 1.969785(s)
Epoch [ 6/30], train loss = 0.074199, train accuracy = 97.89%, valid loss = 0.083705, valid accuracy = 97.45%, duration = 1.925236(s)
Epoch [ 7/30], train loss = 0.062464, train accuracy = 98.21%, valid loss = 0.082251, valid accuracy = 97.42%, duration = 1.951235(s)
Epoch [ 8/30], train loss = 0.053788, train accuracy = 98.50%, valid loss = 0.074143, valid accuracy = 97.65%, duration = 1.948768(s)
Epoch [ 9/30], train loss = 0.045690, train accuracy = 98.73%, valid loss = 0.077395, valid accuracy = 97.55%, duration = 1.958348(s)
Epoch [10/30], train loss = 0.040288, train accuracy = 98.91%, valid loss = 0.069944, valid accuracy = 97.90%, duration = 1.911232(s)
Epoch [11/30], train loss = 0.034152, train accuracy = 99.07%, valid loss = 0.067625, valid accuracy = 97.83%, duration = 1.947888(s)
Epoch [12/30], train loss = 0.029644, train accuracy = 99.21%, valid loss = 0.064568, valid accuracy = 97.97%, duration = 2.002140(s)
Epoch [13/30], train loss = 0.025415, train accuracy = 99.39%, valid loss = 0.066537, valid accuracy = 97.93%, duration = 2.216355(s)
Epoch [14/30], train loss = 0.021551, train accuracy = 99.51%, valid loss = 0.061325, valid accuracy = 98.07%, duration = 2.134767(s)
Epoch [15/30], train loss = 0.019087, train accuracy = 99.59%, valid loss = 0.062347, valid accuracy = 97.98%, duration = 2.034890(s)
Epoch [16/30], train loss = 0.016089, train accuracy = 99.69%, valid loss = 0.061046, valid accuracy = 98.07%, duration = 1.923226(s)
Epoch [17/30], train loss = 0.014163, train accuracy = 99.76%, valid loss = 0.065724, valid accuracy = 98.05%, duration = 2.003844(s)
Epoch [18/30], train loss = 0.011928, train accuracy = 99.82%, valid loss = 0.067204, valid accuracy = 97.92%, duration = 2.143125(s)
Epoch [19/30], train loss = 0.010480, train accuracy = 99.84%, valid loss = 0.061477, valid accuracy = 98.23%, duration = 2.101161(s)
Epoch [20/30], train loss = 0.009274, train accuracy = 99.89%, valid loss = 0.061197, valid accuracy = 98.12%, duration = 1.977715(s)
Epoch [21/30], train loss = 0.008154, train accuracy = 99.90%, valid loss = 0.060628, valid accuracy = 98.22%, duration = 1.988541(s)
Epoch [22/30], train loss = 0.007200, train accuracy = 99.92%, valid loss = 0.061830, valid accuracy = 98.13%, duration = 1.985063(s)
Epoch [23/30], train loss = 0.006194, train accuracy = 99.96%, valid loss = 0.062240, valid accuracy = 98.28%, duration = 2.038684(s)
Epoch [24/30], train loss = 0.005475, train accuracy = 99.95%, valid loss = 0.062303, valid accuracy = 98.27%, duration = 1.907300(s)
Epoch [25/30], train loss = 0.004891, train accuracy = 99.97%, valid loss = 0.061450, valid accuracy = 98.22%, duration = 1.991119(s)
Epoch [26/30], train loss = 0.004318, train accuracy = 99.99%, valid loss = 0.062940, valid accuracy = 98.27%, duration = 1.911819(s)
Epoch [27/30], train loss = 0.004037, train accuracy = 99.97%, valid loss = 0.062862, valid accuracy = 98.22%, duration = 1.999815(s)
Epoch [28/30], train loss = 0.003608, train accuracy = 99.98%, valid loss = 0.063467, valid accuracy = 98.23%, duration = 1.922049(s)
Epoch [29/30], train loss = 0.003268, train accuracy = 99.99%, valid loss = 0.064215, valid accuracy = 98.25%, duration = 1.932601(s)
Epoch [30/30], train loss = 0.002938, train accuracy = 99.99%, valid loss = 0.065118, valid accuracy = 98.23%, duration = 1.906694(s)
Duration for train : 60.672491(s)
<<< Train Finished >>>
Test Accraucy : 97.95%
'''
