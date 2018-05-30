import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-3_board"
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
    optim = tf.train.AdadeltaOptimizer(learning_rate=0.001).minimize(loss)

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
Epoch [ 1/30], train loss = 2.327560, train accuracy = 10.02%, valid loss = 2.297815, valid accuracy = 11.95%, duration = 2.757069(s)
Epoch [ 2/30], train loss = 2.267391, train accuracy = 15.08%, valid loss = 2.238810, valid accuracy = 18.33%, duration = 2.622594(s)
Epoch [ 3/30], train loss = 2.211309, train accuracy = 22.67%, valid loss = 2.183256, valid accuracy = 26.93%, duration = 2.452420(s)
Epoch [ 4/30], train loss = 2.157355, train accuracy = 31.22%, valid loss = 2.129349, valid accuracy = 35.23%, duration = 2.415933(s)
Epoch [ 5/30], train loss = 2.104612, train accuracy = 39.18%, valid loss = 2.076387, valid accuracy = 42.78%, duration = 2.499866(s)
Epoch [ 6/30], train loss = 2.052266, train accuracy = 45.89%, valid loss = 2.023351, valid accuracy = 49.48%, duration = 2.439644(s)
Epoch [ 7/30], train loss = 1.999722, train accuracy = 51.73%, valid loss = 1.970045, valid accuracy = 55.15%, duration = 2.461540(s)
Epoch [ 8/30], train loss = 1.946351, train accuracy = 56.69%, valid loss = 1.915764, valid accuracy = 59.92%, duration = 2.609246(s)
Epoch [ 9/30], train loss = 1.892085, train accuracy = 60.83%, valid loss = 1.860414, valid accuracy = 62.90%, duration = 2.567406(s)
Epoch [10/30], train loss = 1.836822, train accuracy = 64.13%, valid loss = 1.804187, valid accuracy = 65.68%, duration = 2.535159(s)
Epoch [11/30], train loss = 1.780862, train accuracy = 66.71%, valid loss = 1.747222, valid accuracy = 67.70%, duration = 2.537318(s)
Epoch [12/30], train loss = 1.724268, train accuracy = 68.89%, valid loss = 1.689665, valid accuracy = 69.73%, duration = 2.458575(s)
Epoch [13/30], train loss = 1.667353, train accuracy = 70.62%, valid loss = 1.632033, valid accuracy = 71.10%, duration = 2.595829(s)
Epoch [14/30], train loss = 1.610295, train accuracy = 71.95%, valid loss = 1.574181, valid accuracy = 72.47%, duration = 2.589915(s)
Epoch [15/30], train loss = 1.553410, train accuracy = 73.08%, valid loss = 1.517055, valid accuracy = 73.67%, duration = 2.581246(s)
Epoch [16/30], train loss = 1.497584, train accuracy = 74.06%, valid loss = 1.460994, valid accuracy = 74.45%, duration = 3.427026(s)
Epoch [17/30], train loss = 1.442686, train accuracy = 74.91%, valid loss = 1.406246, valid accuracy = 75.20%, duration = 2.837950(s)
Epoch [18/30], train loss = 1.389179, train accuracy = 75.68%, valid loss = 1.352949, valid accuracy = 76.10%, duration = 2.491344(s)
Epoch [19/30], train loss = 1.337276, train accuracy = 76.42%, valid loss = 1.301424, valid accuracy = 76.85%, duration = 2.648328(s)
Epoch [20/30], train loss = 1.287092, train accuracy = 77.13%, valid loss = 1.251814, valid accuracy = 77.52%, duration = 2.746750(s)
Epoch [21/30], train loss = 1.238755, train accuracy = 77.63%, valid loss = 1.204264, valid accuracy = 78.10%, duration = 2.845559(s)
Epoch [22/30], train loss = 1.192690, train accuracy = 78.22%, valid loss = 1.159271, valid accuracy = 78.92%, duration = 2.606690(s)
Epoch [23/30], train loss = 1.149098, train accuracy = 78.79%, valid loss = 1.116502, valid accuracy = 79.45%, duration = 2.747112(s)
Epoch [24/30], train loss = 1.107713, train accuracy = 79.30%, valid loss = 1.076173, valid accuracy = 79.88%, duration = 2.584341(s)
Epoch [25/30], train loss = 1.068533, train accuracy = 79.76%, valid loss = 1.037938, valid accuracy = 80.33%, duration = 2.440548(s)
Epoch [26/30], train loss = 1.031595, train accuracy = 80.24%, valid loss = 1.002021, valid accuracy = 80.73%, duration = 2.357974(s)
Epoch [27/30], train loss = 0.996832, train accuracy = 80.70%, valid loss = 0.968314, valid accuracy = 81.20%, duration = 2.347183(s)
Epoch [28/30], train loss = 0.964157, train accuracy = 81.14%, valid loss = 0.936551, valid accuracy = 81.72%, duration = 2.444421(s)
Epoch [29/30], train loss = 0.933340, train accuracy = 81.51%, valid loss = 0.906670, valid accuracy = 82.02%, duration = 2.387978(s)
Epoch [30/30], train loss = 0.904463, train accuracy = 81.87%, valid loss = 0.878695, valid accuracy = 82.20%, duration = 2.573460(s)
Duration for train : 78.578189(s)
<<< Train Finished >>>
Test Accraucy : 83.37%
'''
