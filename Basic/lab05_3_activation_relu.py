import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-3_board"
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

def relu_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
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
Epoch [ 1/30], train loss = 5.746934, train accuracy = 59.38%, valid loss = 1.159476, valid accuracy = 67.18%, duration = 2.191546(s)
Epoch [ 2/30], train loss = 0.965263, train accuracy = 72.25%, valid loss = 0.898974, valid accuracy = 74.75%, duration = 2.825854(s)
Epoch [ 3/30], train loss = 0.788815, train accuracy = 78.29%, valid loss = 0.801158, valid accuracy = 79.17%, duration = 2.011775(s)
Epoch [ 4/30], train loss = 0.683601, train accuracy = 81.48%, valid loss = 0.735098, valid accuracy = 81.92%, duration = 1.959356(s)
Epoch [ 5/30], train loss = 0.608611, train accuracy = 83.60%, valid loss = 0.692484, valid accuracy = 83.13%, duration = 2.258870(s)
Epoch [ 6/30], train loss = 0.558529, train accuracy = 85.28%, valid loss = 0.622412, valid accuracy = 85.12%, duration = 2.145428(s)
Epoch [ 7/30], train loss = 0.520203, train accuracy = 86.30%, valid loss = 0.618835, valid accuracy = 87.08%, duration = 1.907766(s)
Epoch [ 8/30], train loss = 0.489437, train accuracy = 87.32%, valid loss = 0.566396, valid accuracy = 87.33%, duration = 1.869857(s)
Epoch [ 9/30], train loss = 0.459897, train accuracy = 88.17%, valid loss = 0.539618, valid accuracy = 88.55%, duration = 1.892115(s)
Epoch [10/30], train loss = 0.431693, train accuracy = 88.87%, valid loss = 0.522701, valid accuracy = 89.00%, duration = 1.878902(s)
Epoch [11/30], train loss = 0.417217, train accuracy = 89.37%, valid loss = 0.563255, valid accuracy = 88.33%, duration = 1.881990(s)
Epoch [12/30], train loss = 0.392644, train accuracy = 89.95%, valid loss = 0.486677, valid accuracy = 89.78%, duration = 1.898205(s)
Epoch [13/30], train loss = 0.375912, train accuracy = 90.39%, valid loss = 0.495527, valid accuracy = 89.98%, duration = 1.881390(s)
Epoch [14/30], train loss = 0.361785, train accuracy = 90.72%, valid loss = 0.472481, valid accuracy = 90.78%, duration = 2.107246(s)
Epoch [15/30], train loss = 0.347406, train accuracy = 91.15%, valid loss = 0.488194, valid accuracy = 90.48%, duration = 1.880125(s)
Epoch [16/30], train loss = 0.343628, train accuracy = 91.31%, valid loss = 0.462845, valid accuracy = 90.83%, duration = 1.948622(s)
Epoch [17/30], train loss = 0.325304, train accuracy = 91.63%, valid loss = 0.442175, valid accuracy = 91.05%, duration = 1.980216(s)
Epoch [18/30], train loss = 0.315200, train accuracy = 91.95%, valid loss = 0.451765, valid accuracy = 91.17%, duration = 1.904518(s)
Epoch [19/30], train loss = 0.306683, train accuracy = 92.16%, valid loss = 0.447739, valid accuracy = 91.50%, duration = 2.028176(s)
Epoch [20/30], train loss = 0.298326, train accuracy = 92.35%, valid loss = 0.433608, valid accuracy = 91.60%, duration = 1.991761(s)
Epoch [21/30], train loss = 0.292346, train accuracy = 92.60%, valid loss = 0.444939, valid accuracy = 92.02%, duration = 1.862434(s)
Epoch [22/30], train loss = 0.278712, train accuracy = 92.79%, valid loss = 0.419846, valid accuracy = 92.18%, duration = 2.014117(s)
Epoch [23/30], train loss = 0.273447, train accuracy = 92.91%, valid loss = 0.453525, valid accuracy = 92.13%, duration = 2.292199(s)
Epoch [24/30], train loss = 0.266053, train accuracy = 93.18%, valid loss = 0.419567, valid accuracy = 92.08%, duration = 2.005889(s)
Epoch [25/30], train loss = 0.262477, train accuracy = 93.21%, valid loss = 0.427642, valid accuracy = 92.42%, duration = 1.903828(s)
Epoch [26/30], train loss = 0.255882, train accuracy = 93.39%, valid loss = 0.427178, valid accuracy = 92.47%, duration = 1.851255(s)
Epoch [27/30], train loss = 0.247516, train accuracy = 93.69%, valid loss = 0.413283, valid accuracy = 92.73%, duration = 2.259445(s)
Epoch [28/30], train loss = 0.244011, train accuracy = 93.77%, valid loss = 0.409008, valid accuracy = 92.18%, duration = 2.464999(s)
Epoch [29/30], train loss = 0.239237, train accuracy = 93.81%, valid loss = 0.410599, valid accuracy = 92.70%, duration = 2.549650(s)
Epoch [30/30], train loss = 0.232432, train accuracy = 93.99%, valid loss = 0.407047, valid accuracy = 92.65%, duration = 2.260729(s)
Duration for train : 62.909069(s)
<<< Train Finished >>>
Test Accraucy : 92.28%
'''
