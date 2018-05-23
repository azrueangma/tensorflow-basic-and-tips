#lab04_4 model
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
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def leaky_relu_linear(x, output_dim, name):
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

h1 = leaky_relu_linear(X, 256, 'Relu_Layer1')
h2 = leaky_relu_linear(h1, 128, 'Relu_Layer2')
h3 = leaky_relu_linear(h2, 64, 'Relu_Layer3')
logits = linear(h1, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    optim = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss)

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
        x_trian = x_train[mask]
        epoch_start_time = time.perf_counter()
        for step in range(total_step):
            s = BATCH_SIZE*step
            t = BATCH_SIZE*(step+1)
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[s:t,:], Y: y_train[s:t,:]})
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
Epoch [ 1/20], train loss = 5.287030, train accuracy = 84.33%, valid loss = 2.166616, valid accuracy = 89.72%, duration = 1.608738(s)
Epoch [ 2/20], train loss = 1.842554, train accuracy = 90.61%, valid loss = 1.513280, valid accuracy = 90.98%, duration = 1.575553(s)
Epoch [ 3/20], train loss = 1.202983, train accuracy = 92.11%, valid loss = 1.122095, valid accuracy = 91.97%, duration = 1.525033(s)
Epoch [ 4/20], train loss = 0.892778, train accuracy = 93.06%, valid loss = 0.991229, valid accuracy = 92.18%, duration = 1.581703(s)
Epoch [ 5/20], train loss = 0.706640, train accuracy = 93.80%, valid loss = 0.880721, valid accuracy = 92.62%, duration = 1.434678(s)
Epoch [ 6/20], train loss = 0.583306, train accuracy = 94.36%, valid loss = 0.792353, valid accuracy = 92.85%, duration = 1.507684(s)
Epoch [ 7/20], train loss = 0.492285, train accuracy = 94.84%, valid loss = 0.733190, valid accuracy = 93.02%, duration = 1.467358(s)
Epoch [ 8/20], train loss = 0.423036, train accuracy = 95.22%, valid loss = 0.696091, valid accuracy = 93.33%, duration = 1.460734(s)
Epoch [ 9/20], train loss = 0.371189, train accuracy = 95.48%, valid loss = 0.645756, valid accuracy = 93.52%, duration = 1.547518(s)
Epoch [10/20], train loss = 0.326683, train accuracy = 95.85%, valid loss = 0.619340, valid accuracy = 93.60%, duration = 1.511745(s)
Epoch [11/20], train loss = 0.291570, train accuracy = 96.13%, valid loss = 0.599451, valid accuracy = 93.70%, duration = 1.512438(s)
Epoch [12/20], train loss = 0.262187, train accuracy = 96.40%, valid loss = 0.577124, valid accuracy = 93.92%, duration = 1.462207(s)
Epoch [13/20], train loss = 0.237198, train accuracy = 96.64%, valid loss = 0.553106, valid accuracy = 93.95%, duration = 1.435974(s)
Epoch [14/20], train loss = 0.215821, train accuracy = 96.78%, valid loss = 0.531194, valid accuracy = 94.08%, duration = 1.508642(s)
Epoch [15/20], train loss = 0.196706, train accuracy = 97.00%, valid loss = 0.521413, valid accuracy = 94.03%, duration = 1.402196(s)
Epoch [16/20], train loss = 0.180431, train accuracy = 97.14%, valid loss = 0.515644, valid accuracy = 94.07%, duration = 1.512515(s)
Epoch [17/20], train loss = 0.165527, train accuracy = 97.33%, valid loss = 0.506311, valid accuracy = 94.25%, duration = 1.446111(s)
Epoch [18/20], train loss = 0.152568, train accuracy = 97.47%, valid loss = 0.499792, valid accuracy = 94.27%, duration = 1.470276(s)
Epoch [19/20], train loss = 0.140688, train accuracy = 97.60%, valid loss = 0.490919, valid accuracy = 94.25%, duration = 1.494094(s)
Epoch [20/20], train loss = 0.129872, train accuracy = 97.77%, valid loss = 0.480053, valid accuracy = 94.33%, duration = 1.425433(s)
Duration for train : 32.568390(s)
<<< Train Finished >>>
Test Accraucy : 94.17%
'''