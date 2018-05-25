import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab05-2_board"
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

def tanh_layer(x, output_dim, name):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.tanh(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')
        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = tanh_layer(X, 256, 'Tanh_Layer1')
h2 = tanh_layer(h1, 128, 'Tanh_Layer2')
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
Epoch [ 1/30], train loss = 2.485900, train accuracy = 66.24%, valid loss = 1.225259, valid accuracy = 77.62%, duration = 2.007552(s)
Epoch [ 2/30], train loss = 0.950045, train accuracy = 80.95%, valid loss = 0.853107, valid accuracy = 82.18%, duration = 2.157960(s)
Epoch [ 3/30], train loss = 0.668515, train accuracy = 84.73%, valid loss = 0.697422, valid accuracy = 84.35%, duration = 1.981025(s)
Epoch [ 4/30], train loss = 0.509113, train accuracy = 87.27%, valid loss = 0.618265, valid accuracy = 85.07%, duration = 2.025876(s)
Epoch [ 5/30], train loss = 0.412780, train accuracy = 89.02%, valid loss = 0.574185, valid accuracy = 86.20%, duration = 2.208079(s)
Epoch [ 6/30], train loss = 0.348955, train accuracy = 90.41%, valid loss = 0.549086, valid accuracy = 86.62%, duration = 1.945884(s)
Epoch [ 7/30], train loss = 0.302929, train accuracy = 91.61%, valid loss = 0.530677, valid accuracy = 86.95%, duration = 2.012046(s)
Epoch [ 8/30], train loss = 0.270936, train accuracy = 92.36%, valid loss = 0.512290, valid accuracy = 87.23%, duration = 2.202202(s)
Epoch [ 9/30], train loss = 0.245870, train accuracy = 93.11%, valid loss = 0.505236, valid accuracy = 87.25%, duration = 2.048146(s)
Epoch [10/30], train loss = 0.226337, train accuracy = 93.71%, valid loss = 0.495588, valid accuracy = 87.57%, duration = 1.920010(s)
Epoch [11/30], train loss = 0.210111, train accuracy = 94.21%, valid loss = 0.486702, valid accuracy = 87.97%, duration = 1.907842(s)
Epoch [12/30], train loss = 0.196784, train accuracy = 94.66%, valid loss = 0.482826, valid accuracy = 88.02%, duration = 1.940166(s)
Epoch [13/30], train loss = 0.185308, train accuracy = 95.04%, valid loss = 0.480674, valid accuracy = 87.98%, duration = 1.906586(s)
Epoch [14/30], train loss = 0.175809, train accuracy = 95.38%, valid loss = 0.481390, valid accuracy = 87.83%, duration = 1.879914(s)
Epoch [15/30], train loss = 0.167179, train accuracy = 95.65%, valid loss = 0.475111, valid accuracy = 88.22%, duration = 1.940325(s)
Epoch [16/30], train loss = 0.159554, train accuracy = 95.90%, valid loss = 0.469236, valid accuracy = 88.40%, duration = 1.873565(s)
Epoch [17/30], train loss = 0.152680, train accuracy = 96.09%, valid loss = 0.466605, valid accuracy = 88.57%, duration = 1.881256(s)
Epoch [18/30], train loss = 0.146244, train accuracy = 96.34%, valid loss = 0.468838, valid accuracy = 88.45%, duration = 1.888357(s)
Epoch [19/30], train loss = 0.140768, train accuracy = 96.44%, valid loss = 0.462438, valid accuracy = 88.55%, duration = 1.926085(s)
Epoch [20/30], train loss = 0.135431, train accuracy = 96.65%, valid loss = 0.465327, valid accuracy = 88.48%, duration = 1.881180(s)
Epoch [21/30], train loss = 0.130398, train accuracy = 96.75%, valid loss = 0.462073, valid accuracy = 88.47%, duration = 1.916800(s)
Epoch [22/30], train loss = 0.126275, train accuracy = 96.89%, valid loss = 0.463507, valid accuracy = 88.57%, duration = 1.911122(s)
Epoch [23/30], train loss = 0.121841, train accuracy = 97.01%, valid loss = 0.462337, valid accuracy = 88.75%, duration = 1.900969(s)
Epoch [24/30], train loss = 0.118043, train accuracy = 97.13%, valid loss = 0.467047, valid accuracy = 88.65%, duration = 1.877305(s)
Epoch [25/30], train loss = 0.114845, train accuracy = 97.19%, valid loss = 0.466686, valid accuracy = 88.73%, duration = 1.903540(s)
Epoch [26/30], train loss = 0.111226, train accuracy = 97.31%, valid loss = 0.466012, valid accuracy = 88.68%, duration = 1.888158(s)
Epoch [27/30], train loss = 0.108131, train accuracy = 97.41%, valid loss = 0.464483, valid accuracy = 88.88%, duration = 1.906968(s)
Epoch [28/30], train loss = 0.105065, train accuracy = 97.50%, valid loss = 0.468788, valid accuracy = 88.98%, duration = 1.901833(s)
Epoch [29/30], train loss = 0.102135, train accuracy = 97.59%, valid loss = 0.464822, valid accuracy = 88.83%, duration = 1.894278(s)
Epoch [30/30], train loss = 0.099258, train accuracy = 97.73%, valid loss = 0.469149, valid accuracy = 88.80%, duration = 1.955158(s)
Duration for train : 59.537941(s)
<<< Train Finished >>>
Test Accraucy : 88.42%
'''
