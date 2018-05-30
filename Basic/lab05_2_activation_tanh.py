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
Epoch [ 1/30], train loss = 2.505256, train accuracy = 65.47%, valid loss = 1.233229, valid accuracy = 77.35%, duration = 1.958387(s)
Epoch [ 2/30], train loss = 0.968461, train accuracy = 80.70%, valid loss = 0.880809, valid accuracy = 81.78%, duration = 1.875191(s)
Epoch [ 3/30], train loss = 0.677243, train accuracy = 84.64%, valid loss = 0.732032, valid accuracy = 83.60%, duration = 1.910498(s)
Epoch [ 4/30], train loss = 0.513359, train accuracy = 87.10%, valid loss = 0.648984, valid accuracy = 84.83%, duration = 1.874952(s)
Epoch [ 5/30], train loss = 0.415024, train accuracy = 88.95%, valid loss = 0.601293, valid accuracy = 85.08%, duration = 1.885796(s)
Epoch [ 6/30], train loss = 0.347161, train accuracy = 90.35%, valid loss = 0.574875, valid accuracy = 85.87%, duration = 1.934423(s)
Epoch [ 7/30], train loss = 0.301742, train accuracy = 91.49%, valid loss = 0.555363, valid accuracy = 86.18%, duration = 1.916083(s)
Epoch [ 8/30], train loss = 0.268963, train accuracy = 92.36%, valid loss = 0.538057, valid accuracy = 86.67%, duration = 1.894375(s)
Epoch [ 9/30], train loss = 0.243644, train accuracy = 93.16%, valid loss = 0.536855, valid accuracy = 86.37%, duration = 1.925898(s)
Epoch [10/30], train loss = 0.223696, train accuracy = 93.69%, valid loss = 0.522837, valid accuracy = 86.80%, duration = 1.928324(s)
Epoch [11/30], train loss = 0.207990, train accuracy = 94.15%, valid loss = 0.514210, valid accuracy = 86.98%, duration = 1.886923(s)
Epoch [12/30], train loss = 0.194217, train accuracy = 94.59%, valid loss = 0.506669, valid accuracy = 87.12%, duration = 1.880988(s)
Epoch [13/30], train loss = 0.182866, train accuracy = 94.94%, valid loss = 0.504079, valid accuracy = 87.28%, duration = 1.903431(s)
Epoch [14/30], train loss = 0.172927, train accuracy = 95.24%, valid loss = 0.504549, valid accuracy = 87.32%, duration = 1.876582(s)
Epoch [15/30], train loss = 0.163929, train accuracy = 95.50%, valid loss = 0.500628, valid accuracy = 87.22%, duration = 2.050264(s)
Epoch [16/30], train loss = 0.156534, train accuracy = 95.70%, valid loss = 0.493900, valid accuracy = 87.63%, duration = 1.922688(s)
Epoch [17/30], train loss = 0.149353, train accuracy = 95.88%, valid loss = 0.495986, valid accuracy = 87.57%, duration = 2.089236(s)
Epoch [18/30], train loss = 0.142688, train accuracy = 96.11%, valid loss = 0.493244, valid accuracy = 87.78%, duration = 2.061254(s)
Epoch [19/30], train loss = 0.136955, train accuracy = 96.31%, valid loss = 0.492629, valid accuracy = 87.73%, duration = 1.908718(s)
Epoch [20/30], train loss = 0.131405, train accuracy = 96.45%, valid loss = 0.490661, valid accuracy = 87.92%, duration = 1.932689(s)
Epoch [21/30], train loss = 0.126103, train accuracy = 96.65%, valid loss = 0.489180, valid accuracy = 87.87%, duration = 1.884352(s)
Epoch [22/30], train loss = 0.121505, train accuracy = 96.77%, valid loss = 0.492534, valid accuracy = 87.88%, duration = 1.856726(s)
Epoch [23/30], train loss = 0.117051, train accuracy = 96.94%, valid loss = 0.495520, valid accuracy = 87.88%, duration = 1.858989(s)
Epoch [24/30], train loss = 0.112852, train accuracy = 97.06%, valid loss = 0.492448, valid accuracy = 88.10%, duration = 1.845955(s)
Epoch [25/30], train loss = 0.108864, train accuracy = 97.20%, valid loss = 0.492244, valid accuracy = 88.22%, duration = 1.917238(s)
Epoch [26/30], train loss = 0.104797, train accuracy = 97.35%, valid loss = 0.494899, valid accuracy = 87.87%, duration = 2.051098(s)
Epoch [27/30], train loss = 0.101765, train accuracy = 97.42%, valid loss = 0.492917, valid accuracy = 88.13%, duration = 1.982755(s)
Epoch [28/30], train loss = 0.098449, train accuracy = 97.53%, valid loss = 0.492881, valid accuracy = 88.08%, duration = 1.968945(s)
Epoch [29/30], train loss = 0.095261, train accuracy = 97.60%, valid loss = 0.494490, valid accuracy = 88.32%, duration = 2.029819(s)
Epoch [30/30], train loss = 0.092203, train accuracy = 97.73%, valid loss = 0.496779, valid accuracy = 88.10%, duration = 1.998969(s)
Duration for train : 58.960399(s)
<<< Train Finished >>>
Test Accraucy : 88.64%
'''
