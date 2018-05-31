import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab07-2_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 30
ALPHA = 0.001

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)


def l1_loss (tensor_op, name='l1_loss'):
    output = tf.reduce_sum(tf.abs(tensor_op), name = name)
    return output


def l2_loss (tensor_op, name='l2_loss'):
    output = tf.reduce_sum(tf.square(tensor_op), name = name)/2
    return output


def linear(tensor_op, output_dim, weight_decay = False, regularizer = None, with_W = False, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name = 'h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)
            elif regularizer == 'l2':
                wd = l2_loss(W)
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


def relu_layer(tensor_op, output_dim, weight_decay = False, regularizer = None, with_W = False, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(tensor_op, W), b), name = 'h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)
            elif regularizer == 'l2':
                wd = l2_loss(W)
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = relu_layer(tensor_op=X, output_dim=256, weight_decay=True, regularizer='l2', name='Relu_Layer1')
h2 = relu_layer(tensor_op=h1, output_dim=128, weight_decay=True, regularizer='l2', name='Relu_Layer2')
logits = linear(tensor_op=h2, output_dim=NCLASS, weight_decay=True, regularizer='l2', name='Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    normal_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss + ALPHA*tf.reduce_mean(weight_decay_loss)
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
        loss_per_epoch /= total_step
        acc_per_epoch /= total_step*BATCH_SIZE

        va, vl = sess.run([accuracy, normal_loss], feed_dict={X: x_validation, Y: y_validation})
        epoch_valid_acc = va / len(x_validation)
        epoch_valid_loss = vl

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
Epoch [ 1/30], train loss = 0.296664, train accuracy = 93.36%, valid loss = 0.116459, valid accuracy = 96.23%, duration = 2.790567(s)
Epoch [ 2/30], train loss = 0.176731, train accuracy = 96.93%, valid loss = 0.090287, valid accuracy = 97.28%, duration = 2.741277(s)
Epoch [ 3/30], train loss = 0.155373, train accuracy = 97.60%, valid loss = 0.083723, valid accuracy = 97.60%, duration = 2.648611(s)
Epoch [ 4/30], train loss = 0.140272, train accuracy = 98.00%, valid loss = 0.101427, valid accuracy = 96.88%, duration = 2.665409(s)
Epoch [ 5/30], train loss = 0.131754, train accuracy = 98.11%, valid loss = 0.066378, valid accuracy = 97.85%, duration = 2.866709(s)
Epoch [ 6/30], train loss = 0.123586, train accuracy = 98.27%, valid loss = 0.071955, valid accuracy = 97.93%, duration = 2.659445(s)
Epoch [ 7/30], train loss = 0.116882, train accuracy = 98.43%, valid loss = 0.077666, valid accuracy = 97.70%, duration = 2.586551(s)
Epoch [ 8/30], train loss = 0.112505, train accuracy = 98.57%, valid loss = 0.063127, valid accuracy = 98.17%, duration = 2.737396(s)
Epoch [ 9/30], train loss = 0.110911, train accuracy = 98.51%, valid loss = 0.075176, valid accuracy = 97.52%, duration = 3.005164(s)
Epoch [10/30], train loss = 0.108130, train accuracy = 98.62%, valid loss = 0.071412, valid accuracy = 97.77%, duration = 2.763052(s)
Epoch [11/30], train loss = 0.104534, train accuracy = 98.68%, valid loss = 0.071091, valid accuracy = 97.73%, duration = 2.723475(s)
Epoch [12/30], train loss = 0.103170, train accuracy = 98.71%, valid loss = 0.065978, valid accuracy = 97.87%, duration = 2.760110(s)
Epoch [13/30], train loss = 0.101106, train accuracy = 98.74%, valid loss = 0.076819, valid accuracy = 97.73%, duration = 2.974164(s)
Epoch [14/30], train loss = 0.099668, train accuracy = 98.78%, valid loss = 0.061953, valid accuracy = 97.92%, duration = 2.655274(s)
Epoch [15/30], train loss = 0.098444, train accuracy = 98.82%, valid loss = 0.073158, valid accuracy = 97.83%, duration = 2.634627(s)
Epoch [16/30], train loss = 0.097916, train accuracy = 98.77%, valid loss = 0.101934, valid accuracy = 96.95%, duration = 2.680435(s)
Epoch [17/30], train loss = 0.097161, train accuracy = 98.89%, valid loss = 0.075286, valid accuracy = 97.63%, duration = 2.581523(s)
Epoch [18/30], train loss = 0.095622, train accuracy = 98.87%, valid loss = 0.079962, valid accuracy = 97.48%, duration = 2.665355(s)
Epoch [19/30], train loss = 0.096852, train accuracy = 98.86%, valid loss = 0.060467, valid accuracy = 98.03%, duration = 2.690839(s)
Epoch [20/30], train loss = 0.093392, train accuracy = 98.94%, valid loss = 0.084937, valid accuracy = 97.25%, duration = 2.805560(s)
Epoch [21/30], train loss = 0.093089, train accuracy = 98.97%, valid loss = 0.073198, valid accuracy = 97.88%, duration = 2.664357(s)
Epoch [22/30], train loss = 0.094903, train accuracy = 98.86%, valid loss = 0.061464, valid accuracy = 98.10%, duration = 2.793271(s)
Epoch [23/30], train loss = 0.092312, train accuracy = 98.91%, valid loss = 0.073885, valid accuracy = 97.73%, duration = 2.612739(s)
Epoch [24/30], train loss = 0.093031, train accuracy = 98.94%, valid loss = 0.065158, valid accuracy = 97.98%, duration = 2.816111(s)
Epoch [25/30], train loss = 0.091995, train accuracy = 98.98%, valid loss = 0.065153, valid accuracy = 98.00%, duration = 2.627904(s)
Epoch [26/30], train loss = 0.092225, train accuracy = 98.90%, valid loss = 0.061330, valid accuracy = 98.00%, duration = 2.777530(s)
Epoch [27/30], train loss = 0.091864, train accuracy = 98.94%, valid loss = 0.062736, valid accuracy = 98.02%, duration = 2.775600(s)
Epoch [28/30], train loss = 0.090751, train accuracy = 98.98%, valid loss = 0.063746, valid accuracy = 97.95%, duration = 2.687927(s)
Epoch [29/30], train loss = 0.091406, train accuracy = 99.02%, valid loss = 0.067385, valid accuracy = 97.80%, duration = 2.648910(s)
Epoch [30/30], train loss = 0.090198, train accuracy = 98.97%, valid loss = 0.075372, valid accuracy = 97.77%, duration = 2.728706(s)
Duration for train : 82.698247(s)
<<< Train Finished >>>
Test Accraucy : 97.66%
'''
