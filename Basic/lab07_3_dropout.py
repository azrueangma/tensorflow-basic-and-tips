import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab07-3_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 20
KEEP_PROB = 0.7

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


def linear(tensor_op, output_dim, weight_decay = None, regularizer = None, with_W = False, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(tensor_op, W), b, name = 'h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)*weight_decay
            elif regularizer == 'l2':
                wd = l2_loss(W)*weight_decay
            else:
                wd = tf.constant(0.)
        else:
            wd = tf.constant(0.)

        tf.add_to_collection("weight_decay", wd)

        if with_W:
            return h, W
        else:
            return h


def relu_layer(tensor_op, output_dim, weight_decay = None, regularizer = None, keep_prob = 1.0, with_W = False, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[tensor_op.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(tensor_op, W), b), name = 'h')
        h = tf.nn.dropout(h, keep_prob=keep_prob, name='dropout_h')

        if weight_decay:
            if regularizer == 'l1':
                wd = l1_loss(W)*weight_decay
            elif regularizer == 'l2':
                wd = l2_loss(W)*weight_decay
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
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

h1 = relu_layer(X, 256, keep_prob = keep_prob, name = 'Relu_Layer1')
h2 = relu_layer(h1, 128, keep_prob = keep_prob, name = 'Relu_Layer2')
h3 = relu_layer(h2, 64, keep_prob = keep_prob, name = 'Relu_Layer3')
logits = linear(h1, NCLASS, name = 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    normal_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    weight_decay_loss = tf.get_collection("weight_decay")
    loss = normal_loss+tf.reduce_mean(weight_decay_loss)
    optim = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

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
            a, l, _ = sess.run([accuracy, loss, optim], feed_dict={X: x_train[mask[s:t],:], Y: y_train[mask[s:t],:], keep_prob : KEEP_PROB})
            loss_per_epoch += l
            acc_per_epoch += a
        epoch_end_time = time.perf_counter()
        epoch_duration = epoch_end_time - epoch_start_time
        loss_per_epoch /= total_step*BATCH_SIZE
        acc_per_epoch /= total_step*BATCH_SIZE

        va, vl = sess.run([accuracy, loss], feed_dict={X: x_validation, Y: y_validation, keep_prob : 1.0})
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

    ta = sess.run(accuracy, feed_dict = {X:x_test, Y:y_test, keep_prob : 1.0})
    print("Test Accraucy : {:.2%}".format(ta/ntest))

'''
Epoch [ 1/20], train loss = 0.277987, train accuracy = 91.95%, valid loss = 0.124472, valid accuracy = 96.25%, duration = 2.553065(s)
Epoch [ 2/20], train loss = 0.133117, train accuracy = 95.99%, valid loss = 0.097340, valid accuracy = 96.95%, duration = 2.464948(s)
Epoch [ 3/20], train loss = 0.099918, train accuracy = 96.92%, valid loss = 0.076867, valid accuracy = 97.55%, duration = 2.406296(s)
Epoch [ 4/20], train loss = 0.081288, train accuracy = 97.53%, valid loss = 0.076492, valid accuracy = 97.52%, duration = 2.439633(s)
Epoch [ 5/20], train loss = 0.068619, train accuracy = 97.81%, valid loss = 0.066035, valid accuracy = 97.83%, duration = 2.305432(s)
Epoch [ 6/20], train loss = 0.059843, train accuracy = 98.10%, valid loss = 0.059265, valid accuracy = 98.10%, duration = 2.252216(s)
Epoch [ 7/20], train loss = 0.052069, train accuracy = 98.30%, valid loss = 0.065850, valid accuracy = 97.97%, duration = 2.296361(s)
Epoch [ 8/20], train loss = 0.045053, train accuracy = 98.52%, valid loss = 0.066219, valid accuracy = 98.07%, duration = 2.291678(s)
Epoch [ 9/20], train loss = 0.043268, train accuracy = 98.56%, valid loss = 0.059345, valid accuracy = 98.23%, duration = 2.409814(s)
Epoch [10/20], train loss = 0.040199, train accuracy = 98.68%, valid loss = 0.060506, valid accuracy = 98.30%, duration = 2.184521(s)
Epoch [11/20], train loss = 0.034075, train accuracy = 98.87%, valid loss = 0.063404, valid accuracy = 98.10%, duration = 2.177903(s)
Epoch [12/20], train loss = 0.033103, train accuracy = 98.87%, valid loss = 0.063077, valid accuracy = 98.20%, duration = 2.106277(s)
Epoch [13/20], train loss = 0.031025, train accuracy = 98.97%, valid loss = 0.066112, valid accuracy = 98.17%, duration = 2.260172(s)
Epoch [14/20], train loss = 0.029759, train accuracy = 98.93%, valid loss = 0.057042, valid accuracy = 98.47%, duration = 2.140136(s)
Epoch [15/20], train loss = 0.028456, train accuracy = 99.04%, valid loss = 0.060951, valid accuracy = 98.32%, duration = 2.114386(s)
Epoch [16/20], train loss = 0.028305, train accuracy = 99.03%, valid loss = 0.064858, valid accuracy = 98.28%, duration = 2.097135(s)
Epoch [17/20], train loss = 0.025388, train accuracy = 99.15%, valid loss = 0.067400, valid accuracy = 98.28%, duration = 2.093158(s)
Epoch [18/20], train loss = 0.024719, train accuracy = 99.16%, valid loss = 0.064274, valid accuracy = 98.40%, duration = 2.221512(s)
Epoch [19/20], train loss = 0.024517, train accuracy = 99.19%, valid loss = 0.065565, valid accuracy = 98.45%, duration = 2.261557(s)
Epoch [20/20], train loss = 0.023141, train accuracy = 99.23%, valid loss = 0.066755, valid accuracy = 98.30%, duration = 2.264446(s)
Duration for train : 46.080287(s)
<<< Train Finished >>>
Test Accraucy : 98.29%
'''
