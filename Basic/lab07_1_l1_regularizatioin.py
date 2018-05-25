import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab07-1_board"
INPUT_DIM = np.size(x_train, 1)
NCLASS = len(np.unique(y_train))
BATCH_SIZE = 32

TOTAL_EPOCH = 20
ALPHA = 0.001

ntrain = len(x_train)
nvalidation = len(x_validation)
ntest = len(x_test)

print("The number of train samples : ", ntrain)
print("The number of validation samples : ", nvalidation)
print("The number of test samples : ", ntest)

def l1_loss (x, name='l1_loss'):
    output = tf.reduce_sum(tf.abs(x), name = name)
    return output

def linear(x, output_dim, weight_decay = None, name='linear'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')

        if weight_decay:
            wd = l1_loss(W)*weight_decay
            tf.add_to_collection("weight_decay", wd)

        return h

def relu_layer(x, output_dim, weight_decay = None, name='relu_layer'):
    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=[x.get_shape()[-1], output_dim], dtype=tf.float32,
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W), b), name = 'h')

        if weight_decay:
            wd = l1_loss(W)*weight_decay
            tf.add_to_collection("weight_decay", wd)

        return h

tf.set_random_seed(0)

with tf.variable_scope("Inputs"):
    X = tf.placeholder(shape = [None, INPUT_DIM], dtype = tf.float32, name = 'X')
    Y = tf.placeholder(shape = [None, 1], dtype = tf.int32, name = 'Y')
    Y_one_hot = tf.reshape(tf.one_hot(Y, NCLASS), [-1, NCLASS], name = 'Y_one_hot')

h1 = relu_layer(X, 256, ALPHA, 'Relu_Layer1')
h2 = relu_layer(h1, 128, ALPHA, 'Relu_Layer2')
h3 = relu_layer(h2, 64, ALPHA, 'Relu_Layer3')
logits = linear(h1, NCLASS, ALPHA, 'Linear_Layer')

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
Total step :  1687
Epoch [ 1/20], train loss = 1.248558, train accuracy = 88.46%, valid loss = 0.316609, valid accuracy = 91.63%, duration = 2.299932(s)
Epoch [ 2/20], train loss = 0.691139, train accuracy = 91.96%, valid loss = 0.271549, valid accuracy = 92.78%, duration = 2.616167(s)
Epoch [ 3/20], train loss = 0.587506, train accuracy = 93.09%, valid loss = 0.228051, valid accuracy = 94.07%, duration = 2.404973(s)
Epoch [ 4/20], train loss = 0.539525, train accuracy = 93.68%, valid loss = 0.219030, valid accuracy = 94.35%, duration = 2.205827(s)
Epoch [ 5/20], train loss = 0.515728, train accuracy = 93.92%, valid loss = 0.207298, valid accuracy = 94.37%, duration = 2.251844(s)
Epoch [ 6/20], train loss = 0.499629, train accuracy = 94.23%, valid loss = 0.206907, valid accuracy = 93.90%, duration = 2.182254(s)
Epoch [ 7/20], train loss = 0.487073, train accuracy = 94.36%, valid loss = 0.191067, valid accuracy = 94.58%, duration = 2.239696(s)
Epoch [ 8/20], train loss = 0.478346, train accuracy = 94.50%, valid loss = 0.190820, valid accuracy = 94.98%, duration = 2.168924(s)
Epoch [ 9/20], train loss = 0.473007, train accuracy = 94.59%, valid loss = 0.191065, valid accuracy = 95.07%, duration = 2.233405(s)
Epoch [10/20], train loss = 0.466792, train accuracy = 94.73%, valid loss = 0.182667, valid accuracy = 95.03%, duration = 2.234211(s)
Epoch [11/20], train loss = 0.463222, train accuracy = 94.75%, valid loss = 0.181295, valid accuracy = 95.40%, duration = 2.233730(s)
Epoch [12/20], train loss = 0.460844, train accuracy = 94.90%, valid loss = 0.176587, valid accuracy = 95.28%, duration = 2.218329(s)
Epoch [13/20], train loss = 0.459355, train accuracy = 94.80%, valid loss = 0.180321, valid accuracy = 95.03%, duration = 2.229120(s)
Epoch [14/20], train loss = 0.457416, train accuracy = 94.96%, valid loss = 0.182471, valid accuracy = 95.10%, duration = 2.241327(s)
Epoch [15/20], train loss = 0.456408, train accuracy = 94.86%, valid loss = 0.182931, valid accuracy = 94.73%, duration = 2.213933(s)
Epoch [16/20], train loss = 0.454828, train accuracy = 94.85%, valid loss = 0.186382, valid accuracy = 95.15%, duration = 2.250180(s)
Epoch [17/20], train loss = 0.452696, train accuracy = 94.94%, valid loss = 0.176436, valid accuracy = 95.25%, duration = 2.240983(s)
Epoch [18/20], train loss = 0.451154, train accuracy = 94.95%, valid loss = 0.173702, valid accuracy = 95.20%, duration = 2.250513(s)
Epoch [19/20], train loss = 0.450600, train accuracy = 94.87%, valid loss = 0.172803, valid accuracy = 95.33%, duration = 2.255956(s)
Epoch [20/20], train loss = 0.446478, train accuracy = 95.02%, valid loss = 0.177849, valid accuracy = 95.10%, duration = 2.235796(s)
Duration for train : 45.719216(s)
<<< Train Finished >>>
Test Accraucy : 95.24% 
'''