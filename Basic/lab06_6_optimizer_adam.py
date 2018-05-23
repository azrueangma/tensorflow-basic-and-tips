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
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def relu_linear(x, output_dim, name):
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

h1 = relu_linear(X, 256, 'Relu_Layer1')
h2 = relu_linear(h1, 128, 'Relu_Layer2')
h3 = relu_linear(h2, 64, 'Relu_Layer3')
logits = linear(h1, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
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
Epoch [ 1/20], train loss = 0.234712, train accuracy = 93.22%, valid loss = 0.116754, valid accuracy = 96.57%, duration = 2.048825(s)
Epoch [ 2/20], train loss = 0.095600, train accuracy = 97.17%, valid loss = 0.091591, valid accuracy = 97.27%, duration = 1.835122(s)
Epoch [ 3/20], train loss = 0.064977, train accuracy = 98.05%, valid loss = 0.070056, valid accuracy = 98.05%, duration = 1.973189(s)
Epoch [ 4/20], train loss = 0.045281, train accuracy = 98.61%, valid loss = 0.080030, valid accuracy = 97.73%, duration = 1.811311(s)
Epoch [ 5/20], train loss = 0.033107, train accuracy = 98.95%, valid loss = 0.071732, valid accuracy = 97.90%, duration = 1.945029(s)
Epoch [ 6/20], train loss = 0.025535, train accuracy = 99.21%, valid loss = 0.061824, valid accuracy = 98.08%, duration = 1.786534(s)
Epoch [ 7/20], train loss = 0.020210, train accuracy = 99.35%, valid loss = 0.075915, valid accuracy = 98.05%, duration = 1.846409(s)
Epoch [ 8/20], train loss = 0.014740, train accuracy = 99.56%, valid loss = 0.077617, valid accuracy = 98.00%, duration = 1.913558(s)
Epoch [ 9/20], train loss = 0.013724, train accuracy = 99.56%, valid loss = 0.084861, valid accuracy = 97.82%, duration = 1.740336(s)
Epoch [10/20], train loss = 0.011343, train accuracy = 99.61%, valid loss = 0.074473, valid accuracy = 98.15%, duration = 2.172245(s)
Epoch [11/20], train loss = 0.008796, train accuracy = 99.74%, valid loss = 0.070792, valid accuracy = 98.20%, duration = 1.747205(s)
Epoch [12/20], train loss = 0.009249, train accuracy = 99.70%, valid loss = 0.075104, valid accuracy = 98.12%, duration = 1.731380(s)
Epoch [13/20], train loss = 0.006792, train accuracy = 99.79%, valid loss = 0.094774, valid accuracy = 97.85%, duration = 1.931400(s)
Epoch [14/20], train loss = 0.008149, train accuracy = 99.73%, valid loss = 0.082002, valid accuracy = 98.18%, duration = 1.722156(s)
Epoch [15/20], train loss = 0.006918, train accuracy = 99.77%, valid loss = 0.106239, valid accuracy = 97.62%, duration = 1.878853(s)
Epoch [16/20], train loss = 0.005536, train accuracy = 99.81%, valid loss = 0.093041, valid accuracy = 97.93%, duration = 1.851785(s)
Epoch [17/20], train loss = 0.006034, train accuracy = 99.78%, valid loss = 0.111318, valid accuracy = 97.60%, duration = 1.710436(s)
Epoch [18/20], train loss = 0.005599, train accuracy = 99.80%, valid loss = 0.089959, valid accuracy = 98.00%, duration = 1.987297(s)
Epoch [19/20], train loss = 0.005968, train accuracy = 99.79%, valid loss = 0.111651, valid accuracy = 98.12%, duration = 1.834395(s)
Epoch [20/20], train loss = 0.005103, train accuracy = 99.82%, valid loss = 0.098503, valid accuracy = 98.15%, duration = 1.818242(s)
Duration for train : 37.819932(s)
<<< Train Finished >>>
Test Accraucy : 97.92%
'''
