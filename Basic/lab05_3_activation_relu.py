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
Epoch [ 1/30], train loss = 6.897840, train accuracy = 43.78%, valid loss = 1.408888, valid accuracy = 55.55%, duration = 2.270923(s)
Epoch [ 2/30], train loss = 1.246453, train accuracy = 61.62%, valid loss = 1.089077, valid accuracy = 66.12%, duration = 2.032974(s)
Epoch [ 3/30], train loss = 0.999155, train accuracy = 69.90%, valid loss = 0.924934, valid accuracy = 74.38%, duration = 2.217774(s)
Epoch [ 4/30], train loss = 0.849422, train accuracy = 75.75%, valid loss = 0.802086, valid accuracy = 79.12%, duration = 2.279086(s)
Epoch [ 5/30], train loss = 0.730460, train accuracy = 79.92%, valid loss = 1.003233, valid accuracy = 74.78%, duration = 1.965374(s)
Epoch [ 6/30], train loss = 0.663855, train accuracy = 82.22%, valid loss = 0.640471, valid accuracy = 83.82%, duration = 2.090388(s)
Epoch [ 7/30], train loss = 0.606882, train accuracy = 83.94%, valid loss = 0.628202, valid accuracy = 85.18%, duration = 2.049568(s)
Epoch [ 8/30], train loss = 0.558959, train accuracy = 85.42%, valid loss = 0.640486, valid accuracy = 84.65%, duration = 1.946121(s)
Epoch [ 9/30], train loss = 0.528996, train accuracy = 86.37%, valid loss = 0.556753, valid accuracy = 86.70%, duration = 2.041809(s)
Epoch [10/30], train loss = 0.501799, train accuracy = 87.20%, valid loss = 0.535889, valid accuracy = 87.82%, duration = 1.970776(s)
Epoch [11/30], train loss = 0.485280, train accuracy = 87.69%, valid loss = 0.609567, valid accuracy = 79.43%, duration = 2.191088(s)
Epoch [12/30], train loss = 0.447334, train accuracy = 88.65%, valid loss = 0.529101, valid accuracy = 88.48%, duration = 1.913964(s)
Epoch [13/30], train loss = 0.420071, train accuracy = 89.36%, valid loss = 0.471799, valid accuracy = 88.72%, duration = 1.924522(s)
Epoch [14/30], train loss = 0.400565, train accuracy = 89.79%, valid loss = 0.521478, valid accuracy = 89.57%, duration = 2.034466(s)
Epoch [15/30], train loss = 0.389494, train accuracy = 90.10%, valid loss = 0.485577, valid accuracy = 89.67%, duration = 1.937604(s)
Epoch [16/30], train loss = 0.370105, train accuracy = 90.66%, valid loss = 0.469056, valid accuracy = 90.05%, duration = 2.057356(s)
Epoch [17/30], train loss = 0.356467, train accuracy = 90.98%, valid loss = 0.472904, valid accuracy = 89.82%, duration = 1.958299(s)
Epoch [18/30], train loss = 0.346081, train accuracy = 91.23%, valid loss = 0.443859, valid accuracy = 90.23%, duration = 1.905295(s)
Epoch [19/30], train loss = 0.340504, train accuracy = 91.41%, valid loss = 0.452074, valid accuracy = 90.75%, duration = 2.224430(s)
Epoch [20/30], train loss = 0.331539, train accuracy = 91.65%, valid loss = 0.421831, valid accuracy = 90.82%, duration = 1.897673(s)
Epoch [21/30], train loss = 0.315513, train accuracy = 92.11%, valid loss = 0.401559, valid accuracy = 91.53%, duration = 1.926986(s)
Epoch [22/30], train loss = 0.308919, train accuracy = 92.17%, valid loss = 0.406486, valid accuracy = 91.72%, duration = 1.895089(s)
Epoch [23/30], train loss = 0.296927, train accuracy = 92.51%, valid loss = 0.508247, valid accuracy = 91.10%, duration = 1.886732(s)
Epoch [24/30], train loss = 0.293291, train accuracy = 92.55%, valid loss = 0.408236, valid accuracy = 91.95%, duration = 1.922647(s)
Epoch [25/30], train loss = 0.285061, train accuracy = 92.83%, valid loss = 0.410004, valid accuracy = 91.80%, duration = 1.912449(s)
Epoch [26/30], train loss = 0.275619, train accuracy = 93.06%, valid loss = 0.374263, valid accuracy = 92.30%, duration = 1.948252(s)
Epoch [27/30], train loss = 0.265141, train accuracy = 93.26%, valid loss = 0.495502, valid accuracy = 86.50%, duration = 2.037372(s)
Epoch [28/30], train loss = 0.262416, train accuracy = 93.41%, valid loss = 0.380602, valid accuracy = 92.28%, duration = 2.051364(s)
Epoch [29/30], train loss = 0.259214, train accuracy = 93.50%, valid loss = 0.389632, valid accuracy = 92.30%, duration = 2.021027(s)
Epoch [30/30], train loss = 0.248908, train accuracy = 93.68%, valid loss = 0.390219, valid accuracy = 92.33%, duration = 1.989671(s)
Duration for train : 61.491187(s)
<<< Train Finished >>>
Test Accraucy : 92.40%
'''
