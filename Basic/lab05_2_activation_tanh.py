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
h3 = relu_layer(h2, 64, 'Relu_Layer3')
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
Epoch [ 1/20], train loss = 5.047006, train accuracy = 84.31%, valid loss = 2.113284, valid accuracy = 89.73%, duration = 1.758740(s)
Epoch [ 2/20], train loss = 1.582572, train accuracy = 91.23%, valid loss = 1.392293, valid accuracy = 91.67%, duration = 1.812762(s)
Epoch [ 3/20], train loss = 1.023508, train accuracy = 92.93%, valid loss = 1.079414, valid accuracy = 92.77%, duration = 1.535274(s)
Epoch [ 4/20], train loss = 0.750445, train accuracy = 93.98%, valid loss = 0.960322, valid accuracy = 93.32%, duration = 1.451760(s)
Epoch [ 5/20], train loss = 0.592148, train accuracy = 94.64%, valid loss = 0.854444, valid accuracy = 93.20%, duration = 1.535733(s)
Epoch [ 6/20], train loss = 0.481099, train accuracy = 95.22%, valid loss = 0.794937, valid accuracy = 93.48%, duration = 1.428715(s)
Epoch [ 7/20], train loss = 0.401590, train accuracy = 95.49%, valid loss = 0.784200, valid accuracy = 93.60%, duration = 1.524452(s)
Epoch [ 8/20], train loss = 0.340243, train accuracy = 95.94%, valid loss = 0.733861, valid accuracy = 93.93%, duration = 1.488272(s)
Epoch [ 9/20], train loss = 0.291899, train accuracy = 96.26%, valid loss = 0.689117, valid accuracy = 93.80%, duration = 1.421864(s)
Epoch [10/20], train loss = 0.253879, train accuracy = 96.52%, valid loss = 0.688275, valid accuracy = 93.77%, duration = 1.517622(s)
Epoch [11/20], train loss = 0.222661, train accuracy = 96.79%, valid loss = 0.633492, valid accuracy = 93.80%, duration = 1.504183(s)
Epoch [12/20], train loss = 0.199457, train accuracy = 97.04%, valid loss = 0.603214, valid accuracy = 94.20%, duration = 1.401704(s)
Epoch [13/20], train loss = 0.175629, train accuracy = 97.18%, valid loss = 0.616244, valid accuracy = 94.13%, duration = 1.527559(s)
Epoch [14/20], train loss = 0.155970, train accuracy = 97.33%, valid loss = 0.593933, valid accuracy = 94.10%, duration = 1.435107(s)
Epoch [15/20], train loss = 0.141206, train accuracy = 97.51%, valid loss = 0.577776, valid accuracy = 94.27%, duration = 1.467548(s)
Epoch [16/20], train loss = 0.124347, train accuracy = 97.79%, valid loss = 0.540225, valid accuracy = 94.48%, duration = 1.476709(s)
Epoch [17/20], train loss = 0.113709, train accuracy = 97.82%, valid loss = 0.583097, valid accuracy = 94.23%, duration = 1.409378(s)
Epoch [18/20], train loss = 0.102924, train accuracy = 97.99%, valid loss = 0.566344, valid accuracy = 94.22%, duration = 1.486548(s)
Epoch [19/20], train loss = 0.090667, train accuracy = 98.20%, valid loss = 0.544119, valid accuracy = 94.58%, duration = 1.498093(s)
Epoch [20/20], train loss = 0.084100, train accuracy = 98.23%, valid loss = 0.537123, valid accuracy = 94.58%, duration = 1.362935(s)
Duration for train : 30.576716(s)
<<< Train Finished >>>
Test Accraucy : 94.90%
'''
