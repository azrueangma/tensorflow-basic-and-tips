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

def leaky_relu_layer(x, output_dim, name):
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

h1 = leaky_relu_layer(X, 256, 'Relu_Layer1')
h2 = leaky_relu_layer(h1, 128, 'Relu_Layer2')
h3 = leaky_relu_layer(h2, 64, 'Relu_Layer3')
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
Epoch [ 1/20], train loss = 5.280309, train accuracy = 84.43%, valid loss = 2.192719, valid accuracy = 89.73%, duration = 1.562534(s)
Epoch [ 2/20], train loss = 1.835254, train accuracy = 90.45%, valid loss = 1.471184, valid accuracy = 91.30%, duration = 1.599910(s)
Epoch [ 3/20], train loss = 1.216222, train accuracy = 92.13%, valid loss = 1.120655, valid accuracy = 91.83%, duration = 1.628009(s)
Epoch [ 4/20], train loss = 0.908780, train accuracy = 92.99%, valid loss = 0.916593, valid accuracy = 92.85%, duration = 1.598603(s)
Epoch [ 5/20], train loss = 0.723199, train accuracy = 93.56%, valid loss = 0.854424, valid accuracy = 92.37%, duration = 1.646868(s)
Epoch [ 6/20], train loss = 0.601546, train accuracy = 94.10%, valid loss = 0.753575, valid accuracy = 93.05%, duration = 1.503455(s)
Epoch [ 7/20], train loss = 0.500931, train accuracy = 94.55%, valid loss = 0.782419, valid accuracy = 92.65%, duration = 1.649394(s)
Epoch [ 8/20], train loss = 0.438810, train accuracy = 94.95%, valid loss = 0.655866, valid accuracy = 93.60%, duration = 1.683563(s)
Epoch [ 9/20], train loss = 0.379123, train accuracy = 95.27%, valid loss = 0.678308, valid accuracy = 92.92%, duration = 1.734282(s)
Epoch [10/20], train loss = 0.338763, train accuracy = 95.46%, valid loss = 0.580765, valid accuracy = 93.58%, duration = 1.606895(s)
Epoch [11/20], train loss = 0.303422, train accuracy = 95.67%, valid loss = 0.588278, valid accuracy = 93.57%, duration = 1.620437(s)
Epoch [12/20], train loss = 0.270334, train accuracy = 96.01%, valid loss = 0.546297, valid accuracy = 93.83%, duration = 1.585566(s)
Epoch [13/20], train loss = 0.247943, train accuracy = 96.17%, valid loss = 0.503139, valid accuracy = 94.22%, duration = 1.602130(s)
Epoch [14/20], train loss = 0.223866, train accuracy = 96.45%, valid loss = 0.541635, valid accuracy = 93.80%, duration = 1.487964(s)
Epoch [15/20], train loss = 0.204678, train accuracy = 96.63%, valid loss = 0.521802, valid accuracy = 93.78%, duration = 1.581846(s)
Epoch [16/20], train loss = 0.186970, train accuracy = 96.75%, valid loss = 0.469911, valid accuracy = 94.33%, duration = 1.561904(s)
Epoch [17/20], train loss = 0.171293, train accuracy = 96.96%, valid loss = 0.489764, valid accuracy = 94.10%, duration = 1.498289(s)
Epoch [18/20], train loss = 0.157536, train accuracy = 97.15%, valid loss = 0.459903, valid accuracy = 94.00%, duration = 1.615673(s)
Epoch [19/20], train loss = 0.151098, train accuracy = 97.19%, valid loss = 0.428736, valid accuracy = 94.73%, duration = 1.480815(s)
Epoch [20/20], train loss = 0.136671, train accuracy = 97.36%, valid loss = 0.439614, valid accuracy = 94.33%, duration = 1.440080(s)
Duration for train : 32.272964(s)
<<< Train Finished >>>
Test Accraucy : 94.34%
'''
