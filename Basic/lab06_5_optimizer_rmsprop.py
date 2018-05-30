import tensorflow as tf
import numpy as np
import os
import shutil
import load_data
import time

x_train, x_validation, x_test, y_train, y_validation, y_test \
    = load_data.load_mnist('./data/mnist/', seed = 0, as_image = False, scaling = True)

BOARD_PATH = "./board/lab06-5_board"
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
                            initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(name = 'b', shape = [output_dim], dtype = tf.float32, initializer= tf.constant_initializer(0.0))
        h = tf.nn.bias_add(tf.matmul(x, W), b, name = 'h')
        return h

def relu_layer(x, output_dim, name):
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

h1 = relu_layer(X, 256, 'Relu_Layer1')
h2 = relu_layer(h1, 128, 'Relu_Layer2')
logits = linear(h2, NCLASS, 'Linear_Layer')

with tf.variable_scope("Optimization"):
    hypothesis = tf.nn.softmax(logits, name = 'hypothesis')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y_one_hot), name = 'loss')
    optim = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss)

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
Epoch [ 1/30], train loss = 0.241936, train accuracy = 92.86%, valid loss = 0.102949, valid accuracy = 96.67%, duration = 2.415835(s)
Epoch [ 2/30], train loss = 0.096041, train accuracy = 97.16%, valid loss = 0.079668, valid accuracy = 97.52%, duration = 2.272773(s)
Epoch [ 3/30], train loss = 0.071553, train accuracy = 97.98%, valid loss = 0.092990, valid accuracy = 97.73%, duration = 2.299408(s)
Epoch [ 4/30], train loss = 0.058516, train accuracy = 98.40%, valid loss = 0.093918, valid accuracy = 97.63%, duration = 2.249051(s)
Epoch [ 5/30], train loss = 0.048831, train accuracy = 98.70%, valid loss = 0.100318, valid accuracy = 97.67%, duration = 2.287207(s)
Epoch [ 6/30], train loss = 0.040317, train accuracy = 98.95%, valid loss = 0.086201, valid accuracy = 97.98%, duration = 2.261579(s)
Epoch [ 7/30], train loss = 0.035471, train accuracy = 99.07%, valid loss = 0.123834, valid accuracy = 97.87%, duration = 2.280343(s)
Epoch [ 8/30], train loss = 0.031668, train accuracy = 99.19%, valid loss = 0.096506, valid accuracy = 98.20%, duration = 2.273980(s)
Epoch [ 9/30], train loss = 0.026894, train accuracy = 99.26%, valid loss = 0.113580, valid accuracy = 97.95%, duration = 2.240327(s)
Epoch [10/30], train loss = 0.027197, train accuracy = 99.33%, valid loss = 0.130210, valid accuracy = 98.00%, duration = 2.299568(s)
Epoch [11/30], train loss = 0.021670, train accuracy = 99.43%, valid loss = 0.132793, valid accuracy = 98.02%, duration = 2.240916(s)
Epoch [12/30], train loss = 0.019106, train accuracy = 99.51%, valid loss = 0.188990, valid accuracy = 97.65%, duration = 2.284445(s)
Epoch [13/30], train loss = 0.017270, train accuracy = 99.57%, valid loss = 0.173700, valid accuracy = 97.90%, duration = 2.257956(s)
Epoch [14/30], train loss = 0.018192, train accuracy = 99.60%, valid loss = 0.157364, valid accuracy = 98.12%, duration = 2.272805(s)
Epoch [15/30], train loss = 0.014079, train accuracy = 99.66%, valid loss = 0.166196, valid accuracy = 98.13%, duration = 2.249708(s)
Epoch [16/30], train loss = 0.013541, train accuracy = 99.69%, valid loss = 0.196104, valid accuracy = 97.88%, duration = 2.280041(s)
Epoch [17/30], train loss = 0.012015, train accuracy = 99.71%, valid loss = 0.192319, valid accuracy = 98.03%, duration = 2.248087(s)
Epoch [18/30], train loss = 0.012486, train accuracy = 99.71%, valid loss = 0.233703, valid accuracy = 97.52%, duration = 2.274584(s)
Epoch [19/30], train loss = 0.011841, train accuracy = 99.70%, valid loss = 0.192905, valid accuracy = 97.85%, duration = 2.287104(s)
Epoch [20/30], train loss = 0.010506, train accuracy = 99.76%, valid loss = 0.215779, valid accuracy = 97.77%, duration = 2.260860(s)
Epoch [21/30], train loss = 0.009897, train accuracy = 99.77%, valid loss = 0.200152, valid accuracy = 98.07%, duration = 2.268610(s)
Epoch [22/30], train loss = 0.008089, train accuracy = 99.82%, valid loss = 0.230956, valid accuracy = 97.98%, duration = 2.254467(s)
Epoch [23/30], train loss = 0.007151, train accuracy = 99.80%, valid loss = 0.270788, valid accuracy = 97.77%, duration = 2.295596(s)
Epoch [24/30], train loss = 0.008200, train accuracy = 99.83%, valid loss = 0.238224, valid accuracy = 97.93%, duration = 2.377891(s)
Epoch [25/30], train loss = 0.008002, train accuracy = 99.83%, valid loss = 0.235746, valid accuracy = 98.15%, duration = 2.284709(s)
Epoch [26/30], train loss = 0.007730, train accuracy = 99.83%, valid loss = 0.213068, valid accuracy = 98.23%, duration = 2.288685(s)
Epoch [27/30], train loss = 0.005659, train accuracy = 99.86%, valid loss = 0.226246, valid accuracy = 98.20%, duration = 2.304888(s)
Epoch [28/30], train loss = 0.007042, train accuracy = 99.86%, valid loss = 0.241869, valid accuracy = 97.87%, duration = 2.238082(s)
Epoch [29/30], train loss = 0.003855, train accuracy = 99.91%, valid loss = 0.273931, valid accuracy = 98.10%, duration = 2.302503(s)
Epoch [30/30], train loss = 0.006451, train accuracy = 99.87%, valid loss = 0.219903, valid accuracy = 98.08%, duration = 2.253575(s)
Duration for train : 69.336134(s)
<<< Train Finished >>>
Test Accraucy : 98.07%
'''
